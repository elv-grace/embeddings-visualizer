"""Face-embedding worker, run inside the isolated celeb environment.

Why a separate process
----------------------
model-celeb-vector needs numpy<1.20, torch 1.9 and mxnet 1.9, which cannot
coexist with the SigLIP 2 / Qwen stack the service itself runs on (torch 2.13,
numpy 2.x). So the tagger's model runs in its own interpreter and the service
talks to it over a pipe. Nothing here is a reimplementation: it imports
`CelebVectorizer` and calls the same `tag_frame` the tagger uses.

Protocol
--------
One JSON request per line on stdin, one JSON response per line on stdout:

    {"image_path": "/tmp/x.png"}  ->  {"vector": [...], "box": {...}, "faces": 2}
                                  ->  {"error": "no face detected"}

Long-lived on purpose: loading r100 costs a couple of seconds, so the process is
started once and kept, exactly as the in-process towers keep their weights
resident. stdout carries only protocol; the tagger's own logging goes to stderr
so it cannot corrupt a response.
"""

import json
import os
import sys


def _log(message):
    print(message, file=sys.stderr, flush=True)


def _largest(faces):
    """The biggest detected face: a query photo often has bystanders, and the
    subject is the one the uploader framed."""
    def area(face):
        box = getattr(face, "box", None) or {}
        try:
            return max(0.0, float(box["x2"]) - float(box["x1"])) * max(
                0.0, float(box["y2"]) - float(box["y1"])
            )
        except (KeyError, TypeError, ValueError):
            return 0.0

    return max(faces, key=area)


def main():
    # Both repos on the path: celeb_vector imports common_ml, and its own module
    # adds model-celeb so `celeb.face_model` resolves.
    for var in ("CELEB_EMBEDDING_PATH", "COMMON_ML_PATH"):
        path = os.environ.get(var)
        if path and path not in sys.path:
            sys.path.insert(0, path)

    import cv2
    import numpy as np

    from celeb_vector.config import RuntimeConfig
    from celeb_vector.model import CelebVectorizer

    params = json.loads(os.environ.get("CELEB_PARAMS") or "{}")
    cfg_kwargs = {
        k: params[k]
        for k in ("det_confidence", "min_box_size")
        if params.get(k) is not None
    }
    weights = os.environ.get("CELEB_MODEL_PATH") or "/ml/models/celeb"

    vectorizer = CelebVectorizer(model_input_path=weights, cfg=RuntimeConfig(**cfg_kwargs))
    # The service waits for this before sending anything, so a failure above is
    # reported as a startup error rather than as a timeout on the first query.
    print(json.dumps({"ready": True}), flush=True)
    _log("celeb worker ready")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            image = cv2.imread(request["image_path"])
            if image is None:
                raise ValueError(f"could not read {request['image_path']}")
            # cv2 reads BGR; the tagger expects RGB, as its frames arrive.
            faces = vectorizer.tag_frame(np.ascontiguousarray(image[:, :, ::-1]))
            if not faces:
                response = {"error": "no face detected"}
            else:
                best = _largest(faces)
                response = {
                    "vector": [float(x) for x in best.vector],
                    "box": getattr(best, "box", None),
                    "faces": len(faces),
                }
        except Exception as exc:               # noqa: BLE001 - reported, not raised
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
