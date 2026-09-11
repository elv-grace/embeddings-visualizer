# embeddings-visualizer

Plots the vectors of an Eluvio vectorstore index as a 2D latent space, with
search: a text, image or video query is embedded into the same space and drawn
beside its true nearest neighbours. Runs standalone or as a plug-in inside
elv-core-js.

Built with Claude Code.

---

## Setup

```bash
pip install -r requirements.txt
python3 src/app.py                 # http://localhost:8099
tools/setup_celeb_env.sh
```

That is everything needed to **load and plot** any index. Querying needs the
embedding model that built the index, and each comes with its own cost:

| index `model` | query modes | extra setup |
|---|---|---|
| `frame_vectors` (SigLIP 2) | text, image | none |
| `video_vectors` (Qwen3-VL) | text, image, video | packages listed in `requirements.txt`; first query downloads 16.3 GB and takes ~45 s to load |
| `face_vectors` (InsightFace) | image | `tools/setup_celeb_env.sh` (~1.5 GB) |

Nothing is installed automatically — the service never shells out to pip. A
missing dependency surfaces on the first *query*, not on load, and the error
names the module and the path it looked in.

`EV_PORT` and `EV_LOG_FILE` override the port and the log (default
`logs/visualizer.log`, rotating).

### Why face queries need their own interpreter

model-celeb-vector pins `numpy<1.20` and `torch 1.9`, which cannot coexist with
the torch 2.x that SigLIP 2 and Qwen need. So `tools/setup_celeb_env.sh` builds
`.celebenv/` (Python 3.8, via `uv`) and pip-installs the tagger's own source
into it; `src/celeb_embedder.py` starts `tools/celeb_worker.py` in that
interpreter and talks to it over a pipe. The bridge was checked not to change
the vector: max abs diff `0.0` against calling the tagger directly.

It also needs the **InsightFace r100 weights**, which the script does not
install. Note `config.yml` names `/ml/models/celeb_detection`, but they actually
live at `/ml/models/celeb`; `CELEB_MODEL_PATH` overrides it.

---

## How it works

**The index is enumerated, not scanned.** The vectorstore has no scan endpoint —
every read is a KNN around a reference vector. But its filters are
*pre-filtered*, so once a time window holds fewer rows than `limit`, the top-K
over it *is* the whole window. Walking the timeline in windows therefore
enumerates the index exactly.

**Only a sample carries vectors.** A vector is ~12 KB of JSON, so the counting
pass runs without them (~150 B/row) and only the sample is refetched in full.
The browser receives 2D coordinates plus metadata, never vectors.

**The 2D positions are layout, never a measurement.** Any projection distorts
neighbourhoods. Similarity is ranked by cosine in the *original* space and drawn
as explicit links from the query to its true neighbours — read similarity off
the links and the ranked list, not off the distances. PCA→50→UMAP→2 by default;
PCA alone is available but retains only ~20% of the variance.

**A query node is placed among its matches, not projected.** Every query vector
collapses onto the same "nowhere" point UMAP maps noise to — a consequence of
the modality gap, where a text query sits at cosine ~0.03–0.08 from *every*
indexed vector. So a query is drawn at the weighted centroid of its neighbours
instead. 

Clicking a query node lists and pins its links to 10 closest neighbors, and accessing any of them reveals a Back button to the query node top results page. If several query nodes are pinned and share a closest neighbor: the query node you arrived from right before is preferred. Otherwise, it returns to the first pinned query that links to the node (in ascending order).  

**Query towers import the taggers' own code** rather than reimplementing it, so
a query cannot drift from the vectors it is searching. model-frame-vector v3
made the patch budget and normalize *fixed constants* in the tagger rather than
per-call arguments; when an index was built with different values (its stamped
`additional_info` says so), the query falls back to this repo's own tower, which
honours them and is bit-identical (`max abs diff 0.0`).

**The model comes from the batch.** Each row names its `batch_id`;
`GET /indexes/{qid}/batches/{batch_id}` returns that batch's `model`, which maps
1-1 to a query embedder in `embedder.INDEX_MODELS`. A tagger's stamped
`additional_info` is read only for *tuning* parameters (patch budget, sampling
budget, MRL width) and never chooses a model.

### Layout

| | |
|---|---|
| `src/app.py` | HTTP service; serves the API and the frontend from one origin |
| `src/vectors_api.py` | enumerates and samples an index out of the vectorstore |
| `src/projection.py` | PCA → UMAP to 2D, plus the query's neighbour anchoring |
| `src/embedder.py` | model → embedder registry, and the tuning parameters |
| `src/siglip2_embedder.py` · `qwen_embedder.py` · `celeb_embedder.py` | the query towers |
| `web/app.js` | deck.gl layers, queries, legend, detail panel |
| `web/media.js` | metadata → a frame image (`frame_extract` instead of `frame`) or a playable clip |
| `web/index.html` | header controls, the help panel, and the stage's overlays |
| `web/style.css`  | the dark palette, including the modality colours `app.js` mirrors |
| `web/vendor/`    | elv-client-js's prebuilt `FrameClient` (used only inside core) |

Two endpoints: `POST /api/index` loads, samples and projects;
`POST /api/search/<mode>` embeds a query and ranks it. Everything else is static.

---

## Media playout

A vector's metadata names a content object (`qid`) and a position on its
timeline. Turning that into something watchable needs a URL authorized against
that object.

**One transport, two token sources.** Every URL is built the same way — a fabric
node plus `?authorization=<token>` — and only the token's origin differs: core
mints one, or standalone the viewer pastes one. So the path that runs in core is
the path exercised standalone.

**The token must be object-scoped.** `CreateSignedToken({objectId, grantType:
"read"})` mints `aessjc…` carrying `qid`/`lib`/`gra`, which works.
`CreateFabricToken` mints an account-scoped `acspjc…` with no `qid`, which the
fabric answers with *"no matching policy"* — the same failure both the index load
and the media path hit before moving to signed tokens.

**Offerings are not interchangeable.** On the test content `default` publishes
only DRM formats and 404s for HLS, while `default_clear` publishes `hls-clear`.
So playout picks an offering by what it *publishes*, not by name. Frames are
unfussy and keep the name preference.

Frames come from `rep/frame_extract/<offering>/video?t=<seconds>`, which is more
precise than `rep/frame`. `ignore_trimming=true` matters there: without it the
frame is addressed against the trimmed timeline, not the one taggers record
timestamps against.

**Clips play their own segment**, seeking to `start_time` and pausing at the
end. Rows whose `end_time` is unusable get one reconstructed from the next
segment's start (`derive_segment_ends`), labelled `(derived)`; that exists for
rows tagged before whole-media tags carried a real duration, and goes dormant on
its own once an index is re-tagged.

---

## Core plug-in integration

Two additions on the core side, committed on branch `embeddings-visualizer`:

- `config/configuration.js` — `"Embeddings Visualizer": "http://localhost:8099"`.
  Position in that object *is* position in the app list; it is last, so it lands
  in the Tools box.
- `src/stores/index.js` — added to `darkChromeApps`, since this app is always
  dark and a light core header above a dark iframe reads as a seam.

No core-side icon is needed: `AppInfo.jsx` falls back to `<app url>/Logo.png`.

Inside core the app is a sandboxed iframe with no keys. It asks core for tokens
over the vendored `FrameClient` (`web/vendor/`) — one for the index qid on load,
one per content qid for media — so **no token is ever requested from the
viewer**. The standalone paste-a-token path exists only outside core.

Two traps on that channel: `FrameClient` resolves with the response *itself*
(destructuring `{response}` yields `undefined`), and it *rejects* with core's raw
error object or a bare string, so reading `.message` also yields `undefined` —
use `errorMessage()`.

The frontend is served with `Cache-Control: no-store`, because a stale
`index.html` that loads `app.js` as a classic script dies on its first `import`
and looks like the server being down.
