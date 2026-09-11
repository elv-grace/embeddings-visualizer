# embeddings-visualizer
Visualize vectors in an index in a latent space.

Prototype of a visual (graphic) interface for an index using SigLIP2-base-naflex or Qwen3-VL-Embedding-8B embeddings. (built with Claude Code)

**Plan:**

0. Get index qid from user.
   - iq__WAmru89ENrPpgmhQpBcXeVeuQUq has just frame_vectors (SigLIP 2)
   - iq__8MzaWjtTDuWuyTjAazYWnQDwPew has just video_vectors (Qwen) segmented by shot by the tagger
1. API call to vector store to get vectors in an index. (May visualize all or a subset (top 500?) of them.)
   - extract the metadata
      - embedding model info for index - currently in vector's additional_info, but can be queried from vector batch info without having it repeated in each vector?
   - auth token handled by integration with elv-core
2. Visual:
   - grid/graph background, Eluvio/EVIE color scheme (purplish)
   - each vector as a node
   - hover over a node highlights it and shows the vector's formatted metadata
     - if it is embedded text, show the text too
   - different colors (color-blind-friendly) for different modality embeddings (vector for text vs. image vs. video) with a key/legend in the corner
      - embedding model modality support also in vector's additional_info
      - if there is a frame_idx, show that frame/image
      - if there's a start_time and end_time and end_time > start_time, then it's a video
      - else it's text
3. Search:
   - search box in a corner
   - search query (text, image, or video) is embedded (using the vectors' embedding model (get from index)) and inserted as a new node in the same latent space (similar color to the query mode)
     - assume embedding model always supports text and image queries for now
         - video query breaks for now ? (unless Qwen)
     - later: add check ?
4. Media content:
   - if a vector is an embedded photo or bounding box or video, show the media content (retrieve from fabric by integrating with elv-core (determined by Claude))
      - app should be embeddable/plug-in-able (new localhost in configuration.js) in core-js so it can access private keys
      - codebase matters for future manageability
         - should be in JavaScript (except for maybe embeddings and projection (PCA) in Python backend (no Claude comments)? but still able to create a client Node application) and write comments for frontend



## Running

```
pip install -r requirements.txt
python3 src/app.py            # http://localhost:8099
```

Two different tokens are involved. The **index** token authorizes the vectorstore
read; standalone it is asked for once and kept in `localStorage`, and inside core
it comes from `CreateSignedToken` for the index qid. A **content** token
authorizes the media of one content object, and is only ever asked for
standalone — see *Media*.

### Querying a Qwen index needs extra packages

Loading and plotting any index works with the base install. **Querying** a
Qwen-stamped index additionally needs the tagger's own runtime, because the
query is embedded by importing `model-qwenvl-video-vector`'s embedder rather
than reimplementing it. Those packages are in `requirements.txt` under their own
heading; without them the first query — not the load — fails with:

```
could not embed query: found the tagger at <path> but could not import its
embedder: No module named 'qwen_vl_utils'. Its dependencies (qwen-vl-utils,
decord, transformers>=4.57.3) have to be installed in this environment.
```

That is the intended message, not a bug: the embedder is resolved lazily, so the
error names the path it found the tagger at and the exact module missing.
**Nothing is installed automatically** — the service never shells out to pip, so
this resurfaces in every fresh environment (a container, another checkout,
another account) until `requirements.txt` is installed there. Inside the
tagger's own container it never appears, since that image already carries them
and `/elv` is the first path searched.

`transformers>=4.57.3` in that message is a floor, not a pin — a newer major
version is fine (verified on 5.12.1), so it rarely needs action.

Two costs on the *first* Qwen query, neither repeated: the checkpoint downloads
(`Qwen/Qwen3-VL-Embedding-8B` is **16.3 GB**) and then takes ~45 s to load. The
model then stays resident.

### Querying a celeb (face) index needs its own interpreter

```
tools/setup_celeb_env.sh      # once, ~1.5 GB
```

`face_vectors` indexes load and plot with no setup at all. **Querying** one
needs model-celeb-vector's stack, which cannot coexist with the rest of the
service: its `setup.py` pins `numpy<1.20.0`, `torch==1.9.0` and `mxnet`, where
SigLIP 2 and Qwen run on torch 2.13 and numpy 2.x. Installing it into the
service environment would downgrade numpy below 1.20 and break every other
tower, plus UMAP and scikit-learn — so it is deliberately **not** in
`requirements.txt`. (If the face embedder is updated, this may no longer apply.)

So that stack gets its own interpreter and the service talks to it over a pipe:

| | |
|---|---|
| `tools/setup_celeb_env.sh` | builds `.celebenv/` — Python 3.8, numpy 1.19.5, torch 1.9.0+cpu, mxnet 1.9.1 |
| `tools/celeb_worker.py` | runs inside it; imports the real `CelebVectorizer`, one JSON line per query |
| `src/celeb_embedder.py` | starts the worker once and keeps it, since r100 costs ~2 s to load |

It is cheaper than it looks because the tagger sets `gpu: -1` and runs
InsightFace on CPU deliberately, to match `celeb/model.py`. CPU wheels are
therefore enough: no CUDA 10.1, no cudatoolkit, and none of the ~10 GB the
tagger's conda container would cost. Python 3.8 comes from `uv` — this box has
only 3.10, where `numpy<1.20` has no wheels.

**The bridge does not change the vector.** Checked against calling the tagger
directly on the same image: max absolute difference `0.0`, cosine `1.0`. Images
cross as PNG on disk rather than as arrays over the pipe, because the two
interpreters have incompatible numpy ABIs; decoding happens service-side so EXIF
rotation is applied, which `cv2.imread` would ignore — a sideways phone photo
would otherwise embed as a sideways face.

Without `.celebenv/`, `celeb_embedder` falls back to importing in-process, which
is the path that works *inside* the tagger's own container (`/elv`), and
otherwise fails naming the missing module. `CELEB_PYTHON`,
`CELEB_EMBEDDING_PATH`, `COMMON_ML_PATH` and `CELEB_MODEL_PATH` override the
interpreter, the two checkouts and the weights.

**The weights are not where `config.yml` says.** It names
`/ml/models/celeb_detection`; the r100 checkpoint actually lives at
`/ml/models/celeb` (`models/model-r100-ii/model-{symbol.json,0000.params}`),
which is what `model_input_path` is joined against.

Expect a few seconds per query: MTCNN detection runs on CPU, so a 10-megapixel
upload takes ~7 s while a smaller one is much faster.

Face vectors are **image-only**: they are identity embeddings, so no text tower
can put a description into that space, and a text query is refused rather than
answered with something meaningless.

### GPU selection

The tagger asks for a bare `torch.device("cuda")`, which means *the current
device* — device 0 unless told otherwise. On a shared multi-GPU box that is
reliably the busiest card, so the model would load into the least free memory
while others sat idle, and a long video query would then OOM with several GiB
free elsewhere.

`_use_best_cuda_device()` sets the current device to whichever GPU has the most
free memory before the tagger constructs its model, which redirects both its
`torch.device("cuda")` and the `.to(device)` that follows — no argument the
tagger does not have, and no edit to it. `mem_get_info` reports the driver's
view, so memory held by *other* processes counts, which is what an allocation
actually competes with.

On an OOM the model is moved once to whichever card now has the most room and
the embed is retried; the device is chosen at load, but a long-running service
outlives that snapshot. It is deliberately **not** retried with fewer frames:
`fps`/`max_frames` are the recipe's sampling budget, and a query sampled
differently lands in a different space from the indexed vectors, so it would
return quietly wrong neighbours instead of failing. If it still does not fit,
the error says so in those terms and suggests a shorter clip.

The heuristic is "most free memory", so a small card can win when it happens to
be the emptiest — on a box with mixed GPUs a 16 GB card can be picked for a
model that needs more, which surfaces as a load failure rather than a silent
one.

## Layout

Backend (Python):

| | |
|---|---|
| `src/app.py`          | HTTP service; serves the API and the frontend from one origin |
| `src/vectors_api.py`  | enumerates and samples an index out of the vectorstore |
| `src/projection.py`   | PCA → UMAP to 2D, plus the query's neighbour anchoring |
| `src/embedder.py`     | the model → embedder registry, and the tuning parameters |
| `src/siglip2_embedder.py` | SigLIP 2 image and text towers |
| `src/qwen_embedder.py`| Qwen3-VL text/image/**video** queries, via the tagger's embedder |
| `src/celeb_embedder.py`| InsightFace face queries, via model-celeb-vector's model |

Frontend (JavaScript, no build step):

| | |
|---|---|
| `web/index.html` | header controls, the help panel, and the stage's overlays |
| `web/app.js`     | deck.gl layers, queries, the legend, and the detail panel |
| `web/media.js`   | turns a vector's metadata into a frame image or a playable clip |
| `web/style.css`  | the dark palette, including the modality colours `app.js` mirrors |
| `web/vendor/`    | elv-client-js's prebuilt `FrameClient` (used only inside core) |

Two API endpoints: `POST /api/index` loads, samples and projects an index;
`POST /api/search/<mode>` embeds a query and ranks it. Everything else is static.

## How it works

**Enumeration.** The vectorstore has no scan endpoint — every read is a KNN
around a reference vector. But its search filters (`qids`, `track`,
`start_time_gte/lte`) are *pre-filtered*: the search runs an exact KNN over the
filtered subset. So once a window holds fewer rows than `limit`, the top-K over
it is the whole window, and the reference vector no longer decides membership,
only ordering. Walking the timeline in windows therefore enumerates the index
exactly. `count_windows` bisects any window that comes back full, so it measures
truncation rather than trusting that behaviour.

**Sampling.** A vector is ~12 KB of JSON, so the counting pass runs with
`include_vector=false` (~150 B/row) and only the sample is refetched with
vectors. Each window gets a quota proportional to its population, and each draws
its own random probe direction — so the strata are covered proportionally and
the within-window selection bias varies independently instead of compounding
into one global cone. A single fixed probe, `[0.5]*d` or one random draw alike,
returns a cone of the space rather than a sample of the index.

**Projection.** PCA to 50 dims, then UMAP to 2. PCA first because UMAP's
neighbour search degrades in very high dimensions, and because a quarter of the
columns are zero padding (SigLIP 2 emits 768 dims into a 1024-wide index). The
`PCA` toggle plots the first two components directly instead: honest distances,
poor cluster separation — on the frame index those two components retain **21%**
of the variance, which is why that view looks like one undifferentiated cloud.
t-SNE is not offered: it has no out-of-sample extension, so placing a query
would mean refitting and reshuffling the map under the viewer.

**2D, not 3D.** Every feature here is a 2D interaction — hover, click, tooltip,
legend, a search panel. A third axis adds one more component of explained
variance (~5 points) and costs hover picking to occlusion. The door is left
open: `n_components` is a parameter and deck.gl's `OrbitView` is a view swap.

**2D distance is not similarity.** Any projection distorts neighbourhoods, so a
node beside the query on screen may not be among its nearest vectors. The server
ranks in the original space (`top_k_similar`) and the frontend draws those links
explicitly. On the frame index a text query's true neighbours land in *three*
separate visual clusters — visible proof the picture alone would mislead.

**Similarity scales differ by mode.** SigLIP's sigmoid loss makes cross-modal
cosines small in absolute terms: a strong text→image hit scores ~0.07 where
image→image scores ~0.5 and two adjacent frames score ~0.95. Link opacity is
therefore scaled within a result set, not against an absolute range, and the UI
says to compare within a mode rather than across.

**The embedding recipe comes from the tags, not the index.** A tagger stamps
`additional_info` on every tag it writes — `embedder`, `revision`, `dim`,
`normalize`, `query_modes`, plus whatever else changes the vector
(`max_num_patches` for SigLIP 2; `prompt`, `fps`, `max_frames`, `max_length` for
Qwen). The vectorstore stores it as JSONB and returns it **verbatim on every
search hit**, so the recipe arrives on the same row as the vector and no second
lookup is needed — `read_tuning` parses what `get_vectors` already fetched. It
is opaque to the index though: not indexed and not filterable, so it can ride
along but cannot narrow a search.

**It no longer says which model to use.** That comes from the batch (below);
`additional_info` is read only for the parameters that *tune* an already-chosen
tower, via a whitelist (`embedder.TUNING_KEYS`) so the provenance a tagger also
stamps — `box`, `score`, `upscale`, `crop_padding`, `detector`, `text` — never
reaches an embedder.

It answers the two things the index cannot: which model to embed a query with,
and which query modes that model supports. When a recipe is found the query mode
checkboxes are filled in and disabled; when there is none they stay editable and
the header says **declared, not detected** — an index tagged before taggers
started stamping has no recipe, which today is all of them.

`build_embedder` dispatches on the checkpoint id and threads every recipe
parameter into the embedder, so a query is embedded under the recipe its index
was built with. An unrecognised embedder raises rather than falling back to
SigLIP 2: querying with the wrong model does not fail, it quietly returns
meaningless neighbours.

**Query towers are imported from the taggers, not reimplemented.** Whatever
produced the indexed vectors has to produce the query vector too, and a mismatch
in preprocessing, pooling or normalize does not raise — it quietly returns bad
neighbours. So both embedders run the tagger's own code:

- `src/qwen_embedder.py` wraps `Qwen3VLEmbedder`, whose `process()` already takes
  text, image and video uniformly, so the query side just supplies
  `{text|image|video, instruction, fps, max_frames}`. Video uploads go to a temp
  file keeping their suffix, since the reader opens paths and picks its decoder
  by extension.
- `src/embedder.py`'s image path runs model-frame-vector's `Siglip2CropEmbedder`
  (the repo formerly named model-detection; its package is still
  `general_detection`), falling back to a local vision tower when it is not
  importable. It replaced model-vector's `FeatureExtractor` on 2026-09-10 and
  was checked against the fallback on a non-square image: **bit-identical**, max
  absolute difference `0.000e+00`, cosine `1.0`. Its extra `max_upscale` lever
  only shrinks the patch budget for small *crops*; left at its default of None
  a whole-image query gets the full `max_num_patches`, which is the budget the
  frame tagger always used. Its *text* path has no counterpart to import
  — that tagger only ever loads the vision tower — so the text tower is
  necessarily query-side code.

Both resolve the tagger the same way: plain import first (`/elv` is the tagger
container's WORKDIR and already on `sys.path`), then `QWEN_EMBEDDING_PATH` /
`SIGLIP_EMBEDDING_PATH`, then a sibling checkout. No home paths.

**Modality** is read from which fields a row populates, in order: `text` →
`text`; `frame_idx` → `image`; `end_time` strictly greater than `start_time` →
`video`; otherwise `unknown`, which shows metadata and does not try to load
media. The order matters — frame rows carry `start_time == end_time`, because a
frame is an instant, so a looser video test would swallow every frame in the
index.

A stamped `kind` still wins when a row carries one, but the taggers stopped
stamping it on 2026-09-10, so the field test above is the normal path. What made
that safe was the taggers starting to record a **real duration** on a whole-media
vector. The one case the field test genuinely cannot read is the old
`start == end == 0` sentinel: a lone vector with no interval and no `frame_idx`
is indistinguishable from an unknown row, which is what `kind` had been covering.
With `end > start` it simply matches the video rule.

Rows tagged before that change are still handled, but by
`derive_segment_ends` rather than by `kind` — see *Clips play their own segment*.

**A query node is placed among its matches, not by projecting it.** This is the
one place the pipeline deliberately departs from "same treatment as an index
vector". `transform` itself is sound — indexed vectors round-trip to within 0.12
units on a 30-unit-wide map — but a *query* vector is essentially never near the
fitted manifold, and UMAP has nothing local to place it by. Measured on the frame
index, random unit noise, a text query and an out-of-domain image query all
landed within ~0.2 units of the same point: every query collapsed onto the
"nowhere" spot that noise maps to, so its position said nothing about the query.

The cause is the modality gap. SigLIP's text and image embeddings occupy separate
cones, so a text query sits at cosine ~0.03–0.08 from *every* indexed vector —
near-equidistant, and no neighbourhood to speak of. It is not only cross-modal:
an image query scoring 0.52 still collapsed, because the index's own neighbour
distances are ~0.05, so 0.52 is far outside the manifold too.

So `anchor_to_neighbours` puts the node at the softmax-weighted centroid of its
true neighbours' plotted positions instead. Sharpening the weights keeps it
beside its best match rather than drifting into the empty space between
scattered ones. The raw projected point is still returned as `projected_point`
for comparison; nothing is drawn at it.

**Clicking a query node opens the query itself**, and pins its links — the panel
and the links answer the same question, so having them appear and disappear
separately just makes work. For an uploaded image or clip the panel shows the
file, not its name: the name says nothing about whether the right file was
picked, and what the image was is the whole question a viewer has about an image
query. The upload is held as a blob URL for the query's lifetime and revoked
when it is removed.

**Video plays in both places, from two different sources.** Clicking a *vector*
whose modality is video streams the fabric clip; clicking a *query* node plays
the file that was uploaded, straight from its blob URL — no fabric and no content
token involved. Both render a `<video>` with controls.

Only the vector's playback is bounded, and only when there is a real interval:
it seeks to `start_time` and pauses at `end_time`, but an unsegmented whole-video
vector carries `start == end == 0`, which reads as "no out point" and plays the
whole thing. Its caption says **Whole video** rather than `0:00.000 – 0:00.000`,
which would claim a zero-length clip while the full video is running.

**Queries accumulate.** A new search adds a node rather than replacing the last
one, because comparing where two searches land is the reason to plot them at all.
Each carries a number badge matching its legend row, takes its modality's hue
lightened (a text query belongs beside the text vectors it searches), and owns
its own link lifecycle — one fading out cannot disturb another that is pinned.
Rows remove individually, or all at once.

**Links flash, then get out of the way.** A query's neighbour links are drawn for
a few seconds and then fade: left up permanently they overdraw the map, and the
map is what the viewer came for. Hovering the query node brings them back,
clicking pins them, and the legend's query row toggles the same state. The
dimming of non-neighbour points is tied to the same lifecycle, so the plot
returns to reading as a plain distribution once the links are gone.

## Media

A vector's metadata names a content object (`qid`) and a position on its timeline,
which is not enough to address media — both frames and clips need a URL authorized
against that object.

**Offerings are not interchangeable, and the obvious one is often unplayable.**
On the test content, `default` publishes only DRM formats (`dash-widevine`,
`hls-fairplay`, `hls-sample-aes`) and **404s** for HLS, while `default_clear`
publishes `hls-clear`. So playout picks an offering by what it actually
publishes — `hls-clear` or `hls-aes128` — not by name. This is the same check
EVIE makes when it marks offerings `disabled`. Frames are unfussy: every offering
serves them, so those keep the name preference.

**One transport, two token sources.** Every URL is built the same way — straight
at a fabric node with `?authorization=<token>` — and only the token's origin
differs. Inside core it is minted on demand by `CreateSignedToken({objectId,
grantType: "read"})` over the FrameClient, so nothing is asked of the viewer.
Standalone the viewer pastes one, once per content object rather than once per
node, and a rejected token is discarded rather than left stuck.

**The token's scope is what decides access, not the transport.** This replaced a
split where core used `Rep(..., channelAuth: true)` and `PlayoutOptions`
instead. Those authorize against the *account*: `CreateFabricToken` mints
`acspjc…` carrying only `adr`/`sub`/`spc` — no `qid`, no grant — and the fabric
answers "no matching policy". `CreateSignedToken` mints `aessjc…` carrying
`qid`/`lib`/`gra: read`, which is the shape that works; it resolves `libraryId`
from `objectId` itself and is on FrameClient's allowlist. The same correction
applies to the index load, which authorizes the *index* qid the same way.

Collapsing the two paths also means **the code that runs in core is the code
exercised standalone**, rather than a second implementation that only ran where
it was hardest to test — which is why the core path stayed broken unnoticed.

Minted tokens are cached per `qid` (24 h, re-minted a minute early), so clicking
twenty nodes from one object costs one signature rather than twenty.

`ignore_trimming=true` matters on the frame endpoint: without it the frame is
addressed against the trimmed timeline, which is not the timeline the tagger
recorded timestamps against. The HLS master manifest embeds the authorization
into every child URI, so variant playlists and segments carry auth unaided.

**Read errors through `errorMessage`, never `err.message`.** `FrameClient`
rejects with core's raw error object and, on timeout, with a bare string —
neither has a `.message`, so reading it directly renders the string
`"undefined"` in place of every real failure.

### Clips play their own segment

Playback seeks to `start_time` and pauses at the row's end. A row with no usable
end has no out point and plays to the end of the file.

When the tagger windows a video itself — `segment_length_s` — each window is
stamped with its true bounds and none of this is needed. The awkward case is
**tag-aligned** tagging, where the pipeline cuts the video first (by the
`shot_detection` track, say) and hands the tagger one piece at a time. The
tagger sees a single window in that piece, so it stamps the whole-media values
and the pipeline then re-bases the row into the parent timeline.

Historically that sentinel was `(0, 0)`, and re-basing shifted **both** fields
together, leaving `start == end == segment start` — a row that reads as "whole
video" and plays past its own segment. `derive_segment_ends` reconstructs those:

> **end(segment *i*) = start(segment *i+1*)**, the last one left open

It assumes only that segments **tile the timeline contiguously without
overlap**, which is true of shot alignment and fixed-length alignment alike, so
it keys off neighbouring starts rather than anything shot-specific. It would be
wrong for *overlapping* windows and overshoots a genuine gap; neither is
distinguishable from tiling using these rows alone, since the real end is
exactly what is missing.

Ends are derived from the **counting pass**, which walks the whole index at
~150 B/row, not from the sample. A sampled row therefore carries the bound it
has in the full index rather than one stretched across dropped neighbours.

Since 2026-09-10 the tagger stamps a real duration instead of `(0, 0)`, so
re-tagged rows arrive with `end > start` and the derivation goes dormant on its
own. It stays for rows tagged before that, and is keyed on `end <= start` rather
than on anything about shots.

Nothing is overwritten: the reconstruction is written to `derived_end_time`, so
a row keeps reporting what the tagger recorded and the UI can label a
reconstructed bound `(derived)`.

## Verified against the live vectorstore

`vectorstore-swagger.yaml` in content-search is a **stale copy of the `/spaces`
spec**: it describes `embedding_size` where the deployment serves `/indexes` with
`vector_size`, omits `include_vector`, and its `VectorResponse` predates
`additional_info` — as does content-search's own `Vector` dataclass, which has no
such field and drops it when parsing. Treat <https://docs.eluv.io/api/vectorstore/>
as authoritative, not the checked-in YAML. Checked directly (2026-09-04, against
the frame index):

- `start_time_gte` / `start_time_lte`, `track` and `sources` all filter for real.
- Unknown body fields are ignored **silently** — a typo'd filter reads as no filter.
- `start_time` is an **Int4**: a bound above 2147483647 fails with a 500, not an
  empty result. `MAX_START_TIME_MS` sits on that ceiling.
- `GET /indexes/{qid}` returns only `{qid, vector_size}` — the *index* cannot say
  which model built it. A *vector* can: `additional_info` round-trips per search
  hit, which is where the recipe comes from.
- The tracks response keys each entry `name`, not `track`.
- An absent key is not a missing column. Captured rows in
  `vector_metainfo_data_ref.txt` show no `additional_info` only because those
  tags never set one — Go's `omitempty` elides it.

## Verified against the live fabric

Checked 2026-09-05 against `iq__o8BaaWDEbzGk6EJuGXiAJ97ZQC3`, by curl and then end
to end in a browser:

- The frame endpoint returns a real 640×480 JPEG at the requested timestamp.
- A clip reaches `readyState 4` with the content's true duration (1342 s), seeks
  to its in-point and pauses itself at its out-point. A vector with
  `start == end == 0` instead plays on past 14 s without pausing — the
  whole-video case behaving as intended.
- An uploaded video query plays from its blob URL (`readyState 4`, the upload's
  own 4.2 s duration), so a query clip needs neither the fabric nor a token.
- `?authorization=` is accepted on every one of these endpoints, so the
  standalone path needs no core and no keys.
- **An account-scoped token is not enough, and fails unevenly.** One read
  `/meta/offerings` (200) yet was refused the frame rep (403, `q.read.bccall`:
  "no matching policy"), which reads like a per-capability grant but is the
  token's scope: `CreateFabricToken` names no `qid`. Re-checked 2026-09-09 with
  an object-scoped `aessjc…` token on the same object — `/meta/offerings` 200,
  frame rep 200 (`image/jpeg`, 640×480, 40,776 B), and the `hls-clear` manifest
  200 with the authorization embedded in all 37 child URIs.

## Verified inside core

Checked 2026-09-09 with the app loaded as a plug-in in a running elv-core-js:

- The index loads with no token input — `CreateSignedToken` for the index qid.
- Frame images and video play with no token input — the same call per content
  qid. The standalone unlock form correctly never appears.
- The vectorstore does not evaluate access itself: it forwards the token to the
  fabric node holding the index and relays the verdict. So the status code says
  *which* half failed — **400** (`unknown scheme`) is a malformed or absent
  token, **403** is a well-formed one denied by policy. That reason lives only
  in the response body, which `raise_for_status()` discards; `_check`/`_reason`
  in `vectors_api.py` walk the fabric's nested `cause` chain to the innermost
  `kind`/`op` so a 403 is not indistinguishable from a stopped service.

## Open items

- **Nothing is stamped yet.** The recipe path is wired and unit-tested against
  captured row shapes, but it has never parsed a real recipe: the tagger change
  lives only in `model-vector` locally and the frame index predates it, so its
  rows carry no `additional_info`. Re-tag content and detection takes over on
  its own; until then every index falls back to the declared model.
- ~~**The Qwen path has never run.**~~ Closed 2026-09-09: text and image queries
  both return results against `iq__8MzaWjtTDuWuyTjAazYWnQDwPew`
  (`Qwen/Qwen3-VL-Embedding-8B`, 267 rows). The first query pays a **16.3 GB**
  checkpoint download and then ~45 s to load 749 tensors; later queries reuse
  the resident model. The tagger's runtime deps are now listed in
  `requirements.txt` — they are *not* implied by installing the rest, so a fresh
  environment fails with an import error until they are installed.
- **One model per index.** A query has to be embedded with one model, so if an
  index mixes batches from different taggers the first model wins. All of them
  are returned in `models` (batch id → model) so a mismatch is at least visible.
- **The SigLIP text tower is still query-side code.** The image half now runs the
  tagger's own `Siglip2CropEmbedder`, so that contract cannot drift. The text half
  has nothing to import — the tagger only ever loads the vision tower — so its
  `padding="max_length"`, `max_length=64` tokenization is duplicated with
  content-search's copy, and a mismatch there does not raise, it just returns
  bad neighbours.
- **Payload size.** Points serialize at ~376 B each because `index_id`,
  `batch_id` and `qid` repeat per row. Hoisting the constant fields would cut a
  50k-point load from ~19 MB to a few MB.
- **Loaded indexes are capped at 4** and evicted oldest-first. Each pins its full
  vector matrix and a fitted projector, so an uncapped cache walks the process
  out of memory over a session of reloads. There is no eviction notice: a client
  holding an evicted `index_key` gets "index not loaded" and must reload.
- ~~**The core path is the one piece never run for real.**~~ Closed 2026-09-09:
  index load, frame images and video playback all verified inside a core iframe,
  with no token prompt. See "Verified inside core".
- Video *queries* need an embedder that supports them (Qwen); SigLIP 2 does not.
  The clip branch of the detail panel is exercised, but only by forcing a frame
  row to video modality — modality colouring and a legend with more than one
  entry still need a populated multi-modal index.

## Plug-in integration

Two additions on the core side, both on branch `embeddings-visualizer`:

- `config/configuration.js` — `"Embeddings Visualizer": "http://localhost:8099"`,
  placed **last** in the `apps` object. `AppInfo.jsx` sorts nothing: an app is in
  the Application Suite only if it matches its `appNames` list and is a Tool
  otherwise, and both boxes render in this object's key order. So position in
  that file *is* position in the box.
- No core-side icon: `AppInfo.jsx` falls back to `<app url>/Logo.png` for
  anything absent from its icon map, and `web/Logo.png` answers that — which
  keeps the artwork with the app rather than adding an import to core.
- `src/stores/index.js` — added to `darkChromeApps`, since this app is always
  dark and a light core header above a dark iframe reads as a seam.

The app itself talks to core through the vendored `FrameClient`
(`web/vendor/`), asking for object-scoped tokens with `CreateSignedToken` — for
the index qid on load, and per content qid for media. Two things about that
channel bite silently: `FrameClient` resolves with the response *itself*, so
destructuring `{response}` off it yields `undefined`; and it *rejects* with
core's raw error object or a bare string, so reading `.message` off a failure
also yields `undefined`.

deck.gl compares a layer's `data` by reference, so state that feeds a layer must
be replaced, never mutated. Pushing onto the query array left the node layer
rendering a single point while the links — rebuilt fresh each frame — correctly
showed all of them, which looks like a placement bug rather than a data-diffing
one.

The frontend is served with `Cache-Control: no-store`. It is edited live, and a
stale copy fails in a way that looks like the server being down rather than like
a cache: a cached `index.html` that still loads `app.js` as a classic script hits
a syntax error on its first `import` and renders nothing at all.
