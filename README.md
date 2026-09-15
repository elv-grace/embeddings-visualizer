# embeddings-visualizer

Plots the vectors of an Eluvio vectorstore index as a 2D latent space, with
search: a text, image or video query is embedded into the same space and drawn
beside its true nearest neighbours. Queries are embedded by running the index's
own tagger container, so the service itself holds no models. Runs standalone or
as a plug-in inside elv-core-js.

Built with Claude Code.

---

## Setup

```bash
pip install -r requirements.txt
$EDITOR config.yml                 # fill in the image names
python3 src/app.py                 # http://localhost:8099
```

`requirements.txt` is Flask, requests and the projection stack — there is no
model runtime in it, and none is needed. **A query is embedded by running the
index's own tagger container**, so the only other requirement is a container
runtime (`docker`, or `podman` via `EV_CONTAINER_RUNTIME`), a GPU, and the
images.

| index `model` | query modes | container |
|---|---|---|
| `frame_vectors` (SigLIP 2) | text, image | model-frame-vector |
| `video_vectors` (Qwen3-VL) | text, image, video | model-qwenvl-video-vector |
| `face_vectors` (InsightFace) | image | model-celeb-vector |
| *anything else, or no model at all* | text | the `default` entry |

That table is `config.yml`'s `models:` section, merged over the defaults in
`src/embedder.py`. Each entry is an image, the modalities its space can be
queried with, the **CUDA device** its container runs on, and any runtime
arguments that model needs (a weight mount); a key left unset keeps its built-in
value, so an entry naming only an image does not restate its modalities. Adding
a tagger is adding an entry; nothing else in the service knows one model from
another.

**Every container runs on a GPU, and each one is placed by configuration.**
`device:` takes an index (`0`), several (`[0, 1]`) or `all`, per model or once
under `container:` for all of them; there is no default beyond that, so a model
nothing places raises rather than running somewhere nobody chose. It becomes
`--device nvidia.com/gpu=N` under podman and `--gpus device=N` under docker, and
inside the container the card is renumbered to 0 — so a model asking for a bare
`cuda` lands where it was put, knowing nothing about any of this. That is the
point of placing them here: with one container per model, three models each
picking "the emptiest GPU" all pick the same one.

`config.yml`'s other section, `container:`, is how the containers are run —
`runtime` (docker or podman), the default `device`, `args` for every container,
the per-query `timeout` (900s, which has to cover a cold model load) and the
`keepalive` sweep interval. Every setting is overridden by an environment
variable (`EV_CONTAINER_RUNTIME`, `EV_CONTAINER_DEVICE`, `EV_CONTAINER_ARGS`,
`EV_CONTAINER_TIMEOUT`, `EV_CONTAINER_KEEPALIVE`), because a one-off should not
have to be undone in a file that is committed and shared. `EV_CONFIG` moves the
file itself.

**Containers are started at boot and kept running.** Loading a multi-GB
checkpoint costs tens of seconds, and the protocol lets a container stay alive
across inputs, so one is started per configured model when the service starts
and a sweep puts back any that later dies — backing off as it keeps dying, so a
wrong image is not pulled on a loop. An image that has to be pulled does not
hold up the service: indexes load and plot while it does, and only a *query*
needs a container.

**One model is one container.** The recipe a container embeds under is its
`--params`, which is a *launch* argument — so deriving it from each index's
stamped tuning meant a second index with a different recipe started a second
container, holding a second copy of the same multi-GB checkpoint on the same
card. The recipe is therefore stated once, as `params:` on the model in
`config.yml`, and nothing an index carries can start another container.

An index whose rows stamp something different is **reported, not accommodated**:
`POST /api/index` returns `tuning_mismatch`, the service logs it, and the stats
line says so. That disagreement is worth seeing rather than silently working
around — it means queries are embedded under different parameters than the
vectors were, which does not fail, it just returns worse neighbours. The fix is
to align `params:` or re-tag, both of which are deliberate.

### How a query reaches the model

Over the [tagger model
protocol](https://docs.eluv.io/docs/ai-ml/ai-ml-tagger/model-protocol/): the
container reads newline-separated *file paths* on stdin and appends
newline-delimited `tag` / `progress` / `error` messages to its `--output-path`.
So the query is staged as a file in a directory bind-mounted into the container,
its path is written to stdin, and the service tails the output until a
`progress` message names that file. The vector comes off the `tag`.

**A text query is a file too.** Stdin already carries the path stream, so text
arrives as a `.txt` of newline-separated queries — one vector tag back per line —
which is the one addition this service makes to the protocol.

The container is launched with `config.yml`'s `params:` for that model, and the
tuning stamped on an index's rows is read only to check it against them.

---

## How it works

**The plot is a random sample, drawn in one call.** `/search` takes a
`shuffle_seed`: with one set the rows come back in random order rather than by
distance, so the first `limit` of them are a uniform sample of the index. A
vector is ~12 KB of JSON, so plotting is bounded by `sample_size` (10,000 by
default) rather than by how large the index is. The seed is derived from the
request's `seed`, so the same index draws the same sample — and the same
picture — on every run. The browser receives 2D coordinates plus metadata,
never vectors.

**A search sees the whole index, not the sample.** The sample bounds what is
*drawn*; letting it bound what is *findable* would make every query a search of
ten thousand arbitrary rows. So a query is ranked by the vectorstore itself — an
HNSW lookup over every row — and its top 100 hits come back with their
embeddings. Those are placed in the fitted projection with the same `transform`
an out-of-sample row uses and added to the plot, deduplicated against what was
already there. The nearest 10 are linked and listed, as before; the other 90 are
plotted unlabelled so the neighbourhood the query landed in is visible and not
just its winner.

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

**A query runs the tagger itself**, rather than a reimplementation of it or an
import of its source. That is the only way a query is *guaranteed* to land in
the space it is searching: preprocessing budget, pooling, instruction and
normalize all move a vector without failing, and the container is the version
that built the index. It is also why this service installs no model runtime —
three stacks that cannot coexist in one interpreter (torch 2.x, transformers
≥4.57, numpy <1.20) are three images.

**The model comes from the batch.** Each row names its `batch_id`;
`GET /indexes/{qid}/batches/{batch_id}` returns that batch's `model`, which maps
1-1 to a container in `config.yml`. A tagger's stamped `additional_info`
is read only to check the *tuning* parameters (patch budget, sampling budget,
MRL width) against the recipe its container is configured to run. An index
whose batches report no model is queried through the `default` entry, which is
text-only: text is the one modality every embedding model accepts, so it is the
only assumption safe to make about an unidentified space.

### Layout

| | |
|---|---|
| `src/app.py` | HTTP service; serves the API and the frontend from one origin |
| `src/vectors_api.py` | samples an index out of the vectorstore, and searches all of it |
| `src/projection.py` | PCA → UMAP to 2D, plus the query's neighbour anchoring |
| `src/embedder.py` | model → container registry, the tuning parameters, and the query side |
| `src/tagger.py` | runs a tagger container and reads the vector back over its protocol |
| `src/config.py` | reads `config.yml`; the environment wins over it |
| `config.yml` | the image, modalities and runtime arguments per model, and how containers are run |
| `web/app.js` | deck.gl layers, queries, legend, detail panel |
| `web/media.js` | metadata → a frame image (`frame_extract` instead of `frame`) or a playable clip |
| `web/index.html` | header controls, the help panel, and the stage's overlays |
| `web/style.css`  | the dark palette, including the modality colours `app.js` mirrors |
| `web/vendor/`    | elv-client-js's prebuilt `FrameClient` (used only inside core) |

Two endpoints: `POST /api/index` loads, samples and projects;
`POST /api/search/<mode>` embeds a query, ranks it against the whole index and
returns both its neighbours and the hits to add to the plot. Both take the
fabric token in an `Authorization` header — it is forwarded to the vectorstore
and never stored, so a search needs one of its own. Everything else is static.

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

**Clips play their own segment**, seeking to `start_time` and pausing at
`end_time`. A row needs `end_time > start_time` to be a video at all; one
without it describes no extent and is reported as such in the detail panel
rather than guessed at. That happens to rows written before whole-media tags
carried a real duration — the pipeline re-based a `start == end == 0` sentinel
into both fields — and the fix is to re-tag.

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
