# embeddings-visualizer
Visualize vectors in an index in a latent space.

Prototype of a visual (graphic) interface for an index using SigLIP2-base-naflex embeddings.

**Plan:**

0. Get index qid from user.
   - iq__WAmru89ENrPpgmhQpBcXeVeuQUq has just frame_vectors (SigLIP 2)
   - iq__8MzaWjtTDuWuyTjAazYWnQDwPew will have just video_vectors (Qwen) but is currently empty (need to update Qwen tagger with newest commit first)
1. API call to vector store to get vectors in an index. (May visualize all or a subset (top 500?) of them.)
   - extract the metadata
      - embedding model info for index
   - how to pass auth token ? - use elv-client-js to get auth token that is passed to backend API call to vectorstore
2. Visual:
   - grid/graph background, Eluvio/EVIE color scheme (purplish)
   - each vector as a node
   - hover over a node highlights it and shows the vector's formatted metadata
     - if it is embedded text, show the text too
   - different colors for different modality embeddings (vector for text vs. image vs. video)? with a key/legend in the corner?
      - if there is a frame_idx, show that frame/image
      - if there's text, show the text
      - if it's a video, play the clip from start_time to end_time
3. Search:
   - search box in a corner
   - search query (text, image, or video) is embedded (using the vectors' embedding model (get from index)) and inserted as a new node in the same latent space (similar color to the query mode)
     - assume embedding model always supports text and image queries for now
         - video query breaks for now ? (unless Qwen)
     - later: add check ?
4. Media content:
   - if a vector is an embedded photo or bounding box or video, show the media content (retrieve from fabric):
      - point at elv-core-js and EVIE and have Claude determine how to get the playout as a pop-up
      - tell it app should be embeddable/plug-in-able (new localhost in configuration.js) in core-js so it can access private keys, and codebase matters for future manageability -
         - should be in JavaScript (except for maybe embeddings and projection (PCA) in Python backend (no Claude comments)? but still able to create a client Node application) and write comments for frontend



## Running

```
pip install -r requirements.txt
python3 src/app.py            # http://localhost:8096
```

Two different tokens are involved. The **index** token authorizes the vectorstore
read; standalone it is asked for once and kept in `localStorage`, and inside core
it comes from `CreateFabricToken`. A **content** token authorizes the media of one
content object, and is only ever asked for standalone — see *Media*.

## Layout

| | |
|---|---|
| `src/vectors_api.py` | reads vectors + metadata out of the vectorstore |
| `src/projection.py`  | PCA → UMAP down to 2D, and the query's out-of-sample placement |
| `src/embedder.py`    | SigLIP 2 towers for embedding a text or image query |
| `src/app.py`         | HTTP service; serves the API and the frontend from one origin |
| `web/media.js`       | resolves a vector's metadata into a frame image or a clip |
| `web/`               | the rest of the frontend (deck.gl, no build step) |

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

**Modality** comes from which fields a row populates, in order: `text` → `text`;
`frame_idx` → `image`; `end_time` strictly greater than `start_time` → `video`;
otherwise `unknown`, which shows metadata and does not try to load media. The
order matters — frame rows carry `start_time == end_time`, because a frame is an
instant, so a looser video test would swallow every frame in the index.

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

**Two transports, one shape.** Inside core the FrameClient signs URLs against the
viewer's account: `Rep(..., channelAuth: true)` for frames, `PlayoutOptions` for
clips. Standalone there is no account, so the viewer pastes a token for the
content object and URLs are built directly against a fabric node with
`?authorization=`. The prompt appears once per content object, not once per node,
and a rejected token is discarded rather than left stuck. Both paths choose the
offering identically, and the token path is dormant whenever core is present.

`ignore_trimming=true` matters on the frame endpoint: without it the frame is
addressed against the trimmed timeline, which is not the timeline the tagger
recorded timestamps against. The HLS master manifest embeds the authorization
into every child URI, so variant playlists and segments carry auth unaided.

Clips are bounded client-side: playback seeks to `start_time` and pauses at
`end_time` rather than running on into the next scene.

## Verified against the live vectorstore

`vectorstore-swagger.yaml` in content-search describes `/spaces` with
`embedding_size`, while the deployment serves `/indexes` with `vector_size` and
accepts `include_vector`. The spec is out of sync, so these were checked
directly (2026-09-04, against the frame index):

- `start_time_gte` / `start_time_lte`, `track` and `sources` all filter for real.
- Unknown body fields are ignored **silently** — a typo'd filter reads as no filter.
- `start_time` is an **Int4**: a bound above 2147483647 fails with a 500, not an
  empty result. `MAX_START_TIME_MS` sits on that ceiling.
- `GET /indexes/{qid}` returns only `{qid, vector_size}` — no config, no model.
- The tracks response keys each entry `name`, not `track`.

## Verified against the live fabric

Checked 2026-09-05 against `iq__o8BaaWDEbzGk6EJuGXiAJ97ZQC3`, by curl and then end
to end in a browser:

- The frame endpoint returns a real 640×480 JPEG at the requested timestamp.
- A clip reaches `readyState 4` with the content's true duration (1342 s), seeks
  to its in-point and pauses itself at its out-point.
- `?authorization=` is accepted on every one of these endpoints, so the
  standalone path needs no core and no keys.
- **Permissions are per-capability.** A token that reads `/meta/offerings` (200)
  can still be refused the frame rep (403, `q.read.bccall`: "no matching
  policy") — metadata read and bitcode call are separate grants, so an
  account-level `CreateFabricToken` is not necessarily enough to see media.

## Open items

- **The index does not record which model built it.** Neither the index metadata
  nor the per-vector metadata carries a model id, so the query model and the
  modes it supports are declared in the UI. Getting a model id written into the
  index config at creation time is the real fix, and would also let the
  unsupported-mode check derive itself.
- **`src/embedder.py` is a third copy** of a contract that already exists in
  content-search and in the tagger. All three must agree on checkpoint,
  revision, `normalize`, `max_num_patches` and the text tower's
  `max_length=64`, or query vectors land in a different space — and a mismatch
  does not raise, it just returns bad neighbours.
- **Payload size.** Points serialize at ~376 B each because `index_id`,
  `batch_id` and `qid` repeat per row. Hoisting the constant fields would cut a
  50k-point load from ~19 MB to a few MB.
- **Loaded indexes are capped at 4** and evicted oldest-first. Each pins its full
  vector matrix and a fitted projector, so an uncapped cache walks the process
  out of memory over a session of reloads. There is no eviction notice: a client
  holding an evicted `index_key` gets "index not loaded" and must reload.
- **The core path is the one piece never run for real.** Frames and clips are
  verified against the live fabric through the standalone token path, and the
  FrameClient calls are the same ones EVIE makes, but nothing here has yet run
  inside a core iframe.
- Video *queries* need an embedder that supports them (Qwen); SigLIP 2 does not.
  The clip branch of the detail panel is exercised, but only by forcing a frame
  row to video modality — modality colouring and a legend with more than one
  entry still need a populated multi-modal index.

## Plug-in integration

Two additions on the core side, both on branch `embeddings-visualizer`:

- `config/configuration.js` — `"Embeddings Visualizer": "http://localhost:8096"`
- `src/stores/index.js` — added to `darkChromeApps`, since this app is always
  dark and a light core header above a dark iframe reads as a seam.

The app itself talks to core through the vendored `FrameClient`
(`web/vendor/`), asking for a signed token with `CreateFabricToken`. Note that
`FrameClient` resolves with the response itself — destructuring `{response}` off
it silently yields `undefined`.

deck.gl compares a layer's `data` by reference, so state that feeds a layer must
be replaced, never mutated. Pushing onto the query array left the node layer
rendering a single point while the links — rebuilt fresh each frame — correctly
showed all of them, which looks like a placement bug rather than a data-diffing
one.

The frontend is served with `Cache-Control: no-store`. It is edited live, and a
stale copy fails in a way that looks like the server being down rather than like
a cache: a cached `index.html` that still loads `app.js` as a classic script hits
a syntax error on its first `import` and renders nothing at all.
