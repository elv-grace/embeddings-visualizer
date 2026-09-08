# embeddings-visualizer
Visualize vectors in an index in a latent space.

Prototype of a visual (graphic) interface for an index using SigLIP2-base-naflex embeddings. (built with Claude Code)

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
python3 src/app.py            # http://localhost:8099
```

Two different tokens are involved. The **index** token authorizes the vectorstore
read; standalone it is asked for once and kept in `localStorage`, and inside core
it comes from `CreateFabricToken`. A **content** token authorizes the media of one
content object, and is only ever asked for standalone — see *Media*.

## Layout

Backend (Python):

| | |
|---|---|
| `src/app.py`          | HTTP service; serves the API and the frontend from one origin |
| `src/vectors_api.py`  | enumerates and samples an index out of the vectorstore |
| `src/recipes.py`      | parses the embedding recipe a tagger stamped on each vector |
| `src/projection.py`   | PCA → UMAP to 2D, plus the query's neighbour anchoring |
| `src/embedder.py`     | SigLIP 2 towers, and the recipe → embedder registry |
| `src/qwen_embedder.py`| Qwen3-VL text/image/**video** queries, via the tagger's embedder |

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
`normalize`, `kind`, `query_modes`, plus whatever else changes the vector
(`max_num_patches` for SigLIP 2; `prompt`, `fps`, `max_frames`, `max_length` for
Qwen). The vectorstore stores it as JSONB and returns it **verbatim on every
search hit**, so the recipe arrives on the same row as the vector and no second
lookup is needed — `read_recipes` parses what `get_vectors` already fetched. It
is opaque to the index though: not indexed and not filterable, so it can ride
along but cannot narrow a search.

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
- `src/embedder.py`'s image path runs the SigLIP tagger's `FeatureExtractor`,
  falling back to a local vision tower when the tagger is not importable. The
  two were checked against each other on a real frame: **bit-identical**, max
  absolute difference `0.000e+00`. Its *text* path has no counterpart to import
  — that tagger only ever loads the vision tower — so the text tower is
  necessarily query-side code.

Both resolve the tagger the same way: plain import first (`/elv` is the tagger
container's WORKDIR and already on `sys.path`), then `QWEN_EMBEDDING_PATH` /
`SIGLIP_EMBEDDING_PATH`, then a sibling checkout. No home paths.

**Modality** comes from the track's stamped `kind` when there is one. `kind` is
what the vectors *are*, which settles a case the field heuristic gets wrong: an
unsegmented whole-video tag carries `start == end == 0` and no `frame_idx`, so
the heuristic reads it as `unknown` and refuses to play the clip. Without a
recipe it falls back to which fields a row populates, in order: `text` → `text`;
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
- **Permissions are per-capability.** A token that reads `/meta/offerings` (200)
  can still be refused the frame rep (403, `q.read.bccall`: "no matching
  policy") — metadata read and bitcode call are separate grants, so an
  account-level `CreateFabricToken` is not necessarily enough to see media.

## Open items

- **Nothing is stamped yet.** The recipe path is wired and unit-tested against
  captured row shapes, but it has never parsed a real recipe: the tagger change
  lives only in `model-vector` locally and the frame index predates it, so its
  rows carry no `additional_info`. Re-tag content and detection takes over on
  its own; until then every index falls back to the declared model.
- **The Qwen path has never run.** `QwenQueryEmbedder` is wired, dispatched to
  and unit-tested for construction and parameter threading, but no query has
  been embedded through it — the Qwen index is empty, and the tagger's
  dependencies (`qwen-vl-utils`, `decord`) are not installed here. It will load
  the checkpoint on the first query against a Qwen index.
- **One recipe per index.** A query has to be embedded with one model, so if an
  index mixes tracks from different taggers the first recipe wins. All of them
  are returned in `recipes` so a mismatch is at least visible.
- **The SigLIP text tower is still query-side code.** The image half now runs the
  tagger's own `FeatureExtractor`, so that contract cannot drift. The text half
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
