/* Embeddings Visualizer — frontend.
 *
 * The server holds the vectors and sends only 2D coordinates plus metadata, so
 * everything here is layout and interaction; nothing in this file knows what a
 * 1024-dimensional vector looks like.
 *
 * One rule shapes the whole search interaction: the 2D positions are a *layout*,
 * not a measurement. Any projection distorts neighbourhoods, so a node sitting
 * next to the query on screen is not necessarily among its nearest vectors. The
 * server ranks neighbours by cosine similarity in the original space and we draw
 * those links explicitly, so viewers read similarity off the links rather than
 * off the distances.
 */

import {
  frameImageUrl,
  playoutUrl,
  playClip,
  needsToken,
  setContentToken,
  forgetContentToken,
  errorMessage,
} from "./media.js";

const API = "";

/* Mirrors the modality colours in style.css — change both together. */
const MODALITY = {
  // Mirrors style.css --m-*; see the CVD audit in the comment there.
  // text was [63, 208, 201]; video was [255, 111, 181] (pre-CVD-audit)
  text:    { label: "Text",    css: "--m-text",    rgb: [158, 250, 227] },
  image:   { label: "Image",   css: "--m-image",   rgb: [155, 125, 255] },
  video:   { label: "Video",   css: "--m-video",   rgb: [239,  90, 107] },
  unknown: { label: "Unknown", css: "--m-unknown", rgb: [107, 104, 128] },
};
/** A query node's colour: its modality's hue, lightened.
 *
 * A text query belongs beside the text vectors it is searching, so it takes
 * their hue rather than an unrelated one. Lightening keeps it separable from
 * the data — helped by the query node being three times the radius, stroked,
 * and carrying a number badge, which is also what tells two queries of the same
 * mode apart.
 */
function queryColor(mode) {
  const base = (MODALITY[mode] || MODALITY.unknown).rgb;
  return base.map((c) => Math.round(c + (255 - c) * 0.5));
}

const $ = (id) => document.getElementById(id);

const state = {
  indexKey: null,
  indexQid: null,
  points: [],
  bbox: null,
  modes: [],
  hidden: new Set(),   // modalities toggled off in the legend
  hovered: null,
  selected: null,
  selectedQuery: null,   // the detail panel shows a point or a query, never both
  cameFrom: null,        // the query a selected point was reached from, for "back"
  // Queries accumulate rather than replacing each other: comparing where two
  // searches land in one space is the point of plotting them at all.
  // Each: {id, n, x, y, mode, label, neighbours, pinned, linksVisible, alpha,
  //        timer, raf}
  queries: [],
  nextQueryN: 1,
  viewState: null,
  deck: null,
  client: undefined,   // undefined = not yet probed, null = standalone

  mediaToken: 0,       // guards against a slow selection overwriting a newer one
  mediaTeardown: null,
};

// Exposed for debugging from the console (and for the browser tests).
window.__state = state;

const LINK_HOLD_MS = 4000;
const LINK_FADE_MS = 700;
const ZOOM_STEP = 0.6;

/* ---------------------------------------------------------------- auth
 *
 * Inside elv-core-js the app is a sandboxed iframe with no keys of its own; it
 * asks core for a token over the FrameClient channel. Standalone (development)
 * there is no core to ask, so a token is kept in localStorage instead.
 *
 * The core path is deliberately a single seam — see README, plug-in stage.
 */
const TOKEN_DURATION_MS = 24 * 60 * 60 * 1000;

/** The core client, or null when running standalone. */
function frameClient() {
  if (state.client !== undefined) return state.client;
  // Same-window means no core to talk to; FrameClient would hang on its own
  // parent until the request timed out.
  const embedded = window.self !== window.top && window.FrameClient;
  state.client = embedded ? new window.FrameClient({ target: window.parent, timeout: 60 }) : null;
  return state.client;
}

/**
 * A token authorizing reads of `objectId` (the index object).
 *
 * Object-scoped rather than a bare account token: the vectorstore does not
 * check access itself, it forwards this to the fabric node holding the index
 * and relays the verdict, and that check wants a token naming the qid it is
 * being asked about. `CreateFabricToken` mints `acspjc…` — account and space
 * only, no `qid`, no `gra` — which comes back "no matching policy". The
 * `aessjc…` a signed token mints carries `qid`/`lib`/`gra: read` and is the
 * shape that works. Self-signed, so it scopes the account's rights rather than
 * granting any: if the account genuinely lacks read, this still fails.
 */
async function authToken(objectId) {
  const client = frameClient();
  if (client) {
    // Signed by core, which owns the keys. Note that FrameClient resolves with
    // the response itself — destructuring `{response}` off it silently yields
    // undefined.
    // return await client.CreateFabricToken({ duration: TOKEN_DURATION_MS });
    // libraryId is looked up from objectId by the client when omitted.
    return await client.CreateSignedToken({
      objectId,
      grantType: "read",
      duration: TOKEN_DURATION_MS,
    });
  }

  let token = localStorage.getItem("ev_auth_token");
  if (!token) {
    token = prompt("Fabric auth token (development only — core supplies this when embedded):");
    if (token) localStorage.setItem("ev_auth_token", token.trim());
  }
  return token && token.trim();
}

/* ---------------------------------------------------------------- helpers */

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** ms → m:ss.mmm, the form the fabric's time fields are read in. */
function formatTime(ms) {
  if (ms === null || ms === undefined) return "—";
  const sign = ms < 0 ? "-" : "";
  const t = Math.abs(ms);
  const m = Math.floor(t / 60000);
  const s = Math.floor((t % 60000) / 1000);
  return `${sign}${m}:${String(s).padStart(2, "0")}.${String(t % 1000).padStart(3, "0")}`;
}

/** "google/siglip2-base-patch16-naflex" -> "siglip2-base-patch16-naflex". */
function shortModel(id) {
  return truncate(String(id || "unknown").split("/").pop(), 34);
}

function truncate(value, max = 30) {
  const s = String(value);
  return s.length > max ? s.slice(0, max) + "…" : s;
}

/** The metadata rows worth showing, in a stable order, formatted for reading. */
function metaRows(meta, { full = false } = {}) {
  const rows = [];
  const push = (k, v) => { if (v !== null && v !== undefined && v !== "") rows.push([k, v]); };

  push("id", meta.id);
  push("qid", full ? meta.qid : truncate(meta.qid, 22));
  push("track", meta.track);
  if (meta.frame_idx !== null && meta.frame_idx !== undefined) push("frame", meta.frame_idx);

  // A frame is an instant, so its start and end are equal; collapse them into
  // one row rather than showing the same number twice.
  if (meta.start_time === meta.end_time) {
    push("time", formatTime(meta.start_time));
  } else {
    push("start", formatTime(meta.start_time));
    push("end", formatTime(meta.end_time));
  }
  push("source", full ? meta.source : truncate(meta.source, 22));
  if (full) push("batch", meta.batch_id);
  // Everything the tagger stamped, so the recipe is inspectable rather than
  // having to be inferred from behaviour.
  if (full && meta.additional_info && Object.keys(meta.additional_info).length) {
    for (const [k, v] of Object.entries(meta.additional_info)) {
      // A box reads as four numbers, not as JSON punctuation.
      if (k === "box" && v && typeof v === "object") {
        const n = (x) => (Number.isFinite(Number(x)) ? Number(x).toFixed(3) : "?");
        push("box", `${n(v.x1)}, ${n(v.y1)} → ${n(v.x2)}, ${n(v.y2)}`);
        continue;
      }
      push(k, truncate(typeof v === "object" ? JSON.stringify(v) : String(v), 40));
    }
  }
  return rows;
}

function dl(rows) {
  return `<dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;
}

/* ---------------------------------------------------------------- rendering */

function visible(p) {
  return !state.hidden.has(p.modality);
}

/** Grid lines spanning the visible world, at a spacing that stays legible as
 *  you zoom. Regenerated per view change: a fixed grid would either vanish or
 *  turn solid after a few zoom steps. */
function gridLines(viewState, width, height) {
  if (!viewState) return [];
  const scale = Math.pow(2, viewState.zoom);
  const halfW = width / 2 / scale;
  const halfH = height / 2 / scale;
  const [cx, cy] = viewState.target;

  // Aim for a line roughly every 80px, snapped to a 1/2/5 × 10^n step so the
  // spacing changes in readable jumps.
  const raw = 80 / scale;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) || mag * 10;

  const x0 = Math.floor((cx - halfW) / step) * step;
  const x1 = cx + halfW;
  const y0 = Math.floor((cy - halfH) / step) * step;
  const y1 = cy + halfH;

  const lines = [];
  for (let x = x0; x <= x1; x += step) lines.push({ from: [x, cy - halfH], to: [x, cy + halfH] });
  for (let y = y0; y <= y1; y += step) lines.push({ from: [cx - halfW, y], to: [cx + halfW, y] });
  return lines;
}

function hexToRgb(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [40, 36, 56];
}

function buildLayers() {
  const { ScatterplotLayer, LineLayer, TextLayer } = deck;
  const el = $("canvas");
  const width = el.clientWidth;
  const height = el.clientHeight;
  const gridRgb = hexToRgb(cssVar("--grid"));

  const shown = state.points.filter(visible);
  // Neighbours of every query whose links are currently up. Dimming is tied to
  // the links so the map goes back to reading as a plain distribution once they
  // fade, and so one query's neighbours do not hide another's.
  const neighbourIds = new Set();
  for (const q of state.queries) {
    if (!q.linksVisible) continue;
    for (const n of q.neighbours || []) neighbourIds.add(n.index);
  }
  const emphasize = neighbourIds.size > 0;

  const layers = [
    new LineLayer({
      id: "grid",
      data: gridLines(state.viewState, width, height),
      getSourcePosition: (d) => d.from,
      getTargetPosition: (d) => d.to,
      getColor: gridRgb,
      getWidth: 1,
      pickable: false,
    }),

    new ScatterplotLayer({
      id: "points",
      data: shown,
      getPosition: (d) => [d.x, d.y],
      getFillColor: (d) => {
        const rgb = (MODALITY[d.modality] || MODALITY.unknown).rgb;
        // While the links are up its true neighbours are the signal; fade
        // everything else so the ranking is visible at a glance.
        if (emphasize) return neighbourIds.has(d.i) ? [...rgb, 255] : [...rgb, 55];
        return [...rgb, 200];
      },
      getRadius: (d) => (d.i === state.selected?.i ? 6 : d.i === state.hovered?.i ? 5 : 3),
      radiusUnits: "pixels",
      // Re-evaluate the accessors when the things they close over change.
      updateTriggers: {
        getFillColor: [emphasize, neighbourIds.size, [...state.hidden].join()],
        getRadius: [state.hovered?.i, state.selected?.i],
      },
      pickable: true,
      autoHighlight: true,
      highlightColor: [255, 255, 255, 90],
      onHover: onHover,
      onClick: (info) => (info.object ? select(info.object) : null),
    }),
  ];

  if (state.queries.length) {
    // Every visible query's links go in one layer; each carries its own colour
    // and its own fade alpha, so the layers do not multiply with the queries.
    const links = [];
    for (const q of state.queries) {
      if (!q.linksVisible) continue;
      const rows = (q.neighbours || [])
        .map((n) => ({ n, p: state.points[n.index] }))
        .filter((d) => d.p);
      if (!rows.length) continue;

      // Shade by rank within this query's own results, not by absolute cosine.
      // SigLIP's cross-modal similarities are small by construction — a text
      // query's best match scores ~0.07 where an image query's scores ~0.5 — so
      // an absolute scale would render every text search nearly invisible.
      const sims = rows.map((d) => d.n.similarity);
      const lo = Math.min(...sims);
      const hi = Math.max(...sims);
      const rgb = queryColor(q.mode);
      for (const row of rows) {
        const k = hi > lo ? (row.n.similarity - lo) / (hi - lo) : 1;
        links.push({
          from: [q.x, q.y],
          to: [row.p.x, row.p.y],
          color: [...rgb, Math.round((70 + 185 * k) * q.alpha)],
          // Carried so a click on the link can open the node it points at, and
          // so the node's panel can offer a way back to this query.
          point: row.p,
          query: q,
          pinned: q.pinned,
        });
      }
    }

    if (links.length) {
      layers.push(
        new LineLayer({
          id: "query-links",
          data: links,
          getSourcePosition: (d) => d.from,
          getTargetPosition: (d) => d.to,
          getColor: (d) => d.color,
          getWidth: 1.5,
          pickable: false,
        })
      );

      // A second, invisible copy of the pinned links, fat enough to hit.
      //
      // The link is often easier to aim at than the node it ends on — a
      // neighbour in a dense cluster is a 3px dot among hundreds, while its
      // link has the whole run from the query to grab. deck.gl's picking pass
      // encodes indices rather than colours, so alpha 0 still picks; the
      // visible layer keeps its 1.5px line and this one carries the hit area.
      //
      // Pinned only: during the few seconds a link flashes after a search, a
      // fat invisible target over the map would swallow clicks meant for nodes.
      const pinnedLinks = links.filter((d) => d.pinned);
      if (pinnedLinks.length) {
        layers.push(
          new LineLayer({
            id: "query-links-hit",
            data: pinnedLinks,
            getSourcePosition: (d) => d.from,
            getTargetPosition: (d) => d.to,
            getColor: [0, 0, 0, 0],
            getWidth: 12,
            widthUnits: "pixels",
            pickable: true,
            onHover: (info) => {
              // Same feedback as hovering the node itself, so it is obvious
              // which end of the link you are about to open.
              const next = info.object ? info.object.point : null;
              if (next !== state.hovered) {
                state.hovered = next;
                render();
              }
            },
            onClick: (info) =>
              (info.object ? select(info.object.point, info.object.query) : null),
          })
        );
      }
    }

    layers.push(
      new ScatterplotLayer({
        id: "query-nodes",
        data: state.queries,
        getPosition: (d) => [d.x, d.y],
        getFillColor: (d) => [...queryColor(d.mode), 255],
        getLineColor: (d) => (d.pinned ? [255, 255, 255, 255] : [12, 10, 18, 255]),
        lineWidthUnits: "pixels",
        getLineWidth: 2,
        stroked: true,
        getRadius: 9,
        radiusUnits: "pixels",
        updateTriggers: {
          getLineColor: state.queries.map((q) => q.pinned).join(),
          getFillColor: state.queries.map((q) => q.mode).join(),
        },
        // Hovering re-reveals this query's links and names it; clicking pins
        // them so they survive the pointer leaving.
        pickable: true,
        onHover: onQueryHover,
        onClick: (info) => (info.object ? selectQuery(info.object) : null),
      }),
      // The number is what ties a node to its legend row, and what tells two
      // queries of the same mode apart once they share a colour.
      new TextLayer({
        id: "query-numbers",
        data: state.queries,
        getPosition: (d) => [d.x, d.y],
        getText: (d) => String(d.n),
        getSize: 11,
        getColor: [12, 10, 18, 255],
        fontWeight: 700,
        getTextAnchor: "middle",
        getAlignmentBaseline: "center",
        updateTriggers: { getText: state.queries.map((q) => q.n).join() },
        pickable: false,
      })
    );
  }

  return layers;
}

function render() {
  if (state.deck) state.deck.setProps({ layers: buildLayers(), viewState: state.viewState });
}

/* ---------------------------------------------------------------- link lifecycle */

/* Each query owns its own link state, so one fading out does not disturb
 * another that is pinned or mid-hover. */

function clearLinkTimers(q) {
  clearTimeout(q.timer);
  cancelAnimationFrame(q.raf);
  q.timer = null;
  q.raf = null;
}

/** Show this query's links and leave them up until something else decides. */
function holdLinks(q) {
  clearLinkTimers(q);
  q.linksVisible = true;
  q.alpha = 1;
  render();
}

/** Show them, then fade after a beat — what a fresh query does. */
function flashLinks(q) {
  holdLinks(q);
  q.timer = setTimeout(() => fadeLinks(q), LINK_HOLD_MS);
}

function fadeLinks(q) {
  if (q.pinned || !q.linksVisible) return;
  clearLinkTimers(q);
  const start = performance.now();
  const step = (now) => {
    const k = Math.min(1, (now - start) / LINK_FADE_MS);
    q.alpha = 1 - k;
    render();
    if (k < 1) {
      q.raf = requestAnimationFrame(step);
    } else {
      q.linksVisible = false;
      q.alpha = 1;
      render();
    }
  };
  q.raf = requestAnimationFrame(step);
}

function togglePin(q) {
  q.pinned = !q.pinned;
  q.pinned ? holdLinks(q) : fadeLinks(q);
  drawLegend();
  // The open node panel's back button is derived from what is pinned, so it has
  // to be rebuilt when that changes.
  if (state.selected) select(state.selected, state.cameFrom);
}

/** Stop a query's timers and release the blob URL holding its uploaded file. */
function disposeQuery(q) {
  clearLinkTimers(q);

  // Empty anything still showing the preview before revoking it: an <img> left
  // pointing at a revoked blob reports a failed load, which is console noise and
  // nothing else. The tooltip counts — it keeps the last thumbnail it rendered.
  if (state.selectedQuery?.id === q.id) {
    state.selectedQuery = null;
    $("detail").hidden = true;
    $("detail-body").innerHTML = "";
  }
  $("tip").hidden = true;
  $("tip").innerHTML = "";
  if (q.preview) {
    URL.revokeObjectURL(q.preview);
    q.preview = null;
  }
}

function removeQuery(id) {
  const q = state.queries.find((x) => x.id === id);
  if (q) disposeQuery(q);
  state.queries = state.queries.filter((x) => x.id !== id);
  drawLegend();
  render();
}

function clearQueries() {
  state.queries.forEach(disposeQuery);
  state.queries = [];
  state.nextQueryN = 1;
  drawLegend();
  render();
}

/* ---------------------------------------------------------------- zoom */

function zoomBy(delta) {
  if (!state.viewState) return;
  const { minZoom = -10, maxZoom = 40, zoom } = state.viewState;
  state.viewState = { ...state.viewState, zoom: Math.min(maxZoom, Math.max(minZoom, zoom + delta)) };
  render();
}

function zoomToFit() {
  if (!state.bbox) return;
  state.viewState = initialViewState(state.bbox);
  render();
}

/** Frame the fitted extent with a margin, so a fresh load fills the stage. */
function initialViewState(bbox) {
  const el = $("canvas");
  const w = Math.max(1, bbox.x_max - bbox.x_min);
  const h = Math.max(1, bbox.y_max - bbox.y_min);
  const zoom = Math.log2(Math.min(el.clientWidth / w, el.clientHeight / h) * 0.85);
  return {
    target: [(bbox.x_min + bbox.x_max) / 2, (bbox.y_min + bbox.y_max) / 2, 0],
    zoom: Number.isFinite(zoom) ? zoom : 0,
    minZoom: -10,
    maxZoom: 40,
  };
}

function initDeck() {
  if (state.deck) return;
  state.deck = new deck.Deck({
    parent: $("canvas"),
    views: new deck.OrthographicView({ id: "ortho" }),
    // Controlled rather than uncontrolled, so the zoom buttons can drive the
    // camera; deck then expects every change fed back through onViewStateChange.
    viewState: state.viewState,
    controller: { dragPan: true, scrollZoom: { smooth: true }, doubleClickZoom: false },
    onViewStateChange: ({ viewState }) => {
      state.viewState = viewState;
      // The grid is derived from the viewport, so it has to be rebuilt on move.
      render();
    },
    getCursor: ({ isDragging, isHovering }) =>
      isDragging ? "grabbing" : isHovering ? "pointer" : "grab",
    layers: [],
  });
}

/* ---------------------------------------------------------------- hover + select */

function onHover(info) {
  const tip = $("tip");
  state.hovered = info.object || null;

  if (!info.object) {
    tip.hidden = true;
    render();
    return;
  }

  const p = info.object;
  const mod = MODALITY[p.modality] || MODALITY.unknown;
  // A point can be a neighbour of several queries; show the strongest, tagged
  // with which query it belongs to.
  let sim = null;
  for (const q of state.queries) {
    const hit = q.neighbours?.find((n) => n.index === p.i);
    if (hit && (!sim || hit.similarity > sim.similarity)) sim = { ...hit, q };
  }

  showTip(
    `<div class="head">
       <span class="dot" style="background:${cssVar(mod.css)}"></span>
       <span class="mode">${mod.label}</span>
       ${sim ? `<span class="mode" style="margin-left:auto;color:${rgbCss(queryColor(sim.q.mode))}">
         q${sim.q.n} cos ${sim.similarity.toFixed(3)}</span>` : ""}
     </div>
     ${dl(metaRows(p.meta))}
     ${p.meta.text ? `<p class="text">${escapeHtml(truncate(p.meta.text, 160))}</p>` : ""}`,
    info
  );

  render();
}

/** Hover on the query node: name the query that made it, and re-reveal links. */
function onQueryHover(info) {
  const tip = $("tip");

  if (!info.object) {
    // Pointer left the query layer: every unpinned query goes back to fading.
    state.queries.forEach((q) => fadeLinks(q));
    tip.hidden = true;
    return;
  }

  const q = info.object;
  holdLinks(q);

  const best = q.neighbours?.[0];
  showTip(
    `<div class="head">
       <span class="dot" style="background:${rgbCss(queryColor(q.mode))}"></span>
       <span class="mode">${q.n} · ${q.mode} query</span>
     </div>
     ${q.preview && q.mode === "image"
        ? `<img class="query-thumb" src="${q.preview}" alt="">`
        : ""}
     <p class="query-label">${escapeHtml(q.label || "")}</p>
     ${dl([
       ["neighbours", q.neighbours?.length ?? 0],
       ["best cos", best ? best.similarity.toFixed(3) : "—"],
     ])}
     <p class="text">Click to open this query and pin its links.</p>`,
    info
  );
}

function rgbCss(rgb) {
  return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
}

/** Place the tooltip near the cursor, flipping before it leaves the stage. */
function showTip(html, info) {
  const tip = $("tip");
  tip.innerHTML = html;
  const pad = 14;
  const stage = $("stage").getBoundingClientRect();
  tip.hidden = false;
  const box = tip.getBoundingClientRect();
  const x = info.x + pad + box.width > stage.width ? info.x - box.width - pad : info.x + pad;
  const y = info.y + pad + box.height > stage.height ? info.y - box.height - pad : info.y + pad;
  tip.style.left = `${Math.max(0, x)}px`;
  tip.style.top = `${Math.max(0, y)}px`;
}

/** Open the detail panel on a query, showing what was actually searched with.
 *
 * For an uploaded image or clip the file itself is shown, not its name — the
 * name says nothing about whether the right file was picked, and the whole
 * question a viewer has about an image query is what the image was.
 */
function selectQuery(q) {
  state.selected = null;
  state.selectedQuery = q;
  state.cameFrom = null;
  $("search-panel").hidden = true;
  teardownMedia();

  // Opening a query pins its links: the panel and the links answer the same
  // question, so having them come and go separately just makes work.
  if (!q.pinned) {
    q.pinned = true;
    holdLinks(q);
  }

  const best = q.neighbours?.[0];
  $("detail-body").innerHTML = `
    <h2>
      <span class="dot" style="background:${rgbCss(queryColor(q.mode))}"></span>
      ${q.n} · ${MODALITY[q.mode]?.label || q.mode} query
    </h2>
    ${queryMediaHtml(q)}
    <div class="section">Query</div>
    ${dl([
      ["mode", q.mode],
      ["neighbours", q.neighbours?.length ?? 0],
      ["best cos", best ? best.similarity.toFixed(3) : "—"],
    ])}
    <div class="section">Nearest neighbours</div>
    ${neighbourList(q)}
    <button id="query-pin" class="wide">${q.pinned ? "Unpin links" : "Pin links"}</button>`;
  $("detail").hidden = false;
  drawLegend();
  render();

  wireNeighbourList(q);
  $("query-pin").addEventListener("click", () => {
    togglePin(q);
    $("query-pin").textContent = q.pinned ? "Unpin links" : "Pin links";
  });
}

/** The ranked neighbours, as rows you can hover and click.
 *
 * The map answers "where are they"; this answers "which are they", which the
 * map cannot when a neighbour sits inside a dense cluster. Ranked by cosine in
 * the original space — the same ranking the links draw, and the only ranking
 * that means anything, since 2D distance is layout.
 */
function neighbourList(q) {
  const rows = (q.neighbours || [])
    .map((n) => ({ n, p: state.points[n.index] }))
    .filter((d) => d.p);
  if (!rows.length) return `<p class="hint">No neighbours were returned.</p>`;

  return `<ol class="neighbours">${rows
    .map(({ n, p }, rank) => {
      const rgb = (MODALITY[p.modality] || MODALITY.unknown).rgb;
      return `<li data-i="${p.i}">
        <span class="rank">${rank + 1}</span>
        <span class="dot" style="background:${rgbCss(rgb)}"></span>
        <span class="what">${escapeHtml(neighbourLabel(p))}</span>
        <span class="cos">${n.similarity.toFixed(3)}</span>
      </li>`;
    })
    .join("")}</ol>`;
}

/** Enough to tell two neighbours apart: what it is and where on the timeline. */
function neighbourLabel(p) {
  const m = p.meta || {};
  if (p.modality === "text" && m.text) return truncate(m.text, 30);
  if (m.frame_idx !== null && m.frame_idx !== undefined) {
    return `frame ${m.frame_idx} @ ${formatTime(m.start_time)}`;
  }
  return formatTime(m.start_time);
}

function wireNeighbourList(q) {
  for (const row of document.querySelectorAll("#detail .neighbours li")) {
    const point = state.points[Number(row.dataset.i)];
    if (!point) continue;
    // Hover highlights on the map, so the list can be scanned without losing
    // the query panel; only a click commits to opening the node.
    row.addEventListener("mouseenter", () => { state.hovered = point; render(); });
    row.addEventListener("mouseleave", () => { state.hovered = null; render(); });
    row.addEventListener("click", () => {
      centreOn(point);
      select(point, q);
    });
  }
}

/** The pinned query whose neighbours include this point, or null.
 *
 * `preferred` wins when it qualifies, so arriving from one of several pinned
 * queries goes back to that one rather than to whichever is first.
 */
function pinnedQueryFor(point, preferred = null) {
  const links = (q) =>
    q && q.pinned && (q.neighbours || []).some((n) => n.index === point.i);
  if (links(preferred)) return preferred;
  return state.queries.find(links) || null;
}

/** Pan to a point, keeping the zoom, so a click on a row goes to it. */
function centreOn(p) {
  if (!state.viewState) return;
  state.viewState = { ...state.viewState, target: [p.x, p.y, 0] };
}

function queryMediaHtml(q) {
  if (q.mode === "text") {
    return `<p class="quote">${escapeHtml(q.label || "")}</p>`;
  }
  if (!q.preview) {
    // Only reachable if the blob URL was released while the query lived on.
    return `<div class="placeholder">${escapeHtml(q.label || "uploaded file")}</div>`;
  }
  const media =
    q.mode === "video"
      ? `<video src="${q.preview}" controls playsinline></video>`
      : `<img src="${q.preview}" alt="${escapeHtml(q.label || "query image")}">`;
  return `${media}<p class="caption">${escapeHtml(truncate(q.label || "", 40))}</p>`;
}

function select(point, fromQuery = null) {
  state.selected = point;
  state.selectedQuery = null;
  $("search-panel").hidden = true;
  teardownMedia();

  const mod = MODALITY[point.modality] || MODALITY.unknown;
  const meta = point.meta;
  // Shown whenever a *pinned* query links to this node, not only when the node
  // was opened through one. Keyed on how the node was reached, two neighbours
  // of the same pinned query would disagree — one clicked from the list has a
  // way back, the same node clicked on the map does not — which reads as a bug.
  const back = pinnedQueryFor(point, fromQuery);
  state.cameFrom = back;

  $("detail-body").innerHTML = `
    ${back ? `<button id="detail-back" class="back">← Query ${back.n} ·
        ${MODALITY[back.mode]?.label || back.mode}</button>` : ""}
    <h2><span class="dot" style="background:${cssVar(mod.css)}"></span>${mod.label} vector</h2>
    <div id="detail-media">${mediaSkeleton(point)}</div>
    <div class="section">Metadata</div>
    ${dl(metaRows(meta, { full: true }))}`;
  $("detail").hidden = false;
  render();

  if (back) {
    $("detail-back").addEventListener("click", () => selectQuery(back));
  }
  wireUnlock(point);
  loadMedia(point);
}

/** Release the previous clip before another is attached or the panel closes. */
function teardownMedia() {
  if (state.mediaTeardown) {
    try { state.mediaTeardown(); } catch (e) { /* already gone */ }
    state.mediaTeardown = null;
  }
}

/** What the media area shows before (or instead of) the fabric answers.
 *
 * Text is carried in the metadata itself, so it is final immediately. Unknown
 * modality never attempts media — the metadata is the content.
 */
function mediaSkeleton(point) {
  const meta = point.meta;

  if (point.modality === "text") {
    return `<p class="quote">${escapeHtml(meta.text || "")}</p>`;
  }
  if (point.modality === "unknown") {
    return `<div class="placeholder">${unknownReason(meta)}</div>`;
  }
  if (needsToken(frameClient(), meta.qid)) {
    // Standalone there is no account to authorize against, so the viewer
    // supplies a token for this content object. Asked once per object, not
    // once per node — an index can hold vectors from several.
    return unlockForm(meta.qid);
  }

  return `<div class="placeholder loading">
    <span class="spinner"></span> Resolving media…
  </div>`;
}

function unlockForm(qid, message) {
  return `
    <div class="unlock">
      ${message ? `<p class="error">${escapeHtml(message)}</p>` : ""}
      <p>Media for <b>${escapeHtml(truncate(qid, 26))}</b> needs a token granting
         access to that content object.</p>
      <input id="content-token" type="password" placeholder="aessj…"
             spellcheck="false" autocomplete="off">
      <button id="content-token-go" class="primary">Unlock media</button>
      <p class="hint">Stored in this browser only, and reused for every vector
         from this object.</p>
    </div>`;
}

/** Save a pasted content token and retry the media that was blocked on it. */
function wireUnlock(point) {
  const input = $("content-token");
  if (!input) return;

  const submit = () => {
    const token = input.value.trim();
    if (!token) return;
    setContentToken(point.meta.qid, token);
    $("detail-media").innerHTML = `<div class="placeholder loading">
      <span class="spinner"></span> Resolving media…
    </div>`;
    loadMedia(point);
  };

  $("content-token-go").addEventListener("click", submit);
  input.addEventListener("keydown", (e) => e.key === "Enter" && submit());
  input.focus();
}

/** Why a row could not be classified, in terms of the field that is wrong.
 *
 * "No recognisable media fields" was true but unhelpful: by far the most common
 * cause is a time range that is not a range, and saying so points at the tagger
 * rather than leaving the vector looking merely unsupported.
 */
function unknownReason(meta) {
  const hasStart = meta.start_time !== null && meta.start_time !== undefined;
  const hasEnd = meta.end_time !== null && meta.end_time !== undefined;

  if (hasStart && !hasEnd) {
    return `This vector has a <b>start_time</b> but no <b>end_time</b>, so there
      is no clip to play.<br><small>The tagger did not record an extent for it.
      Showing metadata only.</small>`;
  }
  if (hasStart && hasEnd && meta.end_time <= meta.start_time) {
    const same = meta.end_time === meta.start_time;
    return `This vector's <b>end_time</b> ${same ? "equals" : "precedes"} its
      <b>start_time</b> (${formatTime(meta.start_time)}), so it describes no
      extent.<br><small>Whole-media tags once carried
      <code>start == end == 0</code> as a sentinel, which the pipeline re-based
      into both fields; a row like this was written before that was fixed and
      needs re-tagging. Showing metadata only.</small>`;
  }
  return `No recognisable media fields.<br><small>No text, no frame index and no
    time range, so there is nothing to fetch. Showing metadata only.</small>`;
}

/** The detection box a crop vector was embedded from, or null.
 *
 * Taggers that embed crops repeat the box into `additional_info` precisely
 * because that is the only per-vector field a search row carries back — the
 * `box` on the tag itself lands in `frame_info`, which the index does not
 * store. Coordinates are normalized to [0, 1] against the full frame, so they
 * map onto the frame image without knowing its pixel size.
 *
 * A whole-frame vector carries {0,0,1,1}: a real box, but drawing a rectangle
 * around the entire frame says nothing, so it is treated as "no box".
 */
function detectionBox(meta) {
  const b = meta?.additional_info?.box;
  if (!b) return null;
  const [x1, y1, x2, y2] = [b.x1, b.y1, b.x2, b.y2].map(Number);
  if (![x1, y1, x2, y2].every(Number.isFinite)) return null;
  if (x2 <= x1 || y2 <= y1) return null;
  const wholeFrame = x1 <= 0.001 && y1 <= 0.001 && x2 >= 0.999 && y2 >= 0.999;
  return wholeFrame ? null : { x1, y1, x2, y2 };
}

/** The box as a percentage-positioned overlay, so it scales with the image. */
function boxOverlay(box, label) {
  const pct = (v) => `${(v * 100).toFixed(3)}%`;
  const style = `left:${pct(box.x1)};top:${pct(box.y1)};` +
    `width:${pct(box.x2 - box.x1)};height:${pct(box.y2 - box.y1)}`;
  return `<div class="det-box" style="${style}">${
    label ? `<span>${escapeHtml(label)}</span>` : ""}</div>`;
}

/** How a clip's extent reads.
 *
 * A row only reaches here as `video`, which requires end_time > start_time, so
 * the extent is always real. Rows without one are `unknown` and never get a
 * player — see `unknownReason`.
 */
function clipRange(meta) {
  return `Clip ${formatTime(meta.start_time)} – ${formatTime(meta.end_time)}`;
}


/** Fetch the frame or the clip and swap it into the open panel. */
async function loadMedia(point) {
  const client = frameClient();
  if (point.modality === "text" || point.modality === "unknown") return;
  if (needsToken(client, point.meta.qid)) return;   // waiting on the unlock form

  // Selections can outrun the fabric; only the newest one may write.
  const token = ++state.mediaToken;
  const host = $("detail-media");
  const meta = point.meta;

  try {
    if (point.modality === "image") {
      const url = await frameImageUrl(client, meta.qid, (meta.start_time || 0) / 1000);
      if (token !== state.mediaToken) return;
      if (!url) return mediaFailed(host, "This object has no video offering.");
      const box = detectionBox(meta);
      // `is-loading` holds the spinner in the space the frame will occupy.
      // Having the URL is not having the image: the fabric still has to send
      // it, and without this the panel goes blank for those seconds.
      host.innerHTML = `
        <div class="frame is-loading">
          <img alt="Frame ${meta.frame_idx}" src="${url}">
          ${box ? boxOverlay(box, meta.tag) : ""}
          <div class="media-spinner"><span class="spinner"></span></div>
        </div>
        <p class="caption">Frame ${meta.frame_idx} @ ${formatTime(meta.start_time)}${
          box ? " — box shown" : ""}</p>`;
      const img = host.querySelector("img");
      const frame = host.querySelector(".frame");
      img.addEventListener("load", () => frame.classList.remove("is-loading"), { once: true });
      img.addEventListener("error", () =>
        mediaFailed(host, "The fabric refused this frame."));
      return;
    }

    const url = await playoutUrl(client, meta.qid);
    if (token !== state.mediaToken) return;
    if (!url) return mediaFailed(host, "This object has no playable offering.");

    // Same reservation as the frame: a manifest still has to be fetched and
    // parsed before the first frame paints.
    host.innerHTML = `
      <div class="frame is-loading">
        <video id="clip" controls playsinline></video>
        <div class="media-spinner"><span class="spinner"></span></div>
      </div>
      <p class="caption">
        ${clipRange(meta)}
        <button id="clip-replay">Replay</button>
      </p>`;

    const video = $("clip");
    const wrap = host.querySelector(".frame");
    // loadeddata, not loadedmetadata: metadata arrives before any pixels do.
    video.addEventListener("loadeddata", () => wrap.classList.remove("is-loading"));
    const attach = () => {
      wrap.classList.add("is-loading");
      teardownMedia();
      state.mediaTeardown = playClip(video, url, meta.start_time, meta.end_time);
    };
    $("clip-replay").addEventListener("click", attach);
    attach();
  } catch (err) {
    if (token !== state.mediaToken) return;
    // errorMessage, not err.message: FrameClient rejects with core's raw error
    // object or a bare string, and reading .message off those renders
    // "undefined" in place of the actual failure.
    // if (client) return mediaFailed(host, err.message);
    if (client) return mediaFailed(host, errorMessage(err));
    // A rejected token should not be sticky — drop it and ask for another,
    // rather than leaving the panel permanently broken for this object.
    forgetContentToken(meta.qid);
    host.innerHTML = unlockForm(meta.qid, errorMessage(err));
    wireUnlock(point);
  }
}

function mediaFailed(host, message) {
  host.innerHTML = `<div class="placeholder">${escapeHtml(message)}</div>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------------------------------------------------------------- legend + stats */

function drawLegend() {
  const counts = {};
  for (const p of state.points) counts[p.modality] = (counts[p.modality] || 0) + 1;

  const legend = $("legend");
  let html = Object.entries(MODALITY)
    .filter(([key]) => counts[key])
    .map(([key, m]) => `
      <button data-modality="${key}" class="${state.hidden.has(key) ? "off" : ""}">
        <span class="dot" style="background:${cssVar(m.css)}"></span>
        <span>${m.label}</span>
        <span class="n">${counts[key].toLocaleString()}</span>
      </button>`)
    .join("");

  // One row per query, carrying the number that appears on its node and the
  // query text itself — with several on the map at once, the mode alone no
  // longer identifies which is which. The row toggles that query's links; the
  // × removes it.
  state.queries.forEach((q, i) => {
    html += `
      <div class="query-row${i === 0 ? " first" : ""}">
        <button data-query="${q.id}" title="Show or hide this query's links">
          <span class="dot" style="background:${rgbCss(queryColor(q.mode))}"></span>
          <span class="qn">${q.n}</span>
          <span class="qlabel">${escapeHtml(truncate(q.label || q.mode, 22))}</span>
          <span class="n">${q.pinned ? "pinned" : "links"}</span>
        </button>
        <button class="remove" data-remove="${q.id}" title="Remove this query"
                aria-label="Remove query ${q.n}">×</button>
      </div>`;
  });

  if (state.queries.length > 1) {
    html += `<button data-clear="1" class="clear-queries">Clear all queries</button>`;
  }

  legend.innerHTML = html;
  legend.hidden = false;

  legend.querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => {
      if (b.dataset.remove) return removeQuery(b.dataset.remove);
      if (b.dataset.clear) return clearQueries();
      if (b.dataset.query) {
        const q = state.queries.find((x) => x.id === b.dataset.query);
        if (q) togglePin(q);
        return;
      }
      const key = b.dataset.modality;
      state.hidden.has(key) ? state.hidden.delete(key) : state.hidden.add(key);
      drawLegend();
      render();
    }));
}

/** Reflect whether the model/modes were detected or declared by hand.
 *
 * The model comes from the batch a vector was written in, which is
 * authoritative, so the manual checkboxes are disabled rather than left looking
 * like they still decide anything. `model_id` is "unknown" when no batch could
 * be read — then the declared modes are all there is.
 */
function applyDetectedModel(data) {
  const model = data.model_id && data.model_id !== "unknown" ? data.model_id : null;

  document.querySelectorAll("#modes input").forEach((input) => {
    input.disabled = !!model;
    if (model) input.checked = data.modes.includes(input.value);
  });
  $("modes-field").title = model
    ? `Detected from the index's batches: ${model}`
    : "No batch reported a model for this index, so declare its query modes here.";
  $("modes-field").classList.toggle("detected", !!model);
  return model;
}

function drawStats(data) {
  const model = applyDetectedModel(data);
  const parts = [
    `<b>${data.count.toLocaleString()}</b> vectors · <b>${data.vector_size}</b> dims · ${data.method.toUpperCase()}`,
  ];

  // Naming the source matters: "declared" means nothing verified that the model
  // shown is the one that built these vectors.
  if (model) {
    const tuned = Object.keys(data.tuning || {}).length;
    parts.push(`model <b>${escapeHtml(model)}</b> · from batch${
      tuned ? ` · ${tuned} stamped parameter${tuned === 1 ? "" : "s"}` : ""}`);
  } else {
    parts.push(`model <b>unknown</b> · <span class="warn">no batch reported one</span>`);
  }
  if (data.method === "pca") {
    // Worth surfacing: two PCA components typically retain only ~20% of the
    // variance, which is why the cloud looks undifferentiated.
    parts.push(`retains <b>${(data.explained_variance * 100).toFixed(1)}%</b> of variance`);
  } else {
    parts.push(`<span class="warn">2D distances are layout only</span>`);
  }
  $("stats").innerHTML = parts.join("<br>");
  $("stats").hidden = false;
}

/* ---------------------------------------------------------------- load */

function busy(on, text) {
  $("busy").hidden = !on;
  if (text) $("busy-text").textContent = text;
}

async function loadIndex() {
  const qid = $("index-qid").value.trim();
  if (!qid) return alert("An index QID is required.");

  const token = await authToken(qid);
  if (!token) return alert("An auth token is required.");

  const modes = [...document.querySelectorAll("#modes input:checked")].map((i) => i.value);
  // const sources = $("sources").value.split(",").map((s) => s.trim()).filter(Boolean);
  const sources = [];   // the header no longer exposes a sources filter
  const method = document.querySelector("#method .on").dataset.method;

  $("empty").hidden = true;
  busy(true, `Fetching and projecting vectors (${method.toUpperCase()})…`);
  $("load").disabled = true;

  try {
    const res = await fetch(`${API}/api/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({
        index_qid: qid,
        sources,
        modes,
        method,
        sample_size: Number($("sample-size").value) || 10000,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `request failed (${res.status})`);

    state.indexKey = data.index_key;
    state.indexQid = data.index_qid;
    state.points = data.points;
    state.bbox = data.bbox;
    state.modes = data.modes;
    state.selected = null;
    state.hidden.clear();
    // Coordinates come from a projection fitted to this index, so queries
    // plotted against the previous one mean nothing here.
    state.queries.forEach(disposeQuery);
    state.queries = [];
    state.nextQueryN = 1;
    state.selectedQuery = null;
    $("detail").hidden = true;
    $("search-error").hidden = true;
    $("search-note").hidden = true;

    state.viewState = initialViewState(data.bbox);
    initDeck();
    $("zoom").hidden = false;
    drawLegend();
    drawStats(data);
    render();
  } catch (err) {
    $("empty").hidden = false;
    $("empty").querySelector("h1").textContent = "Could not load index";
    $("empty").querySelector("p").textContent = err.message;
  } finally {
    busy(false);
    $("load").disabled = false;
  }
}

/* ---------------------------------------------------------------- search */

async function runSearch(mode, payload, label, preview) {
  if (!state.indexKey) return alert("Load an index first.");

  const err = $("search-error");
  const note = $("search-note");
  err.hidden = true;
  note.hidden = true;
  busy(true, "Embedding query…");

  try {
    const res = await fetch(`${API}/api/search/${mode}`, payload);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `request failed (${res.status})`);

    const q = {
      id: `q${state.nextQueryN}`,
      n: state.nextQueryN++,
      x: data.query_point.x,
      y: data.query_point.y,
      mode: data.mode,
      label,
      preview,
      neighbours: data.neighbours,
      pinned: false,
      linksVisible: false,
      alpha: 1,
      timer: null,
      raf: null,
    };
    // A new array, not a push: deck.gl compares `data` by reference, so
    // mutating in place leaves the layer rendering the old point count.
    state.queries = [...state.queries, q];
    drawLegend();
    // Up for a beat, then out of the way; the query node brings them back.
    flashLinks(q);

    const best = data.neighbours[0];
    // The absolute number is worth showing but needs its scale named: SigLIP
    // scores a text-to-image match an order of magnitude lower than an
    // image-to-image one, so 0.07 from a text query is a strong hit.
    note.innerHTML = `Query placed. Links mark its <b>${data.neighbours.length}</b> nearest
      vectors by true cosine similarity, not by distance on screen.
      Best ${best ? `<b>${best.similarity.toFixed(3)}</b>` : "—"} — compare within a
      ${mode} search, not across modes. Links fade after a moment — hover or
      click the query node to bring them back.`;
    note.hidden = false;
    render();
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
  } finally {
    busy(false);
  }
}

function searchText() {
  const query = $("q-text").value.trim();
  if (!query) return;
  return runSearch(
    "text",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index_key: state.indexKey, query }),
    },
    query
  );
}

function searchFile(mode, input) {
  const file = input.files?.[0];
  if (!file) return;
  const form = new FormData();
  form.append("index_key", state.indexKey);
  form.append("file", file);
  // Kept so the query node can show what was actually uploaded rather than its
  // filename. Revoked in disposeQuery.
  return runSearch(mode, { method: "POST", body: form }, file.name, URL.createObjectURL(file));
}

/* ---------------------------------------------------------------- wiring */

$("load").addEventListener("click", loadIndex);
$("index-qid").addEventListener("keydown", (e) => e.key === "Enter" && loadIndex());

document.querySelectorAll("#method button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll("#method button").forEach((o) => o.classList.remove("on"));
    b.classList.add("on");
  }));

const showHelp = (on) => { $("help").hidden = !on; };
$("help-toggle").addEventListener("click", () => showHelp($("help").hidden));
$("help-close").addEventListener("click", () => showHelp(false));
// Clicks land on the backdrop only when they miss the card.
$("help").addEventListener("click", (e) => { if (e.target.id === "help") showHelp(false); });

$("zoom-in").addEventListener("click", () => zoomBy(ZOOM_STEP));
$("zoom-out").addEventListener("click", () => zoomBy(-ZOOM_STEP));
$("zoom-fit").addEventListener("click", zoomToFit);

$("detail-close").addEventListener("click", () => {
  $("detail").hidden = true;
  state.selected = null;
  state.selectedQuery = null;
  teardownMedia();
  render();
});

// The detail panel and the search panel share the top-right anchor, so only one
// is ever open.
$("search-toggle").addEventListener("click", () => {
  const panel = $("search-panel");
  panel.hidden = !panel.hidden;
  if (!panel.hidden) {
    $("detail").hidden = true;
    state.selected = null;
    render();
    $("q-text").focus();
  }
});

document.querySelectorAll("#search-tabs button").forEach((b) =>
  b.addEventListener("click", () => {
    const mode = b.dataset.mode;
    document.querySelectorAll("#search-tabs button").forEach((o) => o.classList.remove("on"));
    b.classList.add("on");
    document.querySelectorAll("#search-panel .pane").forEach((p) => {
      p.hidden = p.dataset.pane !== mode;
    });

    // The unsupported-mode message is the server's to give — it owns which
    // modes the index was declared to support — but showing it on tab switch
    // saves the viewer from picking a file only to be refused afterwards.
    const err = $("search-error");
    if (state.indexKey && !state.modes.includes(mode)) {
      err.textContent = `Index does not support ${mode} embeddings.`;
      err.hidden = false;
    } else {
      err.hidden = true;
    }
  }));

$("q-text-go").addEventListener("click", searchText);
$("q-text").addEventListener("keydown", (e) => e.key === "Enter" && searchText());
$("q-image").addEventListener("change", (e) => searchFile("image", e.target));
$("q-video").addEventListener("change", (e) => searchFile("video", e.target));

// deck only reports hover-out while the pointer is still over its canvas, so a
// pointer that leaves for a panel would strand the tooltip on screen.
$("canvas").addEventListener("mouseleave", () => {
  $("tip").hidden = true;
  state.hovered = null;
  state.queries.forEach((q) => fadeLinks(q));
  render();
});

window.addEventListener("resize", render);
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  $("help").hidden = true;
  $("detail").hidden = true;
  $("search-panel").hidden = true;
  state.selected = null;
  state.selectedQuery = null;
  teardownMedia();
  render();
});
