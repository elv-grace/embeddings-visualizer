/* Media resolution: turning a vector's metadata into something watchable.
 *
 * A vector's metadata names a content object (`qid`) and a position on its
 * timeline (`start_time`/`end_time` in ms, `frame_idx` for frames). Neither a
 * frame image nor a playable stream is addressable from that alone — both need
 * a URL authorized against the object.
 *
 * Two transports, one shape
 * -------------------------
 * Inside core, the FrameClient signs URLs against the viewer's account; frames
 * go through `Rep(..., channelAuth: true)` and clips through `PlayoutOptions`.
 * Standalone there is no account, so the viewer pastes a token for the content
 * object and the URLs are built directly against a fabric node with
 * `?authorization=`. Both paths pick the offering the same way, and the token
 * path is skipped entirely whenever core is available.
 *
 * Offerings are not interchangeable
 * ---------------------------------
 * An object can carry several, and the obvious one is often unplayable: on the
 * test content `default` publishes only DRM formats (widevine, fairplay,
 * sample-aes) and 404s for HLS, while `default_clear` publishes `hls-clear`.
 * So playout picks an offering by what it actually publishes, not by name.
 * Frames are less fussy — every offering serves them — so those keep the
 * name preference.
 *
 * The HLS manifest embeds the authorization in every child URI, so the player
 * carries auth into variant playlists and segments without any help.
 */

// Overridable so this can be pointed at another network without a rebuild.
const CONFIG_URL =
  localStorage.getItem("ev_config_url") || "https://main.net955305.contentfabric.io/config";

const TOKEN_KEY = "ev_content_tokens";
const PLAYABLE_HLS = ["hls-clear", "hls-aes128"];

const RESOLVED = new Map();   // qid -> {offerings, frame, playout, format}
let nodePromise = null;

/* ---------------------------------------------------------------- tokens */

function tokenStore() {
  try {
    return JSON.parse(localStorage.getItem(TOKEN_KEY)) || {};
  } catch (e) {
    return {};
  }
}

export function contentToken(qid) {
  return tokenStore()[qid] || null;
}

export function setContentToken(qid, token) {
  const store = tokenStore();
  store[qid] = (token || "").trim();
  localStorage.setItem(TOKEN_KEY, JSON.stringify(store));
  // A new token can change which offerings are visible, so drop what was
  // resolved under the old one.
  RESOLVED.delete(qid);
}

export function forgetContentToken(qid) {
  const store = tokenStore();
  delete store[qid];
  localStorage.setItem(TOKEN_KEY, JSON.stringify(store));
  RESOLVED.delete(qid);
}

/** True when this object cannot be reached yet: no core, and no token saved. */
export function needsToken(client, qid) {
  return !client && !contentToken(qid);
}

/* ---------------------------------------------------------------- fabric node */

async function fabricNode() {
  if (!nodePromise) {
    nodePromise = (async () => {
      const res = await fetch(CONFIG_URL);
      if (!res.ok) throw new Error(`could not read fabric config (${res.status})`);
      const config = await res.json();
      const nodes = config?.network?.services?.fabric_api || [];
      if (!nodes.length) throw new Error("fabric config lists no API nodes");
      return nodes[0].replace(/\/$/, "");
    })().catch((err) => {
      nodePromise = null;   // let the next attempt retry rather than cache the failure
      throw err;
    });
  }
  return nodePromise;
}

/* ---------------------------------------------------------------- offerings */

async function fetchOfferings(client, qid) {
  if (client) {
    return (await client.ContentObjectMetadata({ objectId: qid, metadataSubtree: "offerings" })) || {};
  }

  const token = contentToken(qid);
  if (!token) throw new Error("no token for this content object");

  const node = await fabricNode();
  const res = await fetch(`${node}/q/${qid}/meta/offerings?authorization=${encodeURIComponent(token)}`);
  if (res.status === 403) throw new Error("This token does not grant access to that content.");
  if (!res.ok) throw new Error(`could not read offerings (${res.status})`);
  return (await res.json()) || {};
}

/** Prefer an exact "default", then anything containing it, then alphabetical. */
function preferred(names) {
  return (
    names.find((n) => n === "default") ||
    names.find((n) => n.includes("default")) ||
    [...names].sort()[0] ||
    null
  );
}

async function resolve(client, qid) {
  if (RESOLVED.has(qid)) return RESOLVED.get(qid);

  const offerings = await fetchOfferings(client, qid);
  const names = Object.keys(offerings);

  // Only offerings that publish an unencrypted HLS format can be played here.
  let format = null;
  const playable = names.filter((name) => {
    const formats = Object.keys(offerings[name]?.playout?.playout_formats || {});
    const match = PLAYABLE_HLS.find((f) => formats.includes(f));
    if (match) format = format || match;
    return !!match;
  });

  const entry = { offerings, frame: preferred(names), playout: preferred(playable), format };
  RESOLVED.set(qid, entry);
  return entry;
}

/* ---------------------------------------------------------------- urls */

/** A JPEG of the frame at `seconds`, or null when the object serves no frames. */
export async function frameImageUrl(client, qid, seconds) {
  const { frame } = await resolve(client, qid);
  if (!frame) return null;
  const t = Number(seconds).toFixed(2);

  if (client) {
    const base = await client.Rep({
      objectId: qid,
      rep: `frame/${frame}/video`,
      channelAuth: true,
      // Without this the frame is addressed against the trimmed timeline, which
      // is not the timeline the tagger recorded timestamps against.
      queryParams: { ignore_trimming: true },
    });
    const url = new URL(base);
    url.searchParams.set("t", t);
    return url.toString();
  }

  const node = await fabricNode();
  const token = contentToken(qid);
  return `${node}/q/${qid}/rep/frame/${frame}/video` +
    `?t=${t}&ignore_trimming=true&authorization=${encodeURIComponent(token)}`;
}

/** An HLS manifest URL, or null when nothing unencrypted is published. */
export async function playoutUrl(client, qid) {
  const { playout, format } = await resolve(client, qid);
  if (!playout) return null;

  if (client) {
    let drms = [];
    try {
      drms = ((await client.AvailableDRMs()) || []).filter((d) => ["clear", "aes-128"].includes(d));
    } catch (e) {
      drms = [];
    }
    const options = await client.PlayoutOptions({
      objectId: qid,
      protocols: ["hls"],
      drms,
      offering: playout,
      hlsjsProfile: false,
    });
    const methods = options?.hls?.playoutMethods || {};
    return methods.clear?.playoutUrl || methods["aes-128"]?.playoutUrl || null;
  }

  const node = await fabricNode();
  const token = contentToken(qid);
  return `${node}/q/${qid}/rep/playout/${playout}/${format}/playlist.m3u8` +
    `?authorization=${encodeURIComponent(token)}`;
}

/* ---------------------------------------------------------------- playback */

/** Attach a source to a <video> and hold it to [startMs, endMs].
 *
 * Returns a teardown function; call it before attaching another clip, or the
 * detached hls.js instance keeps buffering in the background.
 */
export function playClip(video, url, startMs, endMs) {
  const start = (startMs || 0) / 1000;
  const end = endMs > startMs ? endMs / 1000 : null;
  let hls = null;

  const seek = () => {
    try {
      video.currentTime = start;
    } catch (e) {
      /* seeking before metadata is ready throws; the ready handler retries */
    }
    video.play().catch(() => {});
  };

  // Stop at the clip's out point rather than running on into the next scene.
  const onTime = () => {
    if (end !== null && video.currentTime >= end) video.pause();
  };
  video.addEventListener("timeupdate", onTime);

  if (window.Hls?.isSupported()) {
    hls = new window.Hls({ maxBufferLength: 30 });
    hls.loadSource(url);
    hls.attachMedia(video);
    hls.on(window.Hls.Events.MANIFEST_PARSED, seek);
  } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
    // Safari plays HLS natively, and hls.js reports unsupported there.
    video.src = url;
    video.addEventListener("loadedmetadata", seek, { once: true });
  } else {
    throw new Error("this browser cannot play HLS");
  }

  return () => {
    video.removeEventListener("timeupdate", onTime);
    video.pause();
    if (hls) hls.destroy();
    video.removeAttribute("src");
    video.load();
  };
}
