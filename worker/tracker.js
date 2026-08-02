/**
 * Newspeak click-tracking Worker.
 *
 * Two routes:
 *   GET /r?u=<dest>&s=<source domain>&sig=<hmac>
 *     Verifies the HMAC (so only links Newspeak minted are honored — this is NOT an open
 *     redirect), increments the per-source click counter in KV, then 302-redirects the
 *     reader to the real article.
 *   GET /stats?token=<STATS_TOKEN>
 *     Returns the cumulative per-source click counts as JSON, e.g. {"techcrunch.com": 30}.
 *     The Newspeak cron pulls this each run to update source reputation.
 *
 * Bindings / secrets (see wrangler.toml + `wrangler secret put`):
 *   env.CLICKS         — KV namespace (keys "clk:<domain>" → integer string)
 *   env.SIGNING_SECRET — must equal the pipeline's TRACKING_SECRET
 *   env.STATS_TOKEN    — must equal the pipeline's TRACKING_STATS_TOKEN
 *
 * The `sign()` here MUST stay byte-for-byte compatible with newspeak/tracking.py:sign()
 * (HMAC-SHA256, hex, first 16 chars). Keep SIGNING_SECRET stable so links in already-sent
 * emails keep resolving.
 */

const KV_PREFIX = "clk:";

// User-Agents of link pre-fetchers / scanners: still redirect, but don't count the click.
const PREFETCH_UA = [/GoogleImageProxy/i, /YahooMailProxy/i, /bingpreview/i, /Slackbot/i, /facebookexternalhit/i, /Twitterbot/i];

async function sign(payload, secret) {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey("raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const mac = await crypto.subtle.sign("HMAC", key, enc.encode(payload));
  const hex = [...new Uint8Array(mac)].map((b) => b.toString(16).padStart(2, "0")).join("");
  return hex.slice(0, 16);
}

// Length-independent constant-time-ish comparison to avoid leaking via timing.
function safeEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let out = 0;
  for (let i = 0; i < a.length; i++) out |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return out === 0;
}

async function bump(env, domain) {
  const key = KV_PREFIX + domain;
  // Eventually-consistent read-modify-write. Fine for a low-volume, deliberately noisy
  // soft signal; concurrent clicks may occasionally under-count.
  const cur = parseInt((await env.CLICKS.get(key)) || "0", 10) || 0;
  await env.CLICKS.put(key, String(cur + 1));
}

async function handleRedirect(request, env, ctx, url) {
  const dest = url.searchParams.get("u");
  const source = url.searchParams.get("s");
  const sig = url.searchParams.get("sig");
  if (!dest || !source || !sig) return new Response("Bad request", { status: 400 });

  const expected = await sign(`${dest}|${source}`, env.SIGNING_SECRET || "");
  if (!safeEqual(sig, expected)) return new Response("Invalid signature", { status: 400 });

  let target;
  try {
    target = new URL(dest);
    if (target.protocol !== "http:" && target.protocol !== "https:") throw new Error("bad scheme");
  } catch {
    return new Response("Bad destination", { status: 400 });
  }

  const ua = request.headers.get("user-agent") || "";
  const isPrefetch = PREFETCH_UA.some((re) => re.test(ua));
  if (request.method === "GET" && !isPrefetch) {
    ctx.waitUntil(bump(env, source)); // count in the background; don't delay the reader
  }
  return Response.redirect(target.toString(), 302);
}

async function handleStats(env, url) {
  const token = url.searchParams.get("token") || "";
  if (!env.STATS_TOKEN || !safeEqual(token, env.STATS_TOKEN)) {
    return new Response("Not found", { status: 404 });
  }
  const out = {};
  let cursor;
  do {
    const list = await env.CLICKS.list({ prefix: KV_PREFIX, cursor });
    for (const k of list.keys) {
      const v = await env.CLICKS.get(k.name);
      out[k.name.slice(KV_PREFIX.length)] = parseInt(v || "0", 10) || 0;
    }
    cursor = list.list_complete ? undefined : list.cursor;
  } while (cursor);
  return Response.json(out);
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    if (url.pathname === "/r") return handleRedirect(request, env, ctx, url);
    if (url.pathname === "/stats") return handleStats(env, url);
    return new Response("Not found", { status: 404 });
  },
};
