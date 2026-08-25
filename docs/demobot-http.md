# Demo Bot — read-only HTTP JSON API

The Telegram demo bot exposes the same twelve agents over plain HTTP so the
team can test them without Telegram. The API is a thin JSON layer over the
**exact same card builders** the bot dispatches to — same data sources, same
caches, same honesty rules. It changes nothing about how the bot behaves in
Telegram.

- **Binary**: built into `bin/demobot`; the server starts automatically
  alongside Telegram long polling.
- **Address**: `DEMOBOT_HTTP_ADDR` env var, default `127.0.0.1:8090`.
  A bind failure at startup is fatal (clear log line), so a running bot
  always implies a running API.
- **Shutdown**: same SIGTERM path as the poller — in-flight requests get a
  5-second drain.
- **Methods**: `GET` only. Anything else → `405` with `Allow: GET`.
- **CORS**: `Access-Control-Allow-Origin: *` on every response — this is a
  public read-only demo, nothing here is sensitive or mutable.
- **Rate limit**: global in-memory token bucket, **10 requests/second**
  across all clients. Over budget → `429` with `Retry-After: 1`.

## Endpoints

| Endpoint | Params | What it returns |
|---|---|---|
| `GET /` | — | Tiny index pointing at `/agents` |
| `GET /agents` | — | The agent list: name, description, supported assets, example URLs |
| `GET /agents/macro` | `?asset=btc\|gold` optional | Risk-on/off regime of big money; with `asset` the lamps re-framed for that asset (see [macro asset views](#macro-asset-views)) |
| `GET /agents/whale` | — | Large on-chain BTC transfers, net flow |
| `GET /agents/funding` | — | Perp funding pressure & liquidations |
| `GET /agents/momentum` | `?asset=` \| `?assets=` \| `?tf=` all optional | RSI/MACD; without params the multi-asset BTC/ETH/XAUUSD card (see [momentum scan](#momentum-scan-assets--tf)) |
| `GET /agents/trend` | `?asset=` optional (default `btc`) | Trend state machine (ADX + EMA50/EMA200), pullback zone in confirmed trends |
| `GET /agents/sr` | `?asset=` optional (default `btc`) | Support/resistance swing clusters with volume-weighted strength and held-of-tests frequency |
| `GET /agents/vol` | `?asset=` optional (default `btc`) | ATR(14) expansion/compression check |
| `GET /agents/fx` | — | Forex overview: EURUSD, GBPUSD, USDJPY, XAUUSD |
| `GET /agents/news` | — | Narrative radar (48h mention window) + AI idea |
| `GET /agents/risk` | `?balance=&risk=&entry=&stop=` all required | Position-size calculator |
| `GET /agents/digest` | — | All agents in one sweep, prioritized; AI brief in `ai_text`, one-liners in `sections` |
| `GET /agents/top` | — | The single strongest signal right now, with the AI brief + why-line |
| `GET /showcase` | — | **Landing catalog**: every agent with `live`/`degraded` status, headline and one-liner (see [landing showcase](#landing-showcase)) |
| `GET /showcase/example` | — | **Landing story**: one worked example — detected → explained → data → conclusion |

**Assets** for `momentum` / `trend` / `sr` / `vol`:
`btc` (default), `eth`, `eurusd`, `gbpusd`, `usdjpy`, `xauusd` — aliases
`xau`, `gold`, `bitcoin` are accepted. Passing `?asset=` to any other agent
(except `macro`, below) is a `400` (those reads are not asset-specific, and
pretending otherwise would be misleading).

## Momentum scan: assets + tf

`momentum` is user-configurable:

- `?assets=btc,eurusd,gold` — scan a comma list of registry assets, **up to
  6 entries**. Every entry must resolve against the registry (aliases fine);
  a bad entry is a `400` whose message names the entry and the allowed list.
  Duplicates collapse. Cannot be combined with `?asset=`.
- `?tf=1h|4h|1d` — the timeframe ALL indicator math runs on. Works with
  `?asset=`, `?assets=`, or alone (alone it re-bases the default BTC/ETH/XAUUSD
  trio). Anything else is a `400` naming the allowed set.
- Binance assets fetch the requested interval natively. Yahoo (FX/gold)
  serves `1h` and `1d` natively; **Yahoo has no native `4h`**, so 4h bars are
  aggregated from 1h — only complete 4-of-4 hourly groups aligned to UTC 4h
  boundaries become a bar, gaps/partial groups are dropped, never padded, and
  the card discloses the aggregation in a `Note:` fact.
- Telegram parity: `/momentum btc,eurusd 1d` (comma list + optional tf token).

**Per-asset machine status — `results`.** Every multi-asset momentum envelope
(the default card and every scan) carries a `results` array beside `facts`:

```json
"results": [
  {"asset": "BTC", "ok": true},
  {"asset": "ETH", "ok": false, "reason": "insufficient_history"},
  {"asset": "EURUSD", "ok": false, "reason": "source_offline"}
]
```

Top-level `ok` is `true` while at least ONE asset produced a real reading; a
mixed scan is machine-distinguishable from a full one by `results`, never by
parsing fact strings. With ZERO real readings the endpoint degrades as a
`503` whose reason is honest about the cause: `insufficient_history` when at
least one source answered but the series was too short, `source_offline` only
when every source was dead (the `503` body keeps its usual
`error`/`ok`/`reason` shape and does not carry `results`).

`?assets=` / `?tf=` on any other agent is a `400`. A repeated parameter
(`?assets=a&assets=b`, duplicated `?asset=`/`?tf=`) is a `400` "duplicate
parameter" — never a silent first-wins.

## ⚠️ Macro correlations changed meaning (B2)

**`/api/v1/macro` `correlations[].coef` and `window` kept their JSON names
but carry NEW semantics since package B**: the coefficient is Pearson over
the last **20-30 DAILY closes** (stooq daily history, date-aligned, refreshed
once a day), replacing the old ~3-hour intraday ring. The `window` string
self-describes the new basis (e.g. `"24 daily closes (20-30d window)"`, or
`"building daily window"` while it fills). Additive fields `ok` (coefficient
present) and `points` (overlapping daily closes) make the state machine-
readable. If a consumer displayed the old window text or interpreted the
coefficient as intraday co-movement, it must re-read this section — the JSON
schema is compatible, the meaning is not.

### Known methodology notes

- **Pearson runs on price LEVELS, not returns.** Two series sharing a trend
  will show a high level-correlation even when their day-to-day moves are
  unrelated — labels like "moving like stocks" are therefore stronger than
  the statistic strictly supports during trending months. A returns-based
  coefficient is planned as the next stage **after customer sign-off**; until
  then the window string states the basis and this note is the honest caveat.

## Macro data sources + `MACRO_SOURCE_ORDER`

**stooq stopped answering on 2026-08-24** — the quote endpoint serves an HTTP
404 HTML page and the daily-history endpoint serves a JavaScript anti-bot
challenge. Every lamp went `null`, the regime went `unknown`, and
`/agents/macro?asset=gold` honestly reported "no tradfin data". The backend now
tries each symbol against an **ordered list of providers** and takes the first
that yields a usable row.

| canonical id | lamp | stooq | Yahoo |
|---|---|---|---|
| `dx.f` | `dxy` | `dx.f` | `DX-Y.NYB` |
| `10yusy.b` | `rates` | `10yusy.b` | `^TNX` |
| `vi.f` | `vix` | `vi.f` | `^VIX` |
| `^spx` | `spx` | `^spx` | `^GSPC` |
| `xauusd` | `gold` | `xauusd` | `GC=F` |
| `btcusd` | — (correlations) | `btcusd` | `BTC-USD` |

The store, the JSON and every internal key stay on the **stooq ids** whichever
provider answered — the Yahoo tickers exist only inside `internal/macro`.

### Configuration

`MACRO_SOURCE_ORDER` (backend env, default `stooq,yahoo`) pins the order:

```bash
MACRO_SOURCE_ORDER=stooq,yahoo   # default: stooq first (it may recover), Yahoo per-symbol fallback
MACRO_SOURCE_ORDER=yahoo         # pin Yahoo, never touch stooq
MACRO_SOURCE_ORDER=yahoo,stooq   # prefer Yahoo, keep stooq as the fallback
```

Comma separated, case- and whitespace-insensitive, duplicates collapse. Unknown
names are logged and dropped; a value naming **no** known provider falls back to
the default rather than leaving the worker with zero sources. The resolved order
is logged once per process: `macro: source order = stooq → yahoo`.

### New field: `source`

`/api/v1/macro` gained an **additive** `"source"` string on each lamp and each
correlation. It is `"stooq"` or `"yahoo"`, and:

- it is present **exactly when a value is** — a lamp with `value: null` always
  has `source: ""`, so the field can never be read as "we had data from X";
- lamps within one response may name **different** providers (the fallback is
  per symbol, not per cycle);
- a correlation is built from two daily histories, so its `source` is
  `"mixed"` when the BTC leg and the paired leg came from different providers —
  never one of the two picked silently.

### Two honesty guards specific to the Yahoo path

**Freshness.** stooq announces a dead symbol explicitly (an `N/D` row). Yahoo's
chart endpoint has no such sentinel — it just returns the last bar it holds — so
a frozen feed would keep serving a weeks-old close as `ok:true`, with a status
and a vote in the composite. Any quote older than **7 days** is therefore
rejected for every source alike. Seven days cannot fire on a normal closure (a
weekend is 3 days from Friday's bar, the longest US holiday break about 5). A
rejected quote still contributes its **date**, so the lamp keeps showing how
stale the market went rather than blanking.

**Daylight saving.** Yahoo shifts its daily-bar timestamps with DST but reports
`meta.gmtoffset` only for the offset *at request time*. Measured over a 1-year
DX-Y.NYB window: 166 bars at 04:00Z (EDT) and 83 at 05:00Z (EST) under a single
`gmtoffset` of −14400. Dating bars with that one scalar mis-dates the other
regime — 166 of 251 bars wrong for a request made during EST — which, since only
the last 30 rows are kept, would have slid the whole correlation window one day
off BTC's UTC days for weeks after each November transition. Session dates are
therefore derived by converting **each bar's own instant** into
`meta.exchangeTimezoneName`, with a raw-UTC fallback that is measured to agree
on 251/251 bars for all six symbols in both regimes.

### Correlation window sizing (changed 2026-08-24)

The per-symbol daily history now keeps **42 rows, not 30**. The cap is applied
per symbol in *rows*, but the correlation joins on calendar *dates*, and the two
legs accumulate rows at different rates — BTC trades 7 days a week, the tradfin
symbols 5. At 30 rows, BTC's window spanned 30 calendar days while SPX's spanned
~42, so only the ~20 SPX sessions inside BTC's window could pair: measured
`points = 20` against a minimum of 20, i.e. **zero margin**. One market holiday
(Labor Day, 2026-09-07) would have dropped it to 19 and blanked `btc_spx` for
about a month. At 42 the measured overlap is 29-30 points — still inside the
documented "20-30 daily closes" window, now with ~10 points of headroom. The
stooq fetch window widened from 60 to 90 calendar days to be able to fill it.

The once-a-day refresh stamp now advances **only when the BTC leg stored**.
Every correlation is BTC↔X, and BTC is fetched last, so a cycle that stored the
tradfin legs but missed BTC used to stamp success and park all three
correlations on an empty leg for 24 hours. It now retries on the next 3-min tick.

### Poll budgets

The quote phase gets **45s**, with an explicit **10s cap per provider attempt**.
With 6 symbols and the default 2-provider order: a healthy cycle is ~2.4s, a
cycle with stooq dead ~4.2s (measured 2.9-5.0s live). The per-attempt cap
matters when a provider *hangs* rather than fails — without it one symbol could
consume the whole phase and every later symbol would be cancelled before being
tried at all. Symbols the budget does not reach store an honest not-ok quote,
exactly like any other failure.

### VIX instrument note

The VIX lamp now reads **spot `^VIX`** (Yahoo), where stooq served `vi.f`, the
front future — which runs roughly 0.5-2 points higher in contango. The 18/25
thresholds are the canonical *spot* levels, so this is a closer fit than before;
they were **not** re-tuned, and must not be re-tuned against the old `vi.f`
series.

### What did NOT change

Honesty behaviour is identical. When **both** providers come up empty for a
symbol the lamp is still `ok:false` / `value:null`, the regime is still
`unknown` when zero lamps carry a value, `generated_idea` is still `""`, and the
`as_of` carry-forward of a last-known date is unchanged. The fallback only adds
attempts; it never softens a failure.

## Macro asset views

`GET /agents/macro?asset=gold` (or `btc`) re-frames the same five lamps for
one asset — strictly these two view keys, anything else is a `400` listing
them. No parameter keeps the global regime card, which itself now carries a
two-line signal map (`BTC view: …` / `Gold view: …`) whenever at least one
real lamp exists.

- **BTC view** — the crypto-centric regime labeled as an asset view:
  risk-on → tailwind, risk-off → headwind, mixed → mixed.
- **Gold view** — the documented mapping: dollar/yields **up = pressure** on
  gold; **fear (VIX > 25) / equities down = support** (safe-haven bid; the
  SPX leg is "mild" via the lowest weight). Weighted vote DXY 40 / 10Y 25 /
  VIX 25 / SPX 10, support > 65, pressure < 35; fewer than 3 live voting
  lamps → honest `ok:false` / `no_data`. The gold lamp itself reports its
  session move but never votes on its own outlook.
- Unknown-regime honesty carries over per asset: zero real lamps →
  `UNKNOWN` verdict with `market_closed` / `no_data` reason, never a view.

`risk` accepts the same tolerant number formats as the Telegram command:
`balance=10,000`, `risk=1%`, `entry=$64000` all parse.

## Landing showcase

Two endpoints for the marketing page. They exist because the landing was
reading `/agents` as a menu of *ideas* — it surfaced one agent as a real-data
demo and labelled the rest "planned", while twelve agents were serving live
data the whole time.

Both run over **one** sweep — the same `gather()` `/digest` uses, through the
same card builders — memoized for **60 seconds** with singleflight. Ten
landing renders cost one sweep; `/showcase/example` rides the build
`/showcase` just made. A page render can never fire twelve uncached upstream
calls. Upstream cost over a plain `/digest`: exactly one extra GET (the
narrative radar).

### `GET /showcase` — the catalog

```json
{
  "generated_at": "2026-08-25T12:31:04Z",
  "live_count": 11,
  "total_count": 12,
  "agents": [
    {
      "slug": "digest",
      "name": "AlphaVizor Digest",
      "category": "tools",
      "status": "live",
      "ok": true,
      "reason": null,
      "headline": "Top signal: Trend Agent — Confirmed UPTREND",
      "one_liner": "🟢 Trend BTC: confirmed uptrend",
      "data_as_of": "2026-08-25T12:31:04Z",
      "example_url": "/agents/digest"
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `generated_at` | When the **sweep** ran, not when the request arrived — with the 60s memo a render can legitimately serve a payload up to a minute old, and saying so is the honest form of a cache |
| `live_count` / `total_count` | Agents that returned `ok: true` on this sweep, out of every agent that exists |
| `status` | `live` \| `degraded` — **never `planned`**. This endpoint only lists agents whose builder actually ran; there is no fictional state and no roadmap entry here |
| `ok` / `reason` | The same machine-readable pair the agent envelopes carry (`source_offline`, `insufficient_history`, `below_threshold`, `no_data`, `market_closed`), `null` when `ok` |
| `headline` | The card's verdict line — for `digest`, the prioritized "Top signal: …" line |
| `one_liner` | The digest-style one-liner, plain text |
| `category` | `crypto` \| `forex` \| `macro` \| `onchain` \| `derivatives` \| `news` \| `tools`. `tools` holds the three that are not a single-market read: `digest`, `top`, `risk` |
| `example_url` | Where the landing links for the full card: `/agents/<slug>` |

**Degraded agents stay in the list.** They are not hidden and they are not
relabelled — the row keeps its `reason` so the landing can decide to show
"source offline" or drop the card, and the honesty rule from the card layer
carries all the way to the marketing page. The rows come back in the same
order as `/agents`.

### `GET /showcase/example` — one worked story

The "what does a user actually get" demonstration, in the order a trader
reads it: **detected → explained → data → conclusion**.

```json
{
  "generated_at": "2026-08-25T12:31:04Z",
  "agent": "Trend Agent",
  "slug": "trend",
  "asset": "BTC",
  "detected": "Trend Agent on BTC — Confirmed UPTREND.",
  "explained": "ADX(14) at 31.2 sits above the 25 confirmation threshold with EMA50 over EMA200.",
  "data": [
    "ADX(14): 31.2 (trend confirms above 25) · RSI(14): 62.0",
    "EMA50 118420 above EMA200 112870",
    "Invalidation: close below 110350"
  ],
  "conclusion": "For a trader this is a bullish reading on BTC: the numbers above lean up, and the read holds only for as long as they do. The structure this read describes breaks on a close below 110350.",
  "levels": { "invalidation": 110350.2, "invalidation_side": "below" },
  "example_url": "/agents/trend",
  "disclaimer": "Analytics, not financial advice",
  "data_as_of": "2026-08-25T12:00:00Z"
}
```

- **Which agent tells the story**: the same deterministic priority rule
  `/top` uses (`topSelection` — RISK-OFF macro first, otherwise the strongest
  deviation from neutral among funding/momentum/trend, ties breaking
  funding > momentum > trend). If that winner is **degraded**, the story
  falls back to the strongest `ok` agent instead of narrating a dead source;
  `digest`, `top` and `risk` are never the subject (an aggregate is not one
  agent's story, and the calculator has no "detected" moment).
- **`explained`** is the AI why-line when one is available, taken from the
  **same 5-minute memo `/top` uses** — the landing costs no extra LLM spend
  and opens no new prompt kind. On the fallback path there is deliberately no
  AI call; the card's strongest fact stands in, and so it does whenever AI is
  disabled or the call failed.
- **`data`** is 3-4 of the card's own live fact lines. Fewer only when the
  card itself carries fewer — nothing is invented to reach a rounder number.
- **`conclusion`** is analytical language only: what the reading means while
  its inputs hold. Never BUY/SELL, never an instruction to enter or exit —
  the same sanitizer rules that apply to `ai_text` apply here.
- **`levels`** rides along when the winning card has one (trend invalidation,
  S/R clusters, vol expansion ratio) — the raw-precision object documented
  under [machine-readable levels](#machine-readable-levels).
- **`503`** when *every* agent is degraded: the standard
  `{"error": …, "ok": false, "reason": …}` body. A story is the one thing
  this API will not fake.

### Verdicts are authoritative for the AI layer

The agent verdicts come from state machines, and the AI text is decoration on
top of them — so the model is never allowed to overrule one. A trend in its
**grey** state ("trend forming, not confirmed") must not be narrated as
confirmed structure just because ADX alone looks convincing. Three layers
enforce this on the shared AI path, so `/agents/digest`, `/agents/top` and
`/showcase/example` are all covered:

1. the payload labels each read `authoritative_verdict` and carries `state`
   plus `confirmation_withheld`;
2. the system prompt forbids asserting a confirmation the verdict withheld,
   and asks for *which condition failed* instead;
3. a post-processing filter drops any sentence that claims confirmation about
   an agent whose state withheld it. If that empties the text, the AI block is
   omitted — an absent decoration beats a contradiction.

## Response envelope

Every agent endpoint answers with one shape:

```json
{
  "agent": "Trend Agent",
  "asset": "EURUSD",
  "ok": true,
  "reason": null,
  "verdict": "Grey zone — trend forming, not confirmed",
  "semaphore": "neutral",
  "facts": ["EMA50 1.1583 vs EMA200 1.1560 — bullish structure", "..."],
  "levels": {"invalidation": 1.148236},
  "confidence": null,
  "ai_text": null,
  "data_as_of": "2026-08-18T09:00:00Z",
  "disclaimer": "Analytics, not financial advice",
  "card_html": "⚪ <b>Trend Agent</b> · EURUSD\n<b>Grey zone — ...</b>\n• ...\n\n<i>Analytics, not financial advice · AlphaVizor · 2026-08-18 09:00 UTC</i>"
}
```

| Field | Type | Meaning |
|---|---|---|
| `agent` | string | Full agent name (`"AlphaVizor Digest"` for the digest) |
| `asset` | string | Asset label; `""` when the read is not asset-specific |
| `ok` | bool | `true` when the agent produced a **real reading**; `false` for every degraded state — see [ok / reason](#machine-readable-status-ok--reason) |
| `reason` | string \| null | `null` when `ok`; otherwise the machine-readable WHY (enum below) — branch on this, never on verdict wording |
| `verdict` | string | The card's headline verdict |
| `semaphore` | string | `bullish` \| `bearish` \| `neutral` — the card's traffic light |
| `facts` | string[] | The card's bullet facts, `[]` when none |
| `levels` | object | **trend / sr / vol only**: raw-precision numeric levels — see [levels](#machine-readable-levels). Absent for other agents and on `ok: false` cards |
| `results` | array | **momentum multi-asset cards only**: per-asset machine outcomes `{"asset","ok","reason"}` — see [momentum scan](#momentum-scan-assets--tf). Absent elsewhere |
| `confidence` | int \| null | 0–100 when the source supplied one, otherwise `null` — never invented |
| `ai_text` | string \| null | Plain-text AI block (mood read / idea / brief / why-line); `null` when AI is disabled or the call failed |
| `sections` | string[] | **digest only**: plain-text one-liners of every other agent (the winner heads the envelope) |
| `data_as_of` | string | RFC3339 UTC; for candle-based agents this is the **close time of the last closed bar used** — the same stamp as the card footer |
| `disclaimer` | string | Always `"Analytics, not financial advice"` |
| `card_html` | string | The exact Telegram HTML message the bot would send (for `digest`/`top`: the full composed message) |

## Machine-readable status: ok / reason

Templates must branch on **why** a value is absent, not on verdict strings.
Every envelope carries the pair; when `ok` is `false`, `reason` is one of:

| `reason` | Served by | Meaning |
|---|---|---|
| `market_closed` | macro | Regime `UNKNOWN` because the tradfin market is closed — no lamps to read |
| `source_offline` | any agent | The source behind the headline reading is unreachable. Usually a `503`; also a `200` on the funding card when rates are dead but the liquidation feed is alive (liq facts still render) |
| `insufficient_history` | momentum, trend, sr, vol | Source alive, but too few **closed** bars for the indicator set (always a `503`) |
| `below_threshold` | news | Narrative radar warming up: the top theme is under 5 mentions/24h (`200`, themes listed without scores), or there are no snapshots yet (`503`) |
| `no_data` | macro, whale | Upstream alive but nothing to read: macro unknown **inside** the open tradfin window; whale feed with no BTC snapshot yet |

For `digest` / `top` the pair (and `levels`) describes the **top signal
card** heading the envelope — with every source dead, the honest macro
fallback yields `ok: false`, `"reason": "source_offline"` inside the 200.

The `503` error body carries the same pair beside the message, so single-agent
degraded states are branchable too:

```json
{"error": "Trend Agent: Insufficient history — no verdict", "ok": false, "reason": "insufficient_history"}
```

## Machine-readable levels

Three agents' readings *are* price levels; their envelopes add a `levels`
object with the raw computed numbers — full float precision, never the
display-rounded strings shown in `facts`:

| Agent | `levels` shape |
|---|---|
| `trend` | `{"invalidation": 63297.4, "invalidation_side": "below", "pullback_zone": {"from": 64850.1, "to": 64210.7}}` — invalidation is **direction-aware**: uptrend and non-directional states break BELOW the EMA cluster (min(EMA50,EMA200) − 1 ATR, `invalidation_side: "below"`); a confirmed DOWNTREND breaks ABOVE it (max + 1 ATR, `"above"`). Omitted only on a degenerate flat series where no ATR exists. `pullback_zone` is the EMA20-EMA50 band and appears **only in confirmed-trend states** (`from` = EMA20, `to` = EMA50 — so `from` > `to` in an uptrend); flat/grey/conflict omit it |
| `sr` | `{"supports": [{"level": 63775.42, "touches": 9, "strength": 11.5, "weakening": false, "breaks": 2, "holds": 6, "last_touch": "2026-08-12T08:00:00Z"}, …], "resistances": […]}` — strength-sorted raw cluster means; an empty side is `[]`, never `null`. `strength` = touches + 0.5 per touch on above-median volume (median over NON-ZERO volumes; on a level whose touches are mostly volume-less the volume features disable and `strength` equals `touches`). `weakening` = ≥7 touches with the last 3 touches' mean volume below the first 3's (same volume gate). `breaks`/`holds` are **frequency counts** of level tests over the 249-bar window: a test = a close entering the ±0.25×ATR band (ATR frozen at the entry bar); within 3 bars a close beyond the level on the far side = **break**, a close back beyond the band on the approach side = **hold** (a hold IS the rejection); price stalling inside the band for 3 bars is **unresolved and dropped** — never counted as a hold. Frequencies, never probabilities. `last_touch` = RFC3339 UTC of the newest touch — the later of the last swing in the cluster and the last close-test of the level |
| `vol` | `{"expansion_ratio": 1.01}` — ATR(14) over its 30-bar average, unrounded |

`sr` with **no levels at all** is never a "Key levels around …" reading: a
window with ZERO swing points (monotone/flat tape) degrades to a `503`
`insufficient_history`; swings present but no cluster clear of the last price
serves an explicit ok `"No significant levels detected in the window"` with
empty arrays.

`trend` confirmation now **requires swing-structure agreement**: an EMA/ADX
uptrend with LH/LL (or mixed) pivots demotes to the grey zone with the fact
`Structure disagrees (LH/LL) — trend not confirmed` — the pullback zone
disappears with the demotion. HH/HL itself is read from the last six
ALTERNATING pivots in time order; a non-alternating pivot tail reads as
`mixed`.

`levels` is absent for every other agent and on degraded (`ok: false`)
envelopes.

## Errors

Errors are always `{"error": "..."}` with an honest message:

| Status | When |
|---|---|
| `400` | Bad arguments: unknown asset, `?asset=` on a non-asset agent, a bad entry / >6 entries in `?assets=`, an unknown `?tf=`, `?assets=`/`?tf=` outside `momentum`, a **repeated parameter** (`?assets=a&assets=b` → "duplicate parameter"), an unknown macro view (`?asset=` on `macro` accepts only `btc`/`gold`), missing/non-numeric/invalid risk params |
| `404` | Unknown agent name or path |
| `405` | Non-GET method (`Allow: GET` header set) |
| `429` | Over the 10 req/s global budget (`Retry-After: 1` header set) |
| `503` | Upstream degraded: the agent's data source is offline, history is too short for the indicator set, or the narrative radar has no snapshots yet — the same states the Telegram card reports in words. The body adds `"ok": false` and the machine-readable `"reason"` (`source_offline` \| `insufficient_history` \| `below_threshold`) |

`digest` and `top` never `503`: they aggregate whatever is alive and label
dead parts honestly inside the payload (an offline agent shows as an
`"⚪ … offline"` one-liner), exactly like the Telegram digest.

## curl examples

```bash
# The agent list
curl -s localhost:8090/agents | jq

# Trend state machine for EURUSD
curl -s 'localhost:8090/agents/trend?asset=eurusd' | jq

# Momentum, default multi-asset card (BTC/ETH/XAUUSD)
curl -s localhost:8090/agents/momentum | jq

# User-configured momentum scan on daily candles
curl -s 'localhost:8090/agents/momentum?assets=btc,eurusd,gold&tf=1d' | jq

# Macro lamps re-framed for gold
curl -s 'localhost:8090/agents/macro?asset=gold' | jq

# Position-size calculator
curl -s 'localhost:8090/agents/risk?balance=10000&risk=1&entry=64000&stop=62500' | jq

# Full prioritized digest (AI brief in ai_text, one-liners in sections)
curl -s localhost:8090/agents/digest | jq

# Machine status only: did the agent produce a real reading, and if not, why?
curl -s localhost:8090/agents/macro | jq '{ok, reason}'

# Raw numeric levels (trend invalidation / sr clusters / vol ratio)
curl -s localhost:8090/agents/sr | jq '.levels'

# Error shapes
curl -si 'localhost:8090/agents/trend?asset=doge'   # 400 unknown asset
curl -si  localhost:8090/agents/nope                # 404 unknown agent

# ── Landing showcase ─────────────────────────────────────────────────────────

# The catalog the landing renders
curl -s localhost:8090/showcase | jq

# How many agents are serving live data right now
curl -s localhost:8090/showcase | jq '{generated_at, live_count, total_count}'

# Just the catalog rows, compact
curl -s localhost:8090/showcase | jq -r '.agents[] | "\(.status)\t\(.slug)\t\(.headline)"'

# Only the degraded ones, with the reason they are degraded
curl -s localhost:8090/showcase | jq '.agents[] | select(.ok == false) | {slug, reason, headline}'

# Proof the word never appears
curl -s localhost:8090/showcase | grep -c planned          # 0

# The worked example: detected → explained → data → conclusion
curl -s localhost:8090/showcase/example | jq

# The story as prose
curl -s localhost:8090/showcase/example | jq -r '.detected, .explained, (.data[]), .conclusion'
```

## Honest notes

- **Data sources**: backend REST (`/api/v1/macro`, `whale-flow`,
  `funding/liquidations`, `narratives`, `market/*`) · Binance spot klines
  and futures premiumIndex (public, no key) · Yahoo Finance chart API for
  FX/gold (`GC=F` is the working gold source; `XAUUSD=X` is dead upstream).
  The macro lamps behind `/agents/macro` are served by stooq **with a Yahoo
  fallback per symbol** since 2026-08-24 — see [macro data
  sources](#macro-data-sources--macro_source_order). Each lamp reports which
  provider produced it.
- **No fake numbers, ever.** A dead source is a `503` (single agents) or an
  explicit `offline` line (digest/top). Too little history for an indicator
  says so instead of rendering a confident flat. Weekend FX data carries the
  `⏸ Forex market closed` banner fact and is computed on Friday's close.
- **Closed bars only**: indicator math never uses the still-forming candle,
  so `data_as_of` can legitimately lag wall clock by up to one interval.
- **AI is decoration, not a dependency**: `ai_text` is `null` whenever
  `ANTHROPIC_API_KEY` is absent or the LLM call fails — the data fields are
  never blocked by it. The AI brief is memoized for **5 minutes per unique
  market state** in one cache shared with the Telegram path, so hammering
  the HTTP digest does not multiply LLM spend.
- **One sweep per minute for the landing**: `/showcase` and
  `/showcase/example` share a 60-second singleflight memo over the whole
  sweep, so ten concurrent renders make one pass over the sources. A sweep
  that found nothing alive is held only 15s, so a blip cannot freeze the
  landing on a blackout.
- **Shared caches**: candles are cached 60s per symbol across both
  transports; ten HTTP digests and a Telegram `/digest` inside a minute hit
  Binance once per symbol, not eleven times.
- **Not an execution feed**: same disclaimer as every card — analytics, not
  financial advice.
