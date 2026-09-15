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
| `GET /agents/macro` | `?asset=btc\|gold` optional | Risk-on/off regime read from 5 tradfin lamps; with `asset` the lamps re-framed for that asset (see [macro asset views](#macro-asset-views)) |
| `GET /agents/whale` | — | Large on-chain BTC transfers, net flow |
| `GET /agents/funding` | — | Perp funding pressure & liquidations |
| `GET /agents/momentum` | `?asset=` \| `?assets=` \| `?tf=` all optional | RSI/MACD; without params the multi-asset BTC/ETH/XAUUSD card (see [momentum scan](#momentum-scan-assets--tf)) |
| `GET /agents/trend` | `?asset=` optional (default `btc`) | Trend state machine (ADX + EMA50/EMA200), pullback zone in confirmed trends |
| `GET /agents/trend/chart` | `?asset=` optional (default `btc`) | Chart data for the Trend Agent: the candles it reads, EMA20/50/200, pivots, zone and invalidation (see [trend chart](#trend-chart)) |
| `GET /agents/sr` | `?asset=` optional (default `btc`) | Support/resistance swing clusters: class, reactions/breaks counts, last touch, content `blocks` |
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

## Momentum card and content blocks

> ⚠️ **Momentum `verdict` format changed 2026-09-15 — do not parse it.**
> Before: the multi-asset verdict listed each asset (`BTC: NEUTRAL · ETH:
> BEARISH · …`) and the single-asset verdict was `BTC: NEUTRAL`. Now the
> multi-asset verdict is a counter (`0 bullish · 1 bearish (ETH) · 2 not
> confirmed`) and the single-asset verdict is `NOT CONFIRMED · 4h — <reason>`,
> without the asset name. The per-asset direction is machine-readable:
> `results[].verdict` (`bullish | bearish | neutral`) and `results[].state`;
> the card colour is `semaphore`. `verdict` and `facts` are display text and
> may change again.

What changed 2026-09-15: the card reads in one pass. The rule is **unchanged**
(bullish = RSI(14) ≥ 55 and MACD(12,26,9) histogram > 0; bearish = RSI ≤ 45
and histogram < 0; else neutral), and so are `semaphore` values, the digest
inputs and caching. New: a reason for every neutral, a two-condition
checklist, a counter header for multi-asset cards, per-asset timeframe, bar
time and freshness, context lines marked as outside the reading, content
blocks, and the read itself in `results[]`. Every line is at most 110
characters.

**Neutral with a reason.** One reason per asset, computed from the same two
numbers as the verdict:

| `state` (results[]) | Reason on the card | When |
|---|---|---|
| `confirmed_bullish` / `confirmed_bearish` | `RSI and MACD agree` | the rule's bullish / bearish |
| `conflict` | `conflict: RSI up, MACD down` / `conflict: RSI down, MACD up` | RSI ≥ 55 with histogram < 0, or RSI ≤ 45 with histogram > 0 |
| `rsi_below_55` | `RSI below the 55 threshold` | 45 < RSI < 55, histogram > 0: only RSI is missing for bullish |
| `rsi_above_45` | `RSI above the 45 threshold` | 45 < RSI < 55, histogram < 0: only RSI is missing for bearish |
| `macd_at_zero` | `RSI up, MACD histogram at 0` / `RSI down, …` | RSI past a threshold, histogram exactly 0 |
| `neutral_zone` | `RSI in the neutral zone 45–55` | 45 < RSI < 55 and histogram exactly 0: both directions miss both conditions |

With a non-zero histogram, an RSI inside 45–55 always leaves one direction
short of RSI alone, so the card names that threshold; `neutral_zone` needs a
histogram of exactly 0.

**Checklist.** The two conditions, evaluated toward the side RSI is past (or,
inside 45–55, the side of the histogram): `RSI 54.7 < 55 ✗ · MACD histogram
above 0 ✓`. All ✓ toward a direction is exactly that direction's verdict. The
card shows the histogram's **sign only**: its raw value is in price units
(BTC +217, EURUSD +0.00013) and cannot be compared across assets; the raw
value is `results[].macd_histogram`. RSI is printed to one decimal **rounded
toward 50**, so a printed value never sits on the other side of 55 or 45 from
the real one (54.96 prints 54.9, 45.04 prints 45.1).

**Single-asset card** (`?asset=`):

```
⚪ Momentum Agent · BTC
NOT CONFIRMED · 4h — RSI below the 55 threshold
• Why: RSI 54.7 < 55 ✗ · MACD histogram above 0 ✓
• Turns bullish when RSI ≥ 55 and the MACD histogram is above 0 (now ✗: RSI)
• Turns bearish when RSI ≤ 45 and the MACD histogram is below 0 (now ✗: RSI, MACD)
• Read on closed 4h candles: RSI(14) and the MACD(12,26,9) histogram
• Context, not part of the reading: volume 0.96× its 20-bar average (4h)
```

A confirmed read says what keeps it: `Stays bullish while RSI ≥ 55 and the
MACD histogram is above 0; any ✗ turns it neutral`. The verdict line carries
the timeframe. `semaphore` stays `bullish | bearish | neutral`; the card text
says `not confirmed` for neutral.

**Multi-asset card** (default trio and every `?assets=` / `?tf=` scan):

```
⚪ Momentum Agent · BTC/ETH/XAUUSD
0 bullish · 0 bearish · 3 not confirmed
• BTC · 4h: not confirmed — RSI below the 55 threshold
• BTC: RSI 54.7 < 55 ✗ · MACD histogram above 0 ✓ · last bar Sep 15 08:00 UTC
• ETH · 4h: not confirmed — conflict: RSI up, MACD down
• ETH: RSI 61.3 ≥ 55 ✓ · MACD histogram below 0 ✗ · last bar Sep 15 08:00 UTC
• GOLD · COMEX GC=F · 1h: not confirmed — conflict: RSI down, MACD up
• GOLD · COMEX GC=F: RSI 37.9 ≤ 45 ✓ · MACD histogram above 0 ✗ · last bar Sep 15 09:00 UTC
• Bullish needs RSI ≥ 55 and MACD histogram above 0; bearish needs RSI ≤ 45 and below 0
• Context, not part of the reading: BTC 4h volume 0.96× its 20-bar average
• Context, not part of the reading: ETH return minus BTC return incl. today, 7d +1.4 pp · 30d +9.6 pp
```

- **Header = counter**: `N bullish (names) · N bearish (names) · N not
  confirmed`, then `· N unavailable` when an asset produced no reading, then
  `· <tf>` whenever every read asset shares one timeframe — any `?tf=` scan,
  a native scan like `?assets=btc,eth` (4h) or `?assets=eurusd,gbpusd` (1h),
  the trio with gold missing. Gold is named `GOLD` inside the counter. This
  is also the `verdict` and the digest one-liner. The counter covers every
  asset, gold and FX included.
- **Semaphore of a multi-asset card** — the counter's rule, never the first
  asset, applied to the **BTC/ETH reads only** (the reads the digest ranks):
  `bullish` when at least one of them reads bullish and none bearish;
  `bearish` mirrored; `neutral` when none is confirmed or they point both
  ways. A gold or FX reading is counted but never colours the card, so a
  colour can never sit beside the digest's "No confirmed reading" line;
  when a card mixes both kinds it says so: `Colour follows BTC/ETH only, the
  reads the digest ranks; other assets are only counted`. When BTC/ETH were
  requested but produced no reading (offline or short history — e.g. the
  default trio with only gold live), the card stays `neutral` and says
  `Colour follows BTC/ETH only, the reads the digest ranks; no BTC/ETH read is
  available`; such a card does not compete in the digest (`no_ranked_read`,
  below). Only a request with no BTC/ETH asset at all (an FX/gold scan such
  as `?assets=eurusd,gbpusd`; the digest never sweeps one) colours by all its
  reads — decided by the requested assets, not by which ones read. The digest inputs
  (`confirmed`, the score |RSI−50|×2 of a confirmed read, the freshness bar)
  come from BTC/ETH only, exactly as before.
- Every asset: a state line (timeframe, verdict word, reason) and a checklist
  line with the close of its last closed bar and its freshness. A failed asset
  gets one line (`data unavailable right now` / `insufficient history for
  RSI/MACD`) and is counted as unavailable.
- A `?tf=` scan adds `All assets read on closed <tf> candles`; a Yahoo 4h
  scan keeps the aggregation `Note:`.

**Context, not part of the reading.** BTC volume (last closed bar against
its 20-bar average; no bands are claimed) and ETH vs BTC. The ETH line is a
**return gap** in percentage points — ETH's return minus BTC's return — not
ETH's own move. Each return runs from the daily close 7 (30) UTC days before
today to the current price: 6 (29) full days plus the still-forming UTC day
("incl. today"; `internal/market/momentum.go`, `pctChangeOverDays`), so it
moves during the day and is a different horizon from the closed-bar
RSI/MACD. Neither feeds the verdict.

**Freshness per asset** (`results[].freshness`, and on the checklist line):

| Value | Card text | Rule |
|---|---|---|
| `on_time` | — | no missing bar the rule can detect. **Always** for Binance (BTC, ETH): the kline cache never serves an answer older than 60 s and a klines answer always ends with the bar still forming (never read), so a read's last closed bar is at most one bar old by the source's contract |
| `market_closed` | `market closed` | a Yahoo asset (FX or gold) while `isForexOpen` says closed: the fixed weekend window Friday 21:00 → Sunday 21:00 UTC. The only "closed" the service knows |
| `data_delayed` | `data delayed` | Yahoo only: the last closed bar is older than two bars (the digest's own stale bound) while the market has been open for those two bars (right after the Sunday reopen a missing bar is not yet due) |

Limits, stated rather than guessed: FX and COMEX gold share the one weekend
window. COMEX has its own hours (a daily maintenance break, a different
weekend edge) and neither calendar knows holidays or DST shifts (±1 h at the
window edges); the service has no exchange calendar, so it does not invent
one. Consequences: a holiday or a COMEX break inside the week can read `data
delayed` (a bar was expected and did not come), never `market closed`; the
single-asset FX/gold card keeps the shared `⏸ Forex market closed (weekend)`
banner, gold included.

**Content blocks** (same `ContentBlocks` object as trend / S/R): top-level
`blocks` on the single-asset card, `results[].blocks` per asset on
multi-asset cards (the digest keeps dropping top-level `blocks` only).
Momentum has no price level, no target and no invalidation price; the fields
say so:

| Field | Momentum meaning | Example |
|---|---|---|
| `what_happened` | the reading and its reason, with the RSI and the histogram's sign | `BTC 4h reads not confirmed: RSI below the 55 threshold (RSI 54.7, MACD histogram above 0).` |
| `why_level` | why the thresholds — there is no price level | `No price level: 55 and 45 are the RSI(14) thresholds, 0 is the MACD histogram line` |
| `scenarios` | exactly two conditional state changes of the rule, never a price direction | `If RSI rises to 55 or above and the MACD histogram stays above 0, the reading turns bullish` · `If RSI falls to 45 or below and the MACD histogram turns below 0, the reading turns bearish` |
| `invalidates` | what ends a **confirmed** reading; `null` otherwise | `A closed 4h candle with RSI below 55 or the MACD histogram at or below 0 ends the bullish reading` |
| `regime` | the asset's **local** momentum state — not a market regime (that is Macro's / Trend's) | `Local momentum · BTC 4h · not confirmed: RSI below the 55 threshold` |

A confirmed read's scenarios are "stays" (both conditions hold) and "turns
neutral" (either fails).

**`results[]` entries** — degraded entries keep `{"asset","ok","reason"}`
only; `ok` entries add:

```json
{"asset": "BTC", "ok": true, "timeframe": "4h",
 "data_as_of": "2026-09-15T08:00:00Z", "freshness": "on_time",
 "verdict": "neutral", "state": "rsi_below_55",
 "why": "RSI below the 55 threshold",
 "rsi": 54.71203, "macd_histogram": 217.3456,
 "blocks": {"what_happened": "…", "why_level": "…", "scenarios": ["…", "…"],
            "invalidates": null, "regime": "…"}}
```

`rsi` and `macd_histogram` are raw precision (a histogram of exactly 0 is
served as `0`); `data_as_of` is the close of that asset's last closed bar
(the envelope's `data_as_of` stays the oldest of them). Indicator output that
is not finite degrades the asset to `insufficient_history`.

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
them. No parameter keeps the global regime card (see [Macro card and content
blocks](#macro-card-and-content-blocks)), which carries one context line per
asset whenever at least one real lamp exists.

- **BTC macro backdrop** — the risk-appetite regime itself, labeled as BTC's
  backdrop: `BTC MACRO BACKDROP: RISK-ON — rule score 83/100; BTC direction
  is not inferred`. One line per lamp with its rule condition and signed
  contribution (`US 10Y 4.9610, session -0.36% (fell) → positive, +7.5`).
- **Gold macro backdrop (experimental)** — a separate model: DXY and US 10Y
  count as in the risk model, VIX and S&P 500 count inverted, weights DXY 40 /
  10Y 25 / VIX 25 / SPX 10, positive > 65, negative < 35; fewer than 3 voting
  lamps → honest `ok:false` / `no_data` (`GOLD MACRO BACKDROP: no read — …`).
  The lamp thresholds are the risk model's and the weights are fixed, not
  validated on gold's history — hence "experimental". The gold lamp itself is
  reported as a fact with its instrument (`GC=F futures` from Yahoo,
  `XAUUSD spot` from stooq) and never votes. The machine `State` keeps its
  values `support` / `pressure` / `neutral` (the Gold Agent branches on them);
  the card words them positive / negative / mixed.
- Unknown-regime honesty carries over per asset: zero real lamps →
  `UNKNOWN` verdict with `market_closed` / `no_data` reason, never a backdrop.

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
      "headline": "Top signal: Trend Agent · BTC — Confirmed UPTREND",
      "one_liner": "🟢 Trend BTC: confirmed uptrend",
      "data_as_of": "2026-08-25T12:31:04Z",
      "example_url": "/agents/digest",
      "digest_status": "live"
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `generated_at` | When the **sweep** ran, not when the request arrived — with the 60s memo a render can legitimately serve a payload up to a minute old, and saying so is the honest form of a cache |
| `live_count` / `total_count` | Rows with `ok: true` on this sweep, out of every agent that exists. The `digest` row counts by the sweep's health (see `ok` below), not by its highlighted card |
| `status` | `live` \| `degraded` — **never `planned`**. This endpoint only lists agents whose builder actually ran; there is no fictional state and no roadmap entry here. For `digest`, `partial` reads `live` here; the three-state value is `digest_status` |
| `ok` / `reason` | The same machine-readable pair the agent envelopes carry (`source_offline`, `insufficient_history`, `below_threshold`, `no_data`, `market_closed`), `null` when `ok`. **Exception — the `digest` row:** the pair speaks for the whole sweep (`ok: false` only when no section is live, `reason` = the first degraded section's), so it can differ from the top-level pair of `/agents/digest`, which describes the highlighted card. Example: macro and the funding/momentum/trend trio offline, whale live → this row `ok: true`, `digest_status: "partial"`; `/agents/digest` `ok: false`, `"reason": "source_offline"` |
| `digest_status` | `digest` row only: `live` \| `partial` \| `degraded` — the same value `/agents/digest` serves as `digest.status`. Absent on every other row |
| `headline` | The card's verdict line — for `digest`, the prioritized `Top signal: <agent> · <asset> — <verdict>` line (the same line as `/agents/digest` `verdict`). The asset is named whenever the winning card has one — since 2026-09-15 the multi-asset momentum card too (`Momentum Agent · BTC/ETH/XAUUSD`), because its verdict became a counter; funding and macro stay bare |
| `one_liner` | The digest-style one-liner, plain text |
| `category` | `crypto` \| `forex` \| `metals` \| `macro` \| `onchain` \| `derivatives` \| `news` \| `tools`. `tools` holds the three that are not a single-market read: `digest`, `top`, `risk`; `metals` holds the gold agent |
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
  "detected": "Trend Agent on BTC — Confirmed UPTREND · 4h.",
  "explained": "ADX(14) at 31.2 is at or above the 25 confirmation threshold with EMA50 over EMA200.",
  "data": [
    "Price 119800 — inside the pullback zone 118420–119950",
    "Confirmation holds while all four conditions stay ✓; any ✗ withdraws it",
    "Invalidated by a closed 4h candle below 110350 (-7.9%, 1 ATR under the EMA cluster)",
    "Why: ADX 31.2 ≥ 25 ✓ · EMA50 > EMA200 ✓ · close > EMA50 ✓ · structure not against ✓ (not determined)"
  ],
  "conclusion": "For a trader this is a bullish reading on BTC: the numbers above lean up, and the read holds only for as long as they do. A closed 4h candle below 110350 invalidates the uptrend idea.",
  "levels": { "invalidation": 110350.2, "invalidation_side": "below" },
  "example_url": "/agents/trend",
  "disclaimer": "Analytics, not financial advice",
  "data_as_of": "2026-08-25T12:00:00Z"
}
```

- **Which agent tells the story**: the same deterministic priority rule
  `/top` uses (`topSelection` — see [digest readout](#digest-readout)). If
  that winner is **degraded**, the story falls back instead of narrating a
  dead source: first funding/momentum/trend under the digest's own rule
  (live and fresh only, confirmed readings first — never a flat trend on raw
  ADX), then the first live card in the fixed order macro → whale → S/R →
  volatility → FX → narrative radar. No cross-agent "strongest" is computed:
  those scales are not comparable. `digest`, `top` and `risk` are never the
  subject (an aggregate is not one agent's story, and the calculator has no
  "detected" moment).
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
- The **invalidation sentence** in `conclusion` appears only for a
  **confirmed** trend and is the card's own `blocks.invalidates` string;
  flat / grey / conflict trend stories carry none (there is nothing to
  invalidate). A flat card reads `"For a trader there is no trend to read on
  BTC: ADX 17.3 is under 20."`; grey / conflict name the checklist items that
  are not met (`"For a trader this is an unconfirmed trend on GOLD · COMEX
  GC=F; failing: ADX."`) — never "nothing leans", since an unconfirmed trend
  card usually shows a clear EMA lean.
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

## Trend chart

`GET /agents/trend/chart?asset=<btc|eth|eurusd|gbpusd|usdjpy|xauusd>` (default
`btc`, same aliases as `/agents/trend`) returns the data for a live chart that
draws **exactly what the Trend Agent reads**. It is not an envelope: it has no
`facts`/`card_html`. State, verdict, price, pullback zone and invalidation come
from the same single read `/agents/trend` uses, so the two endpoints cannot
disagree for the same bars.

```json
{
  "agent": "Trend Agent", "asset": "BTC", "asset_key": "btc", "timeframe": "4h",
  "ok": true, "reason": null,
  "state": "up", "verdict": "Confirmed UPTREND", "price": 78189.0,
  "candles": [{"time": 1757894400, "open": 0, "high": 0, "low": 0, "close": 0}],
  "ema20": [{"time": 1757894400, "value": 0}], "ema50": [], "ema200": [],
  "pullback_zone": {"low": 0, "high": 0},
  "invalidation": {"level": 0, "side": "below"},
  "pivots": [{"time": 1757894400, "price": 0, "kind": "high", "label": "HH"}],
  "structure": "hh_hl",
  "data_as_of": "2026-09-15T08:00:00Z",
  "source": "binance",
  "live": {"provider": "binance", "symbol": "BTCUSDT", "interval": "4h"},
  "disclaimer": "Analytics, not financial advice"
}
```

| Field | Meaning |
|---|---|
| `asset` | Human label, same as the card header. For gold it is `"GOLD · COMEX GC=F"`: the chart draws **futures** prices, not spot XAUUSD |
| `asset_key` | Lowercase registry key (`btc`, …, `xauusd`), aliases resolved |
| `timeframe` | The agent's own interval: `4h` for BTC/ETH (Binance), `1h` for FX and gold (Yahoo). Not configurable |
| `state` / `verdict` | `flat \| grey \| up \| down \| conflict` and the verdict string, identical to `/agents/trend` (`verdict` there, `state` from the same state machine, after the structure gate) |
| `price` | Close of the last closed bar = `candles[-1].close` |
| `candles` | **Closed bars only**, oldest first, the newest 200 of the agent's window. `time` = unix seconds UTC of the bar **open** (lightweight-charts convention) |
| `ema20` / `ema50` / `ema200` | Computed over the **full** agent window (so the last point is the exact value the agent read), then cut to the returned candles. A point exists only where the EMA is defined: at window bar *i* only once *i*+1 ≥ period. The trend window is 999 closed bars on Binance and ~500 on Yahoo, so all three cover every returned candle. Every point's `time` matches a candle |
| `pullback_zone` | EMA20-EMA50 band as `{low, high}` (low ≤ high). **Only in `up`/`down`**, `null` otherwise |
| `invalidation` | `{level, side}` from the same rule as the card (uptrend: min(EMA50,EMA200) − 1 ATR, `below`; downtrend: max + 1 ATR, `above`). **Only in `up`/`down`**, `null` otherwise — the same presence rule and value as `levels.invalidation` on `/agents/trend`: an unconfirmed state has no level anywhere |
| `pivots` | The agent's swing points (wing 3) that fall inside the returned window, chronological. `kind` = `high`\|`low`. A same-bar high+low appears as two entries |
| `pivots[].label` | Set **only** on pivots the structure read compared, and only when `structure` is readable: within the last six alternating pivots, the 2nd and 3rd high are labelled vs the previous high (`HH`/`LH`), the 2nd and 3rd low vs the previous low (`HL`/`LL`). So at most **4** pivots carry a label. The first high and first low of that tail are the comparison baselines and carry `""`; a tie (equal price, which is what makes a read `mixed`) also carries `""`. Every other pivot, and all pivots when `structure` is `""`, carry `""` |
| `structure` | `hh_hl \| lh_ll \| mixed \| ""`. `""` = the agent could not read a structure (too few pivots or non-alternating tail), which counts neither for nor against the trend |
| `data_as_of` | RFC3339 UTC **close** time of the last closed bar. For Binance assets also sent as `Last-Modified` (`If-Modified-Since` → `304`); Yahoo assets send no validator — see [caching](#caching-last-modified-and-304) |
| `source` | `binance` \| `yahoo` |
| `live` | For Binance assets, the public kline WebSocket to continue the chart in the browser (`wss://stream.binance.com:9443/ws/<symbol lowercased>@kline_<interval>`). `null` for Yahoo assets: that data is delayed, poll this endpoint instead (a new bar appears at most once per `timeframe`) |

Errors follow the rest of the API: unknown asset → `400` naming the allowed
values; a repeated `?asset=` or any `?tf=`/`?assets=` → `400`; source offline →
`503 {"error": …, "ok": false, "reason": "source_offline"}`; too few closed bars
→ `503 … "reason": "insufficient_history"`. A `503` carries no
`Last-Modified`. A `200` always has `ok: true, reason: null`.

A live candle from the WebSocket is **forming**: the agent's reading (state,
zone, invalidation, pivots) only moves when a bar closes and this endpoint is
re-fetched. The browser should not recompute the verdict from forming bars.

```bash
curl -s 'localhost:8090/agents/trend/chart?asset=eurusd' | jq '{state, verdict, price, n: (.candles|length), zone: .pullback_zone, live}'
```

## Response envelope

Every agent endpoint answers with one shape:

```json
{
  "agent": "Trend Agent",
  "asset": "EURUSD",
  "ok": true,
  "reason": null,
  "verdict": "Grey zone · 1h — trend forming, not confirmed",
  "semaphore": "neutral",
  "facts": ["Price 1.1590 — 0.1% above EMA50 1.1583 and 0.3% above EMA200 1.1560", "..."],
  "blocks": {"what_happened": "...", "why_level": "...", "scenarios": ["...", "..."], "invalidates": null, "regime": "..."},
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
| `levels` | object | **trend / sr / vol only**: raw-precision numeric levels — see [levels](#machine-readable-levels). Absent for other agents, on `ok: false` cards, and **on trend cards that are not a confirmed trend** (flat / grey / conflict — since 2026-09-15 there is nothing to invalidate there) |
| `results` | array | **momentum only**: per-asset machine outcomes `{"asset","ok","reason"}` — see [momentum scan](#momentum-scan-assets--tf). Since 2026-09-15 also on the single-asset momentum card (one entry), and every `ok` entry carries the read itself (`rsi`, `macd_histogram`, `verdict`, `state`, `why`, `timeframe`, `data_as_of`, `freshness`, `blocks`) — see [momentum card](#momentum-card-and-content-blocks). Absent elsewhere |
| `blocks` | object | **trend, sr, the global macro card and the single-asset momentum card** (additive): ready-made sentences for content — see [trend card and content blocks](#trend-card-and-content-blocks), [S/R card and content blocks](#sr-card-and-content-blocks), [macro card and content blocks](#macro-card-and-content-blocks) and [momentum card](#momentum-card-and-content-blocks) (multi-asset momentum cards carry them per asset, in `results[].blocks`). Absent for every other agent, on `ok: false` cards, on the S/R "No significant levels detected" finding and on the macro asset views; on `/agents/top` they belong to the winning card (never on the digest, below) |
| `macro` | object | **macro cards only** (additive, 2026-09-15): the numbers behind the card — rule score, bands, per-lamp rule / weight / contribution / source / `as_of`, freshness, Fear & Greed age. See [macro card](#macro-card-and-content-blocks). Absent for every other agent and on macro cards without a reading (`UNKNOWN`, offline) |
| `confidence` | int \| null | 0–100 when the source supplied one, otherwise `null` — never invented. **Macro is always `null` since 2026-09-15**: its 0–100 composite is a **rule score**, not a confidence or a strength, and ships in the verdict (`RISK-ON — rule score 83/100 (risk-on above 65, risk-off below 35)`) and in `macro.rule_score`. The `?asset=gold` view has its own `gold score` and also serves `confidence: null` |
| `ai_text` | string \| null | Plain-text AI block (mood read / idea / brief / why-line); `null` when AI is disabled or the call failed |
| `sections` | string[] | **digest only**: plain-text one-liners of every other agent (the winner heads the envelope) |
| `digest` | object | **digest only** (additive, 2026-09-15): unified status, how the highlighted card was selected, and every block `card_html` renders below it (FX and narrative included), each with its own data time — see [digest readout](#digest-readout) |
| `data_as_of` | string | RFC3339 UTC; for candle-based agents this is the **close time of the last closed bar used** — the same stamp as the card footer. **Macro** (since 2026-09-15): the **oldest** `as_of` among the live lamps (the source's session stamp), never the response build time — `UNKNOWN` macro cards keep the response time |
| `Last-Modified` (header) | HTTP date | The validator for `If-Modified-Since` → `304`. Sent **only** where the whole body is a function of the stamped data; which addresses carry it and what it means is in [caching](#caching-last-modified-and-304). `data_as_of` is never affected by it |
| `disclaimer` | string | Always `"Analytics, not financial advice"` |
| `card_html` | string | The exact Telegram HTML message the bot would send (for `digest`/`top`: the full composed message) |

## Caching: Last-Modified and 304

A response carries `Last-Modified` only when its **whole body** is a function
of the data that stamp names. Then a request with `If-Modified-Since` at or
after the stamp gets `304` with no body. Where a component can change while
any single stamp stays put, the response carries **no** `Last-Modified`,
`If-Modified-Since` is ignored, and the answer is always `200` with the body.
`data_as_of` and the card footer are the same either way.

Changed 2026-09-15: `Last-Modified` is left only on single-asset Binance
reads (cards and the trend chart) and `/showcase`. Every Yahoo read (FX pairs
and gold, cards and the trend chart), gold, fx, funding, multi-asset
momentum, news, whale and macro no longer send one. A Binance bar now counts
as closed only if it had closed when the cached candles were fetched, and a
Binance read keeps its stamp only on a complete, contiguous window. The
whale top-3 window now hangs off the snapshot's `captured_at`.

| Address | `Last-Modified` | What it means / why there is none |
|---|---|---|
| `/agents/trend`, `/agents/sr`, `/agents/vol` and `/agents/momentum?asset=` for `btc`, `eth` (Binance; `?tf=` applies to momentum only, the other three answer it with `400`) | yes, **when the Binance window is complete** (below); momentum also not while its read is `data_delayed` (that wording follows the clock, not the bars) | Close of the last closed bar used, equal to `data_as_of`. A bar counts as closed only if it had closed when the candles were fetched (they are cached up to 60 s) **and** Binance already returned the bar after it: a klines answer always ends with the bar still forming, so its last row is never read. A bar with intermediate prices therefore waits for a later fetch rather than appear with numbers that change under the same stamp |
| `/agents/momentum?assets=` with **one** Binance asset | yes, when the window is complete and the read is not `data_delayed` | Same as the single card |
| `/agents/trend/chart` for `btc`, `eth` | yes, when the window is complete | `data_as_of`: the chart reads the closed bars only |
| `/agents/trend`, `/agents/sr`, `/agents/vol`, `/agents/momentum?asset=` for `eurusd`, `gbpusd`, `usdjpy`, `xauusd` (Yahoo, aliases included), `?assets=` with one Yahoo asset, and `/agents/trend/chart` for those assets | **no** | No stamp versions a Yahoo body. Yahoo can publish a bar late and can revise the OHLC of a bar it already served under the same timestamp, so the bar close stays put while the numbers change. On the cards, the market-state wording (the `⏸ Forex market closed` banner, `market closed` on a momentum asset line) follows the clock of the fixed-UTC week (closes Friday 21:00, opens Sunday 21:00 UTC), not the bars |
| `/agents/news` | **no** | The AI idea is generated and cached by the backend per narrative and hour, and a failed generation is not cached, so an idea can appear or change under the same `captured_at` |
| `/agents/whale` | **no** | The transfer list comes from the backend's live table (the newest rows), not from the snapshot `captured_at` names: a new transfer pushes an old one out under the same `captured_at`. The top-3 shows transfers of the 24h up to `captured_at`, none stamped after it (with no parseable `captured_at`: the 24h up to the request time) |
| `/agents/macro`, `?asset=btc`, `?asset=gold` | **no** | The backend's `captured_at` is its request time at one-second resolution, so two different payloads can share it; no lamp stamp versions the body either (a lamp value moves during its session under the same `as_of`) |
| `/showcase` | yes | The sweep time (`generated_at`): the body is fixed for the life of the memoized sweep (up to 60 s) |
| `/agents/gold` | **no** | Daily bars, the hourly price and its age, the macro payload and the weekend clock. The daily close versions none of the rest |
| `/agents/fx` | **no** | Four series plus the banner: an older pair can update, drop out or recover while the newest close stays put |
| `/agents/funding` | **no** | Point-in-time reads (rates, the liquidation feed, a 1h window from the request time, the BTC price). The request time is not a version of that body |
| `/agents/momentum` without params, `?tf=` alone, `?assets=` with two or more assets | **no** | Composite: the oldest bar (`data_as_of`) can stay put while another asset, the ETH-vs-BTC read, an asset's freshness (`market closed` / `data delayed`) or a source failure/recovery changes the body |
| `/agents/digest`, `/agents/top`, `/showcase/example` | **no** | The digest re-sweeps per request; `/top` and the example add per-request AI text that reads the whole sweep |
| `/agents/risk` | **no** | A pure function of the query; its data time is the answering time |
| `/`, `/agents` | **no** | Static text |
| Any `4xx` / `5xx` (including `503` degraded cards) | **no** | A cached failure would keep an agent dark after its source recovers |

**Binance window condition.** Binance always ends a klines answer with the
bar still forming, so a Binance read never uses the **last row** it received:
a bar counts as closed only when the bar after it is already in the answer
(and it had closed by the fetch time). This also covers a REST answer that
lags the close and a fast local clock. The read then uses a fixed window set
by its last closed bar: the newest **999** closed bars for trend and the trend
chart, the newest **249** for momentum, S/R and volatility (the agent's raw
window minus that last row). It keeps `Last-Modified` only when **exactly**
that many closed bars are there and they are **contiguous**, each opening
exactly one interval after the previous. The parser skips malformed or
non-finite rows, and Binance can return a shorter window, and either way
indicators, levels and the chart can change under the same last bar. A read
that fails the check serves the **same body with no `Last-Modified`**.

**Assumption, not checked in code:** a closed Binance candle is treated as
final — the source is assumed never to revise the OHLC of a bar after its
close. The stamp relies on it; nothing in the demobot verifies it.

## Machine-readable status: ok / reason

Templates must branch on **why** a value is absent, not on verdict strings.
Every envelope carries the pair; when `ok` is `false`, `reason` is one of:

| `reason` | Served by | Meaning |
|---|---|---|
| `market_closed` | macro | Regime `UNKNOWN` outside the clock-based tradfin week (Sun 22:00 → Fri 21:00 UTC) — no lamps to read. The window has no holiday calendar, so the card words it `scheduled tradfin weekend`, never "market closed"; the enum value is kept for compatibility |
| `source_offline` | any agent | The source behind the headline reading is unreachable. Usually a `503`; also a `200` on the funding card when rates are dead but the liquidation feed is alive (liq facts still render) |
| `insufficient_history` | momentum, trend, sr, vol | Source alive, but too few **closed** bars for the indicator set (always a `503`). For `trend` also when an indicator or the derived level comes out non-finite (NaN/Inf, e.g. overflowing prices) — such input degrades instead of producing a reading |
| `below_threshold` | news | Narrative radar warming up: the top theme is under 5 mentions/24h (`200`, themes listed without scores), or there are no snapshots yet (`503`) |
| `no_data` | macro, whale | Upstream alive but nothing to read: macro unknown **inside** the open tradfin window; whale feed with no BTC snapshot yet |

For `digest` / `top` the pair (and `levels`) describes the **top signal
card** heading the envelope — with every source dead, the honest macro
fallback yields `ok: false`, `"reason": "source_offline"` inside the 200.
The health of the whole digest sweep is **not** in this pair: it is
`digest.status` (`live` \| `partial` \| `degraded`, the same value as the
`/showcase` digest row's `digest_status`), with `digest.live_sections` and
`digest.degraded_sources`. `digest.selection.highlight_ok` /
`highlight_reason` repeat the top-level pair. `blocks` (one agent's content sentences) ship on
`/agents/top` only — the digest envelope never carries them, since they
describe that one card, not the whole sweep (fixed 2026-09-15).

### Digest readout

`digest` on `/agents/digest` (additive). The rule, in order:

1. Macro `risk_off` takes the slot only when the card is `ok`, has at least
   the rule's minimum voting lamps behind a rule score, and its oldest live
   lamp is at most **80 h** old (72 h weekend gap between session stamps + 8 h
   for a late feed). Otherwise `selection.macro_risk_off_gate` says why
   (`degraded` | `partial_lamps` | `stale`) and the crypto rule decides.
2. Among funding, momentum, trend: only **eligible** candidates compete —
   status `ok` and fresh: funding ≤ **15 min** (a request-time read),
   momentum and trend ≤ **8 h** (two 4 h bars; momentum judged on its BTC/ETH
   bars, not gold). Momentum also needs at least one BTC/ETH reading: a card
   with only gold/FX read is excluded as `no_ranked_read`. The strongest **confirmed** reading wins (funding
   crowded, momentum bullish/bearish on BTC/ETH, trend up/down); ties
   funding > momentum > trend.
3. Scores (0–100, **not calibrated against each other**): funding = widest
   rate against its own side's threshold (either threshold = 30; +0.10% or
   −0.033% = 100); momentum = |RSI−50|×2 of a confirmed read, else 0; trend =
   ADX×2 when confirmed, else 0.
4. Nothing confirmed → `selection.state: "no_highlight"`. Until the product
   decision the digest still shows a card (the highest eligible unconfirmed
   score, ties momentum > trend > funding — a balanced funding never wins a
   tie); `selection.line` says nothing was confirmed. No eligible candidate
   at all → the macro card (`fallback_macro`).

| Field | Meaning |
|---|---|
| `status` | `live` \| `partial` \| `degraded` over the seven digest agents + the FX block. Same value as the `/showcase` digest row's `digest_status` |
| `live_sections` / `total_sections` / `degraded_sources` | Coverage behind `status` |
| `generated_at` | Sweep time; freshness is judged at this instant |
| `selection.state` | `selected` \| `no_highlight` |
| `selection.rule` | `macro_risk_off` \| `strongest_confirmed` \| `fallback_unconfirmed` \| `fallback_macro` |
| `selection.winner` | Agent key of the highlighted card |
| `selection.line` | The one-line reason printed under the digest header (≤110 chars). Since 2026-09-15 it names what each agent's ranked reading covers: `Funding, Momentum (BTC/ETH), Trend (BTC)` — momentum ranks its BTC/ETH reads only, so "No confirmed reading" can sit next to a momentum card that counts a confirmed gold read (that read never colours the card, see [momentum card](#momentum-card-and-content-blocks)) |
| `selection.highlight_ok` / `highlight_reason` / `highlight_data_as_of` | The highlighted card's own status and data time |
| `selection.macro_regime` / `macro_risk_off_gate` | The regime; the gate outcome only when it is `risk_off` |
| `selection.scales_calibrated` | Always `false` for now |
| `selection.candidates[]` | `{agent, eligible, excluded (degraded\|no_ranked_read\|stale\|no_data_time\|null), confirmed, score, data_as_of, max_age_minutes}`. `no_ranked_read` (2026-09-15): the momentum card carries no BTC/ETH reading (both offline or short history) — its gold/FX reads are shown but never ranked, so it does not compete instead of being judged on the gold bar |
| `sections[]` | Every block `card_html` renders below the highlighted card, in order: `{key, title, lines, ok, reason, data_as_of}`; `key` is an agent key, `fx` or `narrative`; `lines` are the plain-text lines exactly as rendered |

The `503` error body carries the same pair beside the message, so single-agent
degraded states are branchable too:

```json
{"error": "Trend Agent: Insufficient history — no verdict", "ok": false, "reason": "insufficient_history"}
```

## Machine-readable levels

> **What changed 2026-09-15 (trend).**
> 1. `levels.invalidation` and `levels.invalidation_side` are now
>    **conditional**: present only in a confirmed trend (`up`/`down`).
>    Flat, grey (structure-demoted included) and conflict cards carry **no
>    `levels` object at all** — previously they shipped an invalidation
>    number for every state. There is never a zero level and never a side
>    without a level. Field names are unchanged.
> 2. On Binance assets (BTC, ETH) the Trend Agent and `/agents/trend/chart`
>    read **999 closed 4h bars** (1000 raw) instead of 249, so EMA200 matches
>    its converged value. Momentum, S/R and volatility still read 249. FX and
>    gold (Yahoo) windows are unchanged.

> **What changed 2026-09-15 (S/R).** Wording and fields only — the rules
> (swing wing 3, cluster tolerance 0.5%, top 3 by strength per side,
> test/break counting, window) are unchanged.
> 1. Each `sr` level gains four **additive** fields: `label`, `class`,
>    `display_rank`, `strength_rank`. Existing fields and the array order
>    (strength) are unchanged; `weakening` keeps its name and meaning.
> 2. S/R envelopes now carry `blocks` (same shape as trend) whenever at
>    least one level is shown.
> 3. Card text: prices at instrument precision (USDJPY no longer prints
>    154.938 as "155"); "held X of Y tests" → "X reactions / Y breaks";
>    "S1/R1" → "nearest shown support/resistance"; each level names its
>    class and last touch; "weakening: volume fading" → "Last 3 pivots on
>    lower volume than first 3"; the verdict is no longer "Key levels
>    around …" but where price is against the nearest shown level. The
>    method paragraph moved to the how-it-works text.

Three agents' readings *are* price levels; their envelopes add a `levels`
object with the raw computed numbers — full float precision, never the
display-rounded strings shown in `facts`:

| Agent | `levels` shape |
|---|---|
| `trend` | **Confirmed trend (`up`/`down`) only** — flat, grey (incl. structure-demoted) and conflict envelopes have **no `levels` key**. Shape: `{"invalidation": 63297.4, "invalidation_side": "below", "pullback_zone": {"from": 64850.1, "to": 64210.7}}`. `invalidation` is **direction-aware**: an uptrend's sits BELOW the EMA cluster (min(EMA50,EMA200) − 1 ATR, `"below"`), a downtrend's ABOVE it (max + 1 ATR, `"above"`), checked on a **closed** candle of the agent's timeframe. `invalidation` and `invalidation_side` are present together or not at all: both are absent only on a degenerate series with no ATR, leaving `{"pullback_zone": …}`. `pullback_zone` is the EMA20-EMA50 band, always present when confirmed (`from` = EMA20, `to` = EMA50 — so `from` > `to` in an uptrend) |
| `sr` | `{"supports": [{"level": 76406.66375, "label": "76407", "class": "established", "display_rank": 1, "strength_rank": 1, "touches": 8, "strength": 11, "weakening": false, "breaks": 2, "holds": 1, "last_touch": "2026-09-14T00:00:00Z"}, …], "resistances": […]}` — strength-sorted raw cluster means (the three strongest per side); an empty side is `[]`, never `null`. **The card lines render the same three levels nearest-first** — `levels` keeps strength order, so `supports[0]` is the strongest, not necessarily the nearest. The order is explicit per point: `strength_rank` = 1-based position in this array, `display_rank` = 1-based position of the level's line on the card within its side (nearest to price = 1). `label` = the level exactly as the card prints it (instrument precision: BTC 0 decimals, ETH and gold 1, EURUSD/GBPUSD 4, USDJPY 2); every % on the card is computed from the printed numbers. `class` = `established` (touches ≥ 7) · `candidate` (2–6) · `single_swing` (1). On the card `touches` is worded "pivots"; `holds` are worded "reactions". `strength` = touches + 0.5 per touch on above-median volume (median over NON-ZERO volumes; on a level whose touches are mostly volume-less the volume features disable and `strength` equals `touches`). `weakening` = ≥7 touches with the last 3 touches' mean volume below the first 3's (same volume gate). `breaks`/`holds` are **frequency counts** of level tests over the 249-bar window: a test = a close entering the ±0.25×ATR band (ATR frozen at the entry bar); within 3 bars a close beyond the level on the far side = **break**, a close back beyond the band on the approach side = **hold** (a hold IS the rejection); price stalling inside the band for 3 bars is **unresolved and dropped** — never counted as a hold. Frequencies, never probabilities. `last_touch` = RFC3339 UTC of the newest touch — the later of the last swing in the cluster and the last close-test of the level |
| `vol` | `{"expansion_ratio": 1.01}` — ATR(14) over its 30-bar average, unrounded |

`sr` with **no levels at all** is never a "Key levels around …" reading: a
window with ZERO swing points (monotone/flat tape) degrades to a `503`
`insufficient_history`; swings present but no cluster clear of the last price
serves an explicit ok `"No significant levels detected in the window"` with
empty arrays.

`trend` confirmation **must not contradict the swing structure**: an EMA/ADX
uptrend with LH/LL (or mixed) pivots demotes to the grey zone, verdict
`Grey zone · 4h — not confirmed: swing structure against the trend` (mixed
pivots: `… swing structure not aligned`) — the pullback zone disappears with
the demotion. HH/HL is read from the last six ALTERNATING pivots in time
order; a window that cannot be read (too few pivots, or a non-alternating
tail) never demotes and is shown as `structure not against ✓ (not
determined)`. The card names no pivot patterns.

## Trend card and content blocks

The trend card reads top to bottom: **verdict with timeframe → where price
is → what keeps or changes the reading → the checklist**. Wording only; the
rules (ADX 20/25, EMA alignment, structure gate, EMA20–EMA50 zone, 1 ATR)
are unchanged.

```
🔴 Trend Agent · EURUSD
Confirmed DOWNTREND · 1h
• Price 1.1551 — 0.1% below the pullback zone 1.1559–1.1580
• Confirmation holds while all four conditions stay ✓; any ✗ withdraws it
• Invalidated by a closed 1h candle above 1.1619 (+0.6%, 1 ATR over the EMA cluster)
• Why: ADX 46.5 ≥ 25 ✓ · EMA50 < EMA200 ✓ · close < EMA50 ✓ · structure not against ✓ (not determined)
```

```
⚪ Trend Agent · GOLD · COMEX GC=F
Grey zone · 1h — trend forming, not confirmed
• Price 4329 — 0.9% below EMA50 4367 and 2.4% below EMA200 4433
• Confirms as a DOWNTREND only when all four conditions are ✓ (now ✗: ADX)
• Why: ADX 22.2 < 25 ✗ · EMA50 < EMA200 ✓ · close < EMA50 ✓ · structure not against ✓ (not determined)
```

- **One checklist is the rule.** The `Why:` line lists the four confirm
  conditions — ADX ≥ 25 · EMA50 vs EMA200 · close vs EMA50 · swing structure
  not against the direction — toward the EMA lean, each marked by the rule's
  own evaluation; all four ✓ is exactly a confirmation. It is the **only**
  place conditions are listed: every other line refers to "all four
  conditions" and at most names the ✗ items. It always has the same four
  items. An unreadable structure never demotes, so it is ✓ `(not
  determined)`; a failing one says why without naming pivot patterns —
  `✗ (runs against the trend)` or `✗ (swings not aligned)`. With EMA50 =
  EMA200 there is no direction: `EMA50 = EMA200 ✗ (no direction) · close vs
  EMA50 ✗ · structure not against ✗`, and the confirm line reads `No
  direction to confirm while EMA50 equals EMA200`.
- **Verdicts**: `Flat · 4h — no trend to read (ADX under 20)` ·
  `Grey zone · 1h — trend forming, not confirmed` ·
  `Grey zone · 4h — not confirmed: swing structure against the trend` ·
  `Confirmed UPTREND · 4h` / `Confirmed DOWNTREND · 1h` ·
  `Indicator conflict · 1h — ADX ≥ 25 but the EMA conditions disagree`.
- **ADX is printed once per card**, in the checklist, floored to one decimal
  (a value below a threshold can never print at or above it). Thresholds
  read `≥ 25` — the rule confirms AT 25. Every line of trend text — facts,
  each `blocks` field and the landing conclusion — fits ≤ 110 characters.
- **Two different levels, two different words.** Losing any checklist
  condition **withdraws** the confirmation (state goes to conflict or grey).
  The ATR level further out **invalidates** the directional idea (EMA
  cluster ± 1 ATR).
- **Invalidation line only on confirmed trends**, checked on a closed candle
  of the agent's timeframe. Unconfirmed states show what would confirm
  instead. RSI is not on the trend card (it is not part of the rule).
- Distances are % of the current price; `<0.1%` when smaller.

`blocks` on trend envelopes (additive; S/R has its own wording, next section):

| Field | Meaning |
|---|---|
| `what_happened` | State, timeframe and where price is: `"Confirmed downtrend on 1h: price 1.1551 — 0.1% below the pullback zone 1.1559–1.1580."` |
| `why_level` | What the level is made of — confirmed: `"1.1619 = max(EMA50, EMA200) + 1 ATR(14) on a closed 1h candle; a checklist ✗ withdraws confirmation sooner"`; otherwise `"No invalidation level: the trend is not confirmed. Confirmation needs all four checklist conditions"` (and no `levels` object) |
| `scenarios` | Exactly two `"If … , the reading …"` sentences. A confirmed state is named only with the whole checklist (`"If all four conditions turn ✓ for a downtrend, the reading confirms as a downtrend"`); the other is a transition the rule fully determines (`"If ADX falls below 20, the reading returns to flat"`, or for confirmed trends `"If a 1h candle closes above 1.1619, the reading is invalidated as a downtrend (1 ATR over the EMA cluster)"`). No targets, no probabilities, no forecast |
| `invalidates` | Confirmed trends: `"A closed 1h candle above 1.1619 invalidates the downtrend idea"`; `null` for flat / grey / conflict |
| `regime` | One line: `"confirmed downtrend · 1h · ADX 46.5"` |

`levels` is absent for every other agent and on degraded (`ok: false`)
envelopes.

## S/R card and content blocks

Reads top to bottom: **where price is against the nearest shown level →
every shown level, each side nearest first → observations → window**.
Wording only; the rules are unchanged (see the `sr` row above).

```
⚪ S/R Agent · ETH
Price 2515.8 — 0.6% below the nearest shown resistance 2531.0 (established, 7 pivots)
• Resistance 2531.0 (+0.6%) · established, 7 pivots · 5 reactions / 1 break · last touch Sep 14
• Resistance 2546.5 (+1.2%) · candidate, 3 pivots · 1 reaction / 1 break · last touch Sep 14
• Resistance 2666.0 (+6.0%) · single swing, 1 pivot · no resolved tests · last touch Sep 11
• Support 2436.4 (-3.2%) · candidate, 5 pivots · 5 reactions / 3 breaks · last touch Sep 11
• Support 1892.1 (-24.8%) · established, 7 pivots · 4 reactions / 2 breaks · last touch Aug 18
• Support 1862.4 (-26.0%) · candidate, 6 pivots · 1 reaction / 0 breaks · last touch Aug 16
• Last 3 pivots on lower volume than first 3: 2531.0, 1892.1
• Window: 249 closed 4h candles · test = a close within 0.25 ATR of a level, resolved within 3 candles
```

- **Nearest shown, not nearest overall.** The card shows the three strongest
  levels per side; the verdict names the nearest of those. No `S1/R1` labels.
- **Neutral test counts.** A test is a close entering ±0.25 ATR of the level;
  the first close out of that band within 3 candles resolves it — back on the
  approach side = **reaction**, past the far edge = **break**; unresolved
  tests are dropped. The counting does not record which side price came
  from, so the card never says "support held".
- **Classes**: established (≥ 7 pivots) · candidate (2–6) · single swing (1).
- **Precision per instrument** (BTC 0, ETH 1, gold 1, EURUSD/GBPUSD 4,
  USDJPY 2 decimals). Distances are computed from the printed price and
  level; when the two print identically the card says `at`, never a %.
- **An empty side is said plainly**: `No clustered support below price in
  this 504-candle 1h window`.
- The volume flag (`weakening` in JSON) is an observation — no claim that the
  level is weaker. The card line names a level only when all six compared
  pivots (its first 3 and last 3) carry volume; `weakening` in JSON is served
  exactly as the rule computes it, which needs volume on only half the pivots.
- Every line — verdict, facts, each `blocks` field — fits ≤ 110 characters;
  the method is the how-it-works text (≤ 200, also the catalog description).

`blocks` (S/R envelopes with at least one shown level; absent on the
"No significant levels detected" finding and on degraded cards). All fields
are about the nearest shown level:

| Field | Meaning |
|---|---|
| `what_happened` | `"On 4h: price 2515.8 — 0.6% below the nearest shown resistance 2531.0 (established, 7 pivots)."` |
| `why_level` | What the level is made of: `"2531.0 = mean of 7 pivots · 5 reactions / 1 break in 6 resolved tests · last touch Sep 14"` |
| `scenarios` | Exactly two market events, worded from the side of price the level is on now: `"If a 4h close tests 2531.0 and a close within 3 candles exits its band below, the level holds as resistance"` / `"If a 4h candle closes above 2531.0's band, the level is broken and moves below price"` (mirrored for a support below price). No counts, no targets, no probabilities, no forecast |
| `invalidates` | What makes the level no longer this side: `"A closed 4h candle above 2531.0 puts it below price: it no longer reads as resistance"` |
| `regime` | Local level context only (not macro, not trend): `"Levels on both sides · nearest shown: resistance, 0.6% away · 4h"`, or `"Resistance only, none below price · nearest shown 0.7% away · 1h"` |

## Macro card and content blocks

The global card (`/agents/macro`, 2026-09-15) says what five tradfin prices
look like under a fixed rule, and nothing more: no forecast for BTC or gold,
no causes ("favors crypto", "haven bid" are gone), no confidence. The rules
themselves are unchanged — lamp thresholds (DXY / US 10Y / Gold: any
session fall is positive, a rise counts negative only above +0.5%; S&P 500
mirrored; VIX by **level**, < 18 positive, > 25 negative), weights
(DXY / VIX / S&P 500 25, US 10Y 15, Gold 10), the composite, bands 35 / 65
and the 3-voting-lamp minimum all live in `internal/macro/compute.go`, and
the card prints them from those constants.

```
🟢 Macro Agent
RISK-ON — rule score 83/100 (risk-on above 65, risk-off below 35)
• Positive: VIX 17.10 (<18) → +12.5 · S&P 500 +0.11% (rose) → +12.5 · 1 more
• Negative: none in this model
• Risk-on holds while the rule score stays above 65 with at least 3 voting lamps
• Data: 5 of 5 lamps live · Sep 14: US 10Y, VIX, S&P 500 · Sep 15: DXY, Gold
• Rule score 83 ≈ 50 + US 10Y +7.5 + VIX +12.5 + S&P 500 +12.5 · neutral: DXY, Gold
• BTC macro backdrop: risk-on (the regime itself); BTC direction is not inferred
• Gold macro backdrop: mixed, gold score 45/100 (experimental model, own weights)
• Crypto Fear & Greed 69 (Greed), Sep 15 · separate index, not in the rule score
```

- **Order**: regime and rule score → main factors (up to two of the side
  with the larger total, one of the other) → what holds the regime → data
  dates → the breakdown → BTC / gold context → Fear & Greed.
- **Contributions**: a voting lamp adds its weight (positive), half of it
  (neutral) or nothing (negative), renormalised over the voting lamps, so
  `score = 50 + Σ vs_neutral`. `=` only when 50 plus the contributions **as
  printed** (one decimal) adds up exactly to the printed score; otherwise `≈`
  (e.g. `83 ≈ 50 + 7.5 + 12.5 + 12.5`, `100 ≈ 50 + 16.7 + 16.7 + 16.7`; the
  gap is always under 1). The numbers print only when they reproduce the
  backend's composite; on a version skew the card shows the plain score.
- **No Last-Modified.** `data_as_of` and the footer are the oldest live lamp.
  Macro sends no validator and always answers `200`: a lamp value moves
  during its session under the same `as_of`, and the backend's `captured_at`
  (its request time, one-second resolution) can be shared by two different
  payloads — see [caching](#caching-last-modified-and-304).
- **Printed numbers never cross their threshold**: `+0.499` prints `+0.50%`
  (neutral), `+0.5001` prints `+0.51%`, `17.999` prints `17.99`.
- **VIX** never shows its session change — the rule reads its level.
- **Freshness**: dates are the lamps' own session stamps (`as_of`), grouped
  on one line when they differ. Outside the clock-based week the first fact
  is `Scheduled tradfin weekend: session changes are from each lamp's last
  session`. The change is Close − session Open, never called "24h". The
  footer names the providers (`lamps: Yahoo`, `lamps: Stooq 2 · Yahoo 3`).
- **Fear & Greed** is a separate index and never part of a score. The backend
  now sends its `as_of` (the index's own UTC day) and `fetched_at`; older
  than **36h** (one daily update missed plus 12h of publishing lag) the card
  says `stale, last update Sep 10 (5d ago)`; without any time it says
  `update time unknown`.
- The backend's `generated_idea` is no longer rendered on the card (the card
  words the same rule itself; an older backend's causal sentence must not
  leak through).
- Every visible line — verdict, facts, each `blocks` field, the footer — fits
  ≤ 110 characters (swept over 3 125 worst-case payloads).

`blocks` (global card only, when there is a reading). Macro has **no price
level**, so `why_level` is always `""`:

| Field | Meaning |
|---|---|
| `what_happened` | `"Lamps (sessions Sep 14 to Sep 15): 3 positive, 0 negative, 2 neutral in this model; rule score 83/100"` |
| `regime` | The main block: `"Risk-on by this model's rule (score 83 above 65): a tradfin backdrop, not a BTC or gold forecast"` |
| `scenarios` | Exactly two regime changes by the bands, conditional: `"If the rule score stays above 65 with 3+ voting lamps, the model keeps reading risk-on"` / `"If the rule score falls to 35-65, the reading turns mixed; below 35, risk-off"` |
| `invalidates` | `"Risk-on ends at a rule score of 65 or below, or with fewer than 3 voting lamps (now 83, 5 voting)"`; mixed: ends above 65 or below 35; `null` without a rule score |
| `context` | Macro only: `"Backdrops: BTC risk-on (direction not inferred) · gold mixed (experimental) · F&G 69, not scored"` |
| `why_level` | Always `""` for macro |

`macro` (all macro cards with a reading):

| Field | Meaning |
|---|---|
| `model` | `risk_appetite` (global, `?asset=btc`) \| `gold_backdrop` (`?asset=gold`) |
| `experimental` | `true` for the gold model |
| `reading` | `risk_on` \| `mixed` \| `risk_off` (risk model — the backend regime); `positive` \| `mixed` \| `negative` \| `no_read` (gold) |
| `rule_score`, `rule_score_unrounded` | The rounded score (backend composite for the risk model) and the unrounded sum (`null` when the numbers do not reproduce the score) |
| `bands` | `{"low": 35, "high": 65}` — below `low` / above `high` is the reading's side, both edges belong to mixed |
| `min_voting_lamps`, `voting_lamps`, `live_lamps` | The rule's minimum and the counts behind this card |
| `lamps[]` | `key`, `label`, `instrument` (gold: `GC=F futures` \| `XAUUSD spot`), `value`, `delta_pct` (Close − session Open), `rule` `{input: "level" \| "session_change_pct", positive_when, negative_when}`, `contribution` (`positive` \| `neutral` \| `negative`, `""` = not voting), `voting`, `weight` (nominal), `points`, `max_points`, `vs_neutral` (renormalised; `null` without a score), `source`, `as_of` |
| `freshness` | `oldest_as_of` (= `data_as_of`), `tradfin_as_of`, `session_dates[]`, `mixed`, `scheduled_weekend`, `captured_at` (backend response time), `sources` (provider → live lamp count) |
| `fear_greed` | `value`, `label`, `as_of`, `fetched_at`, `age_hours` (`null` without a time), `stale`, `stale_after_hours` (36), `in_score` (always `false`); `null` without a live value |
| `is_forecast` | Always `false` |

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
  so `data_as_of` can legitimately lag wall clock by up to one interval, plus
  up to 60 s after a close. On every candle-based card (Binance and Yahoo:
  trend, S/R, vol, momentum single and composite, fx, gold, the trend chart)
  a bar counts as closed only if it had closed when the cached candles were
  fetched; Binance reads also never use the last row of the answer (the bar
  still forming), so a new Binance bar appears once a fetch returns the bar
  after it.
- **Conditional requests**: send back exactly the `Last-Modified` you
  received. A cache that builds `If-Modified-Since` from `Date` instead can
  get a false `304`.
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
