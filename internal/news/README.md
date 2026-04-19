# News Aggregator

7 parallel sources feed the Fundamental analysis node. Each source gracefully
skips (returns empty) when its credential is missing or its access fails —
aggregator never errors out on a single source.

| Source | API key | Status | Coverage | Category |
|--------|---------|--------|----------|----------|
| Finnhub `/general` | `FINNHUB_API_KEY` | ✅ works | Macro, geopolitics | macro · geopolitics · regulation |
| Finnhub `/crypto` | same | ✅ works | Crypto-specific news | crypto |
| CoinDesk RSS | — | ✅ free | Crypto news | crypto |
| Cointelegraph RSS | — | ✅ free | Crypto news | crypto |
| Alpha Vantage `NEWS_SENTIMENT` | `ALPHA_VANTAGE_API_KEY` | ✅ free key | Pre-labeled sentiment | macro · crypto |
| **LunarCrush posts** | `LUNARCRUSH_API_KEY` | ⚠ paid since Nov 2025 | KOL Twitter/Reddit | social |
| **Reddit hot** | — | ⚠ 403 without OAuth since Jul 2023 | Reddit r/{coin} | social |

## Social sources — the reality in 2026

The "crypto Twitter insider" feature ran into 2026 API tightening:

- **LunarCrush** removed free tier in November 2025. Their `posts` endpoints
  now require **Individual subscription ($29/mo)**. Scaffold is in place —
  subscribe and set `LUNARCRUSH_API_KEY` to enable.
- **Reddit** blocks unauthenticated JSON requests from server IPs with HTTP 403
  (since July 2023). To use Reddit, add OAuth2 client-credentials flow
  (register app at reddit.com/prefs/apps, free). Scaffold is in place in
  `reddit.go` — TODO add `GetAccessToken()` helper.
- **CryptoPanic** free tier discontinued April 2026.

Without paid/OAuth social sources, the aggregator still provides ~50 news
items per request from CoinDesk + Cointelegraph + Finnhub, which covers the
main macro/crypto/regulation sentiment signal. The **social** category will
just be empty until one of the social sources is enabled.

## Sentiment mapping

- CoinDesk/Cointelegraph/Finnhub: keyword-based heuristic (bullish/bearish word lists)
- Alpha Vantage: pre-labeled sentiment score from their ML
- LunarCrush: 1..5 → [-1, +1] via `(score - 3) / 2`
- Reddit: keyword-based on title + body

## Sentiment mapping (LunarCrush → our scale)

LunarCrush returns 1..5 per post (1=very bear, 3=neutral, 5=very bull).
We map to [-1, +1] via `(score - 3) / 2`, then clamp.

## Dedupe strategy

`dedupeByURL` keeps first-seen. Source order in aggregator prioritizes
original sources (CoinDesk, Cointelegraph, AlphaVantage, LunarCrush) over
Finnhub which may syndicate the same articles — so attribution stays honest.
