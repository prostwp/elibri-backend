# News Aggregator

6 parallel sources feed the Fundamental analysis node:

| Source | API key | Coverage | Category |
|--------|---------|----------|----------|
| Finnhub `/general` | `FINNHUB_API_KEY` | Macro, geopolitics, general | macro · geopolitics · regulation |
| Finnhub `/crypto` | same | Crypto-specific news | crypto |
| CoinDesk RSS | — (free) | Crypto news | crypto |
| Cointelegraph RSS | — (free) | Crypto news | crypto |
| Alpha Vantage `NEWS_SENTIMENT` | `ALPHA_VANTAGE_API_KEY` | Pre-labeled sentiment, macro + crypto | macro · crypto |
| **LunarCrush `/coins/:symbol/posts`** | `LUNARCRUSH_API_KEY` | **Verified KOL Twitter/Reddit posts** | **social** |

## LunarCrush setup

1. Sign up at https://lunarcrush.com/developers
2. Free tier: 200 req/min, 10k/day (sufficient for our cache-backed aggregator)
3. Copy API key and set `LUNARCRUSH_API_KEY` env var before starting backend
4. Posts appear in `CryptoFundamental` node under the **Social** filter chip
5. Posts from accounts with ≥10k followers AND ≥100 interactions are flagged
   as **KOL** (key opinion leader) in the UI with ⚡ badge

Without the key, the aggregator silently skips LunarCrush — no errors.

## Sentiment mapping (LunarCrush → our scale)

LunarCrush returns 1..5 per post (1=very bear, 3=neutral, 5=very bull).
We map to [-1, +1] via `(score - 3) / 2`, then clamp.

## Dedupe strategy

`dedupeByURL` keeps first-seen. Source order in aggregator prioritizes
original sources (CoinDesk, Cointelegraph, AlphaVantage, LunarCrush) over
Finnhub which may syndicate the same articles — so attribution stays honest.
