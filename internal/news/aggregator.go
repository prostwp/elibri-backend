package news

import (
	"context"
	"strconv"
	"strings"
	"sync"
	"time"
)

// coinFullName returns the lowercase full coin name for a known ticker
// ("BTC" → "bitcoin"). Returns empty string for unknown.
func coinFullName(slug string) string {
	m := map[string]string{
		"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "BNB": "binance",
		"XRP": "xrp", "DOGE": "dogecoin", "ADA": "cardano", "TRX": "tron",
		"AVAX": "avalanche", "DOT": "polkadot", "LINK": "chainlink", "MATIC": "polygon",
		"SHIB": "shiba", "LTC": "litecoin", "BCH": "bitcoin cash", "UNI": "uniswap",
		"ATOM": "cosmos", "NEAR": "near", "APT": "aptos", "SUI": "sui",
	}
	return m[strings.ToUpper(slug)]
}

// dedupeByURL removes duplicate items that share the same URL (multiple aggregators
// can emit the same article).
func dedupeByURL(items []Item) []Item {
	seen := make(map[string]bool, len(items))
	out := make([]Item, 0, len(items))
	for _, it := range items {
		if it.URL == "" {
			out = append(out, it)
			continue
		}
		if seen[it.URL] {
			continue
		}
		seen[it.URL] = true
		out = append(out, it)
	}
	return out
}

// Item is the unified news entry returned to frontend.
type Item struct {
	Source       string    `json:"source"`       // finnhub | coindesk | cointelegraph | alphavantage
	Category     string    `json:"category"`     // macro | geopolitics | regulation | adoption | crypto
	Headline     string    `json:"headline"`
	Summary      string    `json:"summary,omitempty"`
	URL          string    `json:"url"`
	PublishedAt  time.Time `json:"published_at"`
	Sentiment    float64   `json:"sentiment"`     // -1..+1 (bearish..bullish)
	MentionsCoin bool      `json:"mentions_coin"` // true if headline/summary mentions the queried symbol
}

type Aggregate struct {
	Items            []Item  `json:"items"`
	OverallSentiment float64 `json:"overall_sentiment"` // -1..+1 weighted by recency
	BullishCount     int     `json:"bullish_count"`
	BearishCount     int     `json:"bearish_count"`
	NeutralCount     int     `json:"neutral_count"`
	FetchedAt        time.Time `json:"fetched_at"`
}

// Simple sentiment heuristic from headline keywords (fallback when votes unavailable).
var bullishWords = []string{
	"cut", "ease", "stimulus", "launch", "approve", "adopt", "partnership", "listing", "upgrade",
	"rally", "surge", "record", "breakout", "bullish", "positive", "beat", "outperform", "etf approved",
}
var bearishWords = []string{
	"hike", "tighten", "ban", "hack", "exploit", "lawsuit", "investigation", "crash", "drop",
	"bearish", "miss", "reject", "delay", "decline", "warning", "slump", "plunge", "sanction",
}

func scoreSentiment(text string) float64 {
	t := strings.ToLower(text)
	var b, s int
	for _, w := range bullishWords {
		if strings.Contains(t, w) {
			b++
		}
	}
	for _, w := range bearishWords {
		if strings.Contains(t, w) {
			s++
		}
	}
	total := b + s
	if total == 0 {
		return 0
	}
	return float64(b-s) / float64(total)
}

// ─── In-memory cache (Redis is optional) ──────────────────

type cacheEntry struct {
	data      Aggregate
	expiresAt time.Time
}

var (
	cacheMu  sync.Mutex
	cacheMap = make(map[string]cacheEntry)
)

const cacheTTL = 5 * time.Minute
const cacheMaxEntries = 200

// pruneExpired drops expired entries and, if still over capacity, evicts the
// oldest-expiring ones. Caller must hold cacheMu.
func pruneExpired() {
	now := time.Now()
	for k, e := range cacheMap {
		if now.After(e.expiresAt) {
			delete(cacheMap, k)
		}
	}
	if len(cacheMap) <= cacheMaxEntries {
		return
	}
	// Simple eviction: delete arbitrary entries until under cap. Not LRU, but
	// good enough for a cache with 5-minute TTL — stale entries are already gone.
	excess := len(cacheMap) - cacheMaxEntries
	for k := range cacheMap {
		if excess <= 0 {
			return
		}
		delete(cacheMap, k)
		excess--
	}
}

// ─── Main aggregator ──────────────────────────────────────

// Fetch combines 5 sources in parallel:
// - Finnhub /general (macro + geopolitics)
// - Finnhub /crypto (crypto-specific via same API key)
// - CoinDesk RSS + Cointelegraph RSS (crypto)
// - Alpha Vantage NEWS_SENTIMENT (macro + crypto with pre-labeled sentiment) — only if AV key set
func Fetch(ctx context.Context, finnhubKey, alphaVantageKey, symbol string, hours int) (Aggregate, error) {
	cacheKey := symbol + "|" + strconv.Itoa(hours)
	cacheMu.Lock()
	if entry, ok := cacheMap[cacheKey]; ok && time.Now().Before(entry.expiresAt) {
		cacheMu.Unlock()
		return entry.data, nil
	}
	cacheMu.Unlock()

	var wg sync.WaitGroup
	var fhGeneral, fhCrypto, cdItems, ctItems, avItems []Item

	wg.Add(5)
	go func() {
		defer wg.Done()
		if finnhubKey != "" {
			fhGeneral, _ = FetchFinnhubGeneral(ctx, finnhubKey, hours)
		}
	}()
	go func() {
		defer wg.Done()
		if finnhubKey != "" {
			fhCrypto, _ = FetchFinnhubCrypto(ctx, finnhubKey, hours)
		}
	}()
	go func() {
		defer wg.Done()
		cdItems, _ = FetchRSS(ctx, "https://www.coindesk.com/arc/outboundfeeds/rss?outputType=xml", "coindesk", "crypto", hours)
	}()
	go func() {
		defer wg.Done()
		ctItems, _ = FetchRSS(ctx, "https://cointelegraph.com/rss", "cointelegraph", "crypto", hours)
	}()
	go func() {
		defer wg.Done()
		avItems, _ = FetchAlphaVantage(ctx, alphaVantageKey, hours)
	}()
	wg.Wait()

	// Order matters: dedupeByURL keeps first-seen. We want primary sources
	// (CoinDesk/Cointelegraph/AlphaVantage) attributed directly rather than
	// Finnhub's syndicated copies of the same article.
	all := cdItems
	all = append(all, ctItems...)
	all = append(all, avItems...)
	all = append(all, fhGeneral...)
	all = append(all, fhCrypto...)

	// Mark items that mention the queried coin.
	coinSlug := coinSlugFromSymbol(symbol) // e.g., "BTC"
	if coinSlug != "" {
		slugLower := strings.ToLower(coinSlug)
		for i := range all {
			t := strings.ToLower(all[i].Headline + " " + all[i].Summary)
			if strings.Contains(t, slugLower) ||
				strings.Contains(t, coinFullName(coinSlug)) {
				all[i].MentionsCoin = true
			}
		}
	}

	// Log before/after dedupe for debugging source overlap.
	all = dedupeByURL(all)

	// Sort by PublishedAt desc (most recent first).
	sortByRecency(all)

	// Compute weighted sentiment: recency + direct coin mention boost the weight.
	now := time.Now()
	var weightedSum, weightTotal float64
	var bull, bear, neu int
	for _, it := range all {
		hoursOld := now.Sub(it.PublishedAt).Hours()
		weight := 1.0 / (1.0 + hoursOld/6.0) // 6-hour half-life
		if it.MentionsCoin {
			weight *= 2.0 // direct coin mention doubles importance
		}
		weightedSum += it.Sentiment * weight
		weightTotal += weight
		if it.Sentiment > 0.15 {
			bull++
		} else if it.Sentiment < -0.15 {
			bear++
		} else {
			neu++
		}
	}
	overall := 0.0
	if weightTotal > 0 {
		overall = weightedSum / weightTotal
	}

	agg := Aggregate{
		Items:            all,
		OverallSentiment: overall,
		BullishCount:     bull,
		BearishCount:     bear,
		NeutralCount:     neu,
		FetchedAt:        now,
	}

	cacheMu.Lock()
	cacheMap[cacheKey] = cacheEntry{data: agg, expiresAt: now.Add(cacheTTL)}
	pruneExpired()
	cacheMu.Unlock()

	return agg, nil
}

func sortByRecency(items []Item) {
	// Insertion sort (lists are <50 items, no allocations needed)
	for i := 1; i < len(items); i++ {
		j := i
		for j > 0 && items[j].PublishedAt.After(items[j-1].PublishedAt) {
			items[j], items[j-1] = items[j-1], items[j]
			j--
		}
	}
}

// coinSlugFromSymbol: "BTCUSDT" → "BTC", "ETHUSDT" → "ETH", "" → ""
func coinSlugFromSymbol(sym string) string {
	if sym == "" {
		return ""
	}
	for _, suffix := range []string{"USDT", "USD", "BUSD", "USDC"} {
		if strings.HasSuffix(sym, suffix) {
			return strings.TrimSuffix(sym, suffix)
		}
	}
	return sym
}

