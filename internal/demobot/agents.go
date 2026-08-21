package demobot

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// Agents builds one Card per command from live data. It never invents
// numbers: any source failure degrades to an honest offline card (or an
// honest per-fact note when only part of the data is missing).
type Agents struct {
	api    *BackendClient
	klines *klineCache
	ai     *aiClient // nil until EnableAI — every AI block silently omitted
}

func NewAgents(api *BackendClient) *Agents {
	return &Agents{api: api, klines: newKlineCache()}
}

const (
	klineInterval = "4h"
	// 250 raw bars → 249 CLOSED bars after the forming bar is dropped,
	// keeping EMA200 above its minimum-history requirement.
	klineLimit = 250
)

// errInsufficientHistory marks a fetch that succeeded but returned too few
// closed bars for the requested indicator set.
var errInsufficientHistory = errors.New("insufficient history")

var fundingSymbols = []string{"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"}

// candlesFor dispatches to the right candle source for an asset and returns
// CLOSED bars only — the still-forming last bar is dropped after the cache
// (item 7: signals must not flicker mid-bar). Yahoo symbols go through the
// same 60s cache as Binance ones; the gold fallback (GC=F) kicks in when the
// primary Yahoo symbol fails or comes back empty.
func (a *Agents) candlesFor(ctx context.Context, spec assetSpec) ([]types.OHLCVCandle, error) {
	var candles []types.OHLCVCandle
	var err error
	if spec.Source == srcYahoo {
		// The cache key carries the interval (B1): a 4h-rebased EURUSD must
		// never collide with the native 1h series.
		key := "yahoo|" + spec.Symbol + "|" + spec.Interval
		candles, err = a.klines.cached(key, func() ([]types.OHLCVCandle, error) {
			c, ferr := fetchYahooCandlesTF(ctx, spec.Symbol, spec.Interval)
			if (ferr != nil || len(c) == 0) && spec.Fallback != "" {
				return fetchYahooCandlesTF(ctx, spec.Fallback, spec.Interval)
			}
			return c, ferr
		})
	} else {
		candles, err = a.klines.fetch(ctx, spec.Symbol, spec.Interval, klineLimit)
	}
	if err != nil {
		return nil, err
	}
	return dropUnclosedBars(candles, spec.Interval, time.Now()), nil
}

// sourceCard tailors the degraded/asset presentation per source: FX cards
// carry the Yahoo footer note and the FX-specific offline wording.
func assetOffline(spec assetSpec, agent, shortName, command, how string) Card {
	c := offlineCard(agent, shortName, spec.Display, command, how)
	if spec.Source == srcYahoo {
		c.Verdict = fxOfflineVerdict
		c.SourceNote = "data: Yahoo Finance"
	}
	return c
}

// decorateFX applies the shared FX card furniture: source note and the
// weekend banner as the first fact. DataTime is the caller's job — it must
// be the close time of the last closed bar actually used (item 7).
func decorateFX(c *Card) {
	c.SourceNote = "data: Yahoo Finance"
	if !isForexOpen(time.Now()) {
		c.Facts = append([]string{fxClosedBanner}, c.Facts...)
	}
}

// insufficientCard is the honest short-history state: neutral semaphore,
// explicit text, no verdict, no confidence — and excluded from the priority
// rule (Offline flag) because "not enough data" is not a signal.
func insufficientCard(spec assetSpec, agent, shortName, command, how, what string) Card {
	c := Card{
		Emoji:      emojiNeutral,
		Agent:      agent,
		ShortName:  shortName,
		Asset:      spec.Display,
		Command:    command,
		HowItWorks: how,
		DataTime:   time.Now().UTC(),
		Verdict:    "Insufficient history — no verdict",
		Short:      "insufficient history",
		Offline:    true,
		Status:     statusInsufficientHistory,
	}
	c.Facts = []string{"insufficient history for " + what}
	if spec.Source == srcYahoo {
		c.SourceNote = "data: Yahoo Finance"
	}
	return c
}

// howTexts back the [ℹ️ How it works] button. Telegram caps callback alerts
// at 200 characters — keep every entry under that.
var howTexts = map[string]string{
	keyMacro:    "Reads 5 tradfin lamps (DXY, US 10Y, VIX, S&P 500, Gold) plus crypto Fear & Greed and blends them into a risk-on/off regime. RISK-OFF outranks every other signal in /digest.",
	keyWhale:    "Watches large on-chain transfers touching known exchange wallets. Net inflow to exchanges = potential sell pressure; net outflow = accumulation.",
	keyFunding:  "Compares perp funding rates across majors. High positive funding = crowded longs (squeeze risk); negative = crowded shorts. Liquidation feed shows where forced exits cluster.",
	keyMomentum: "RSI(14) + MACD histogram. RSI 55+ with positive MACD = bullish; RSI 45- with negative = bearish; else neutral. Default 4h crypto / 1h FX; custom scan: /momentum btc,eurusd 1d.",
	keyTrend:    "State machine on 4h candles: ADX<20 = flat (no trade), 20-25 = grey zone, ADX 25+ with price and EMA50/EMA200 aligned = confirmed trend; misaligned = conflict.",
	keySR:       "Finds swing highs/lows on 4h candles and clusters levels within 0.5%. Strength = touches + 0.5 per above-median-volume touch; held/break counts show how levels behaved when tested.",
	keyVol:      "ATR(14) now vs its 30-bar average. Ratio 1.25+ = volatility expanding (breakout regime); 0.8- = compressed (range regime).",
	keyRisk:     "Position size = (balance × risk%) ÷ |entry − stop|. Keeps one losing trade at a fixed fraction of the account. Works for any asset.",
	keyFX:       "EMA50 vs EMA200 direction, RSI(14) and 24h change on 1h Yahoo Finance data for EURUSD, GBPUSD, USDJPY, XAUUSD. Weekend closures (Fri 21:00–Sun 21:00 UTC) are flagged, never hidden.",
	keyDigest:   "Deterministic priority: RISK-OFF macro always tops; otherwise the strongest deviation from neutral among funding, momentum, trend. Ties break funding > momentum > trend.",
	keyTop:      "Deterministic priority: RISK-OFF macro always tops; otherwise the strongest deviation from neutral among funding, momentum, trend. Ties break funding > momentum > trend.",
	keyNews:     "Crypto themes ranked by 48h mention growth across news/Reddit. Stage: early/trending/mainstream/declining; trend score 0-100. The top narrative carries an AI-generated observation.",
}

// ── Macro ────────────────────────────────────────────────────────────────────

// MacroCard returns the card plus the effective regime for the priority rule
// ("" when the source is offline).
//
// Honesty contract (team-testing defect 2026-08): a regime verdict is a
// knowledge claim, so it needs at least one real lamp behind it. Regime
// "unknown" (or an older backend's "mixed" with zero real lamps — reclassified
// here) renders an explicit no-data card: UNKNOWN verdict, neutral semaphore,
// facts limited to what IS known (crypto F&G), and never the "signals are
// split" idea line.
func (a *Agents) MacroCard(ctx context.Context) (Card, string) {
	m, err := a.api.Macro(ctx)
	if err != nil {
		return offlineCard("Macro Agent", "Macro", "", keyMacro, howTexts[keyMacro]), ""
	}
	c := Card{
		Agent:      "Macro Agent",
		ShortName:  "Macro",
		Command:    keyMacro,
		HowItWorks: howTexts[keyMacro],
		DataTime:   parseWhen(m.CapturedAt),
	}

	// A lamp is REAL when the payload carries a value for it; zero real lamps
	// reclassify an old backend's "mixed" to unknown (a MIXED claim needs at
	// least one input). Shared with the asset views — see effectiveMacroRegime.
	regime, real := effectiveMacroRegime(m)

	switch regime {
	case "risk_on":
		c.Emoji, c.Verdict, c.Short = emojiBull, "RISK-ON — big money leaning into risk", "risk-on"
	case "risk_off":
		c.Emoji, c.Verdict, c.Short = emojiBear, "RISK-OFF — big money defensive", "risk-off"
	case "unknown":
		// No tradfin inputs at all — say so instead of claiming a regime read.
		// Machine status splits the two absences templates must distinguish:
		// closed window = market_closed, open window with a dark feed = no_data.
		verdict := "UNKNOWN — market closed, no tradfin data"
		c.Status = statusMarketClosed
		if m.TradfinOpen {
			verdict = "UNKNOWN — no tradfin data right now"
			c.Status = statusNoData
		}
		c.Emoji, c.Verdict, c.Short = emojiNeutral, verdict, "unknown (no data)"
	default:
		c.Emoji, c.Verdict, c.Short = emojiNeutral, "MIXED — no single regime in control", "mixed"
	}

	var tail, head, neut int
	for _, l := range m.Lamps {
		switch l.Status {
		case "tailwind":
			tail++
		case "headwind":
			head++
		case "neutral":
			neut++
		}
	}
	if real == 0 && len(m.Lamps) > 0 {
		// Zero real lamps → the counts line would be a fake "0/0/0 reading";
		// state the absence instead (with the clock context).
		note := "no tradfin data right now"
		if !m.TradfinOpen {
			note += " (market closed)"
		}
		c.Facts = append(c.Facts, "Lamps: "+note)
	} else {
		c.Facts = append(c.Facts, fmt.Sprintf("Lamps: %d tailwind / %d headwind / %d neutral", tail, head, neut))
	}
	// Signal map (B2): the same lamps read for both assets, one line each —
	// only with at least one real lamp behind them (an asset verdict is a
	// knowledge claim like any other).
	if real > 0 {
		c.Facts = append(c.Facts, macroViewLines(regime, m.Lamps)...)
	}
	// What IS known even when tradfin is dark: the crypto side (F&G is 24/7),
	// rendered only when its own ok flag says the read is live.
	if m.FNG != nil && m.FNG.OK {
		c.Facts = append(c.Facts, fmt.Sprintf("Crypto Fear & Greed: %d — %s", m.FNG.Value, m.FNG.Label))
	}
	// The generated idea (e.g. "Macro signals are split…") is a claim about
	// lamp data — it must NEVER render without at least one real lamp. The
	// backend already blanks it for unknown; this guard also covers version
	// skew.
	if idea := strings.TrimSpace(m.GeneratedIdea); idea != "" && real > 0 {
		c.Facts = append(c.Facts, truncate(idea, 180))
	}
	if m.Composite != nil {
		c.Confidence = m.Composite
		c.Deviation = clampInt(abs(*m.Composite-50)*2, 0, 100)
	}
	// ALPHAVIZOR AI mood read (backend Haiku, 10-min server cache) — appended
	// when available, silently omitted otherwise. Never blocks the card.
	if mood, err := a.api.MoodRead(ctx); err == nil {
		if txt := strings.TrimSpace(mood.Read); txt != "" {
			c.AIHTML = "<b>AI read:</b> <i>" + esc(truncateAtSentence(txt, 400)) + "</i>"
		}
	}
	return c, regime
}

// ── Whale flow ───────────────────────────────────────────────────────────────

func (a *Agents) WhaleCard(ctx context.Context) Card {
	w, err := a.api.WhaleFlow(ctx, 10)
	if err != nil {
		return offlineCard("Whale Flow Agent", "Whale", "BTC", keyWhale, howTexts[keyWhale])
	}
	c := Card{
		Agent:      "Whale Flow Agent",
		ShortName:  "Whale",
		Asset:      "BTC",
		Command:    keyWhale,
		HowItWorks: howTexts[keyWhale],
		DataTime:   parseWhen(w.CapturedAt),
	}

	var btc *WhaleFlow
	for i := range w.Flows {
		if w.Flows[i].Asset == "BTC" {
			btc = &w.Flows[i]
			break
		}
	}
	switch {
	case btc == nil:
		c.Emoji, c.Verdict, c.Short = emojiNeutral, "No BTC flow snapshot yet", "no data"
		c.Status = statusNoData // upstream alive, nothing to read yet
	case btc.Direction == "inflow":
		c.Emoji, c.Verdict, c.Short = emojiBear, "Net INFLOW to exchanges — potential sell pressure", "inflow (sell pressure)"
	case btc.Direction == "outflow":
		c.Emoji, c.Verdict, c.Short = emojiBull, "Net OUTFLOW from exchanges — accumulation read", "outflow (accumulation)"
	default:
		c.Emoji, c.Verdict, c.Short = emojiNeutral, "Flows balanced over 24h", "balanced"
	}
	if btc != nil {
		flowLine := fmt.Sprintf("Net flow 24h: %s (%d large tx)", usd(btc.NetFlowUSD24h), btc.TxCount24h)
		if btc.Partial {
			flowLine += " — partial data, labeled wallets only"
		}
		c.Facts = append(c.Facts, flowLine)
		// Baseline comparison — only when the payload actually carries a
		// prior-24h figure (a zero baseline means "no snapshot to compare").
		if btc.NetFlowPrev24h != nil && *btc.NetFlowPrev24h != 0 {
			base := fmt.Sprintf("Prior 24h net flow: %s", usd(*btc.NetFlowPrev24h))
			if btc.FlowPct != nil {
				base += fmt.Sprintf(" → %+.0f%% change", *btc.FlowPct)
			}
			c.Facts = append(c.Facts, base)
		}
		if btc.Confidence > 0 {
			conf := btc.Confidence
			c.Confidence = &conf
		}
	}

	// Top-3 BTC transfers of the last 24h by USD size.
	cutoff := time.Now().Add(-24 * time.Hour)
	var recent []WhaleTransfer
	for _, t := range w.Transfers {
		if t.Chain == "BTC" && t.Timestamp.After(cutoff) {
			recent = append(recent, t)
		}
	}
	sort.Slice(recent, func(i, j int) bool { return recent[i].AmountUSD > recent[j].AmountUSD })
	if len(recent) == 0 {
		c.Facts = append(c.Facts, "No large BTC transfers in the last 24h")
	}
	for i, t := range recent {
		if i == 3 {
			break
		}
		where := t.Exchange
		if where == "" {
			where = "unlabeled wallet"
		}
		c.Facts = append(c.Facts, fmt.Sprintf("%s BTC ≈ %s · %s · %s",
			trimFloat(t.AmountNative), usd(t.AmountUSD), t.Direction, where))
	}
	return c
}

// ── Narrative Radar (/news) ──────────────────────────────────────────────────

// newsMinMentions is the radar's silence threshold: below this many 24h
// mentions for the TOP narrative there is too little signal to score at all —
// a trend score computed off a handful of posts is noise dressed as a
// finding. Under the threshold the card says "warming up" and lists themes as
// name + mention count only (no scores, no stages, no confidence, no AI
// idea). HTTP /agents/news mirrors this automatically via the shared card.
const newsMinMentions = 5

// NewsCard renders the top-3 crypto narratives from the backend radar (48h
// mention window) plus the backend's AI-generated idea for the leading one.
// The server sorts by trend_score DESC and attaches generated_idea to the
// top narrative only.
func (a *Agents) NewsCard(ctx context.Context) Card {
	n, err := a.api.Narratives(ctx)
	if err != nil {
		return offlineCard("Narrative Radar", "Narrative", "", keyNews, howTexts[keyNews])
	}
	c := Card{
		Agent:      "Narrative Radar",
		ShortName:  "Narrative",
		Command:    keyNews,
		HowItWorks: howTexts[keyNews],
		DataTime:   parseWhen(n.CapturedAt),
		SourceNote: "48h mention window",
	}
	if len(n.Narratives) == 0 {
		c.Emoji = emojiNeutral
		c.Verdict = "No narrative snapshots yet — radar warming up"
		c.Short = "no data"
		c.Offline = true // "not enough data" is not a signal
		c.Status = statusBelowThreshold
		return c
	}
	top := n.Narratives[0]
	// Silence threshold: a thin mention base cannot back a scored finding.
	// Present names + mention counts only, and say why there is no verdict.
	if top.MentionCount < newsMinMentions {
		c.Emoji = emojiNeutral
		c.Status = statusBelowThreshold
		mentions := "mentions"
		if top.MentionCount == 1 {
			mentions = "mention"
		}
		c.Verdict = fmt.Sprintf("Radar warming up — top theme '%s' has only %d %s in 24h; not enough to score",
			top.Narrative, top.MentionCount, mentions)
		c.Short = "warming up"
		for i, item := range n.Narratives {
			if i == 3 {
				break
			}
			c.Facts = append(c.Facts, fmt.Sprintf("%d. %s — %d mentions/24h", i+1, item.Narrative, item.MentionCount))
		}
		// No confidence, no AI idea: nothing below the threshold is a finding.
		return c
	}
	switch top.SentimentLabel {
	case "bull":
		c.Emoji = emojiBull
	case "bear":
		c.Emoji = emojiBear
	default:
		c.Emoji = emojiNeutral
	}
	c.Verdict = fmt.Sprintf("Top narrative: %s — %s, trend score %d/100", top.Narrative, top.Stage, top.TrendScore)
	c.Short = fmt.Sprintf("%s (%s)", top.Narrative, top.Stage)
	if top.Confidence > 0 {
		conf := top.Confidence
		c.Confidence = &conf
	}
	for i, item := range n.Narratives {
		if i == 3 {
			break
		}
		line := fmt.Sprintf("%d. %s — %s · score %d · %d mentions/24h", i+1, item.Narrative, item.Stage, item.TrendScore, item.MentionCount)
		if item.SentimentLabel != "" {
			line += " · " + item.SentimentLabel
		}
		c.Facts = append(c.Facts, line)
	}
	for _, item := range n.Narratives {
		if idea := strings.TrimSpace(item.GeneratedIdea); idea != "" {
			c.AIHTML = "<b>AI idea:</b> <i>" + esc(truncateAtSentence(idea, 500)) + "</i>"
			break
		}
	}
	return c
}

// ── Funding ──────────────────────────────────────────────────────────────────

// Funding-rate verdict thresholds (8h rate, absolute):
// +0.03% and above = longs crowded; -0.01% and below = shorts crowded
// (negative funding is rarer, so its threshold is tighter). In between =
// balanced. Deviation scales |widest| against 0.10%/8h == 100.
const (
	fundingLongsCrowded  = 0.0003
	fundingShortsCrowded = -0.0001
	fundingDevFullScale  = 0.0010
)

func (a *Agents) FundingCard(ctx context.Context) Card {
	rates, ratesErr := fetchFundingRates(ctx, fundingSymbols)
	liq, liqErr := a.api.FundingLiquidations(ctx)
	if ratesErr != nil && liqErr != nil {
		return offlineCard("Funding Agent", "Funding", "", keyFunding, howTexts[keyFunding])
	}
	c := Card{
		Agent:      "Funding Agent",
		ShortName:  "Funding",
		Command:    keyFunding,
		HowItWorks: howTexts[keyFunding],
		// Funding IS a point-in-time read — now() is honest here, and the
		// footer labels it so (item 7 exempts funding but requires the label).
		DataTime:   time.Now().UTC(),
		SourceNote: "as of request time",
	}

	if ratesErr == nil {
		widestSym, widest := "", 0.0
		for sym, r := range rates {
			if math.Abs(r) > math.Abs(widest) || widestSym == "" {
				widestSym, widest = sym, r
			}
		}
		switch {
		case widest >= fundingLongsCrowded:
			c.Emoji, c.Verdict, c.Short = emojiBear, "Longs crowded — squeeze risk building", "longs crowded"
		case widest <= fundingShortsCrowded:
			c.Emoji, c.Verdict, c.Short = emojiBull, "Shorts crowded — squeeze fuel above", "shorts crowded"
		default:
			c.Emoji, c.Verdict, c.Short = emojiNeutral, "Funding balanced — no crowd to punish", "balanced"
		}
		side := "longs pay shorts"
		if widest < 0 {
			side = "shorts pay longs"
		}
		c.Facts = append(c.Facts, fmt.Sprintf("Widest skew: %s %+.4f%%/8h (%s)", widestSym, widest*100, side))
		if btc, ok := rates["BTCUSDT"]; ok && widestSym != "BTCUSDT" {
			c.Facts = append(c.Facts, fmt.Sprintf("BTC funding: %+.4f%%/8h", btc*100))
		}
		c.Deviation = clampInt(int(math.Round(math.Abs(widest)/fundingDevFullScale*100)), 0, 100)
	} else {
		c.Emoji, c.Verdict, c.Short = emojiNeutral, "Funding rates unavailable — liquidations only", "rates offline"
		// The agent's headline reading (funding skew) was not produced — the
		// envelope must say ok=false/source_offline even though the card still
		// renders the liquidation facts as a 200.
		c.Status = statusSourceOffline
		c.Facts = append(c.Facts, "Funding-rate source offline right now")
	}

	switch {
	case liqErr != nil:
		c.Facts = append(c.Facts, "Liquidation feed offline right now")
	case len(liq.Feed) == 0:
		c.Facts = append(c.Facts, "Liquidation feed live but quiet — no forced exits recently")
	default:
		var longUSD, shortUSD float64
		cutoff := time.Now().Add(-1 * time.Hour)
		for _, l := range liq.Feed {
			if l.TS.Before(cutoff) {
				continue
			}
			if l.Side == "long_liq" {
				longUSD += l.USDValue
			} else {
				shortUSD += l.USDValue
			}
		}
		line := fmt.Sprintf("Liquidations 1h: %s longs vs %s shorts", usd(longUSD), usd(shortUSD))
		if skew := liqSkew(longUSD, shortUSD); skew != "" {
			line += " — " + skew
		}
		c.Facts = append(c.Facts, line)
		if len(liq.Zones) > 0 {
			z := liq.Zones[0]
			// Prefer the BTC zone nearest to the live BTC price when the
			// kline cache can supply one; otherwise keep the served order.
			if btcPrice := a.lastBTCClose(ctx); btcPrice > 0 {
				z = nearestZone(liq.Zones, "BTCUSDT", btcPrice)
			}
			c.Facts = append(c.Facts, fmt.Sprintf("Magnet zone: %s %s (%s, %d hits)", z.Symbol, z.PriceBand, usd(z.TotalUSD), z.Count))
		}
	}
	return c
}

// liqSkew words which side the 1h liquidation flow is punishing. "" when
// the window saw no volume.
func liqSkew(longUSD, shortUSD float64) string {
	total := longUSD + shortUSD
	if total <= 0 {
		return ""
	}
	longShare := longUSD / total
	switch {
	case longShare >= 0.65:
		return fmt.Sprintf("longs taking %d%% of the pain", int(math.Round(longShare*100)))
	case longShare <= 0.35:
		return fmt.Sprintf("shorts taking %d%% of the pain", int(math.Round((1-longShare)*100)))
	default:
		return "both sides roughly balanced"
	}
}

// bandMid parses a "118200-118250" price band into its midpoint.
func bandMid(band string) (float64, bool) {
	parts := strings.Split(band, "-")
	if len(parts) != 2 {
		return 0, false
	}
	lo, err1 := strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
	hi, err2 := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
	if err1 != nil || err2 != nil {
		return 0, false
	}
	return (lo + hi) / 2, true
}

// nearestZone picks the magnet zone for `symbol` whose band midpoint sits
// closest to price. Falls back to the first served zone when price is
// unknown or no band parses — exactly the pre-depth behavior.
func nearestZone(zones []LiqZone, symbol string, price float64) LiqZone {
	if price <= 0 {
		return zones[0]
	}
	best := -1
	bestDist := math.MaxFloat64
	for i, z := range zones {
		if z.Symbol != symbol {
			continue
		}
		mid, ok := bandMid(z.PriceBand)
		if !ok {
			continue
		}
		if d := math.Abs(mid - price); d < bestDist {
			bestDist, best = d, i
		}
	}
	if best < 0 {
		return zones[0]
	}
	return zones[best]
}

// lastBTCClose returns the last closed BTC 4h close from the kline cache,
// 0 when unavailable — callers must treat 0 as "price unknown".
func (a *Agents) lastBTCClose(ctx context.Context) float64 {
	candles, err := a.candlesFor(ctx, btcSpec)
	if err != nil || len(candles) == 0 {
		return 0
	}
	return candles[len(candles)-1].Close
}

// ── Momentum ─────────────────────────────────────────────────────────────────

// Momentum verdict rule (documented, deterministic): RSI≥55 with positive
// MACD histogram = bullish; RSI≤45 with negative histogram = bearish; else
// neutral.
//
// REGULATORY LANGUAGE (team review batch 2): verdicts are analytical READINGS,
// never trade instructions — "bullish"/"bearish", NOT "buy"/"sell". The
// product must read as analytics; advice-words are banned from every card,
// one-liner and envelope. Factual market-mechanics wording ("longs pay
// shorts", "crowded longs", "sell pressure" as a flow description) stays.
func momentumVerdict(rsi, macdHist float64) string {
	switch {
	case rsi >= 55 && macdHist > 0:
		return "bullish"
	case rsi <= 45 && macdHist < 0:
		return "bearish"
	default:
		return "neutral"
	}
}

// momentumRead computes one asset's RSI/MACD snapshot.
type momentumRead struct {
	name    string
	verdict string
	rsi     float64
	hist    float64
	source  string
	closeAt time.Time // close time of the last closed bar used
}

// momentumReadFromCandles computes one asset's snapshot from an already
// fetched series, so a read and any derived facts (volume line) always come
// from the SAME bars — a transient refetch can't produce a half-coherent card.
func momentumReadFromCandles(spec assetSpec, candles []types.OHLCVCandle) (momentumRead, error) {
	closes := closesOf(candles)
	rsi, okRSI := rsiWilder(closes, 14)
	_, _, hist, okMACD := macdLast(closes)
	if !okRSI || !okMACD {
		return momentumRead{name: spec.Display, source: spec.Source}, errInsufficientHistory
	}
	return momentumRead{
		name:    spec.Display,
		verdict: momentumVerdict(rsi, hist),
		rsi:     rsi,
		hist:    hist,
		source:  spec.Source,
		closeAt: closeTimeOf(candles, spec.Interval),
	}, nil
}

func (a *Agents) momentumReadFor(ctx context.Context, spec assetSpec) (momentumRead, error) {
	candles, err := a.candlesFor(ctx, spec)
	if err != nil {
		return momentumRead{}, err
	}
	return momentumReadFromCandles(spec, candles)
}

func momentumEmoji(verdict string) string {
	switch verdict {
	case "bullish":
		return emojiBull
	case "bearish":
		return emojiBear
	default:
		return emojiNeutral
	}
}

// assetResult builds one momentum results entry from a card status.
func assetResult(display string, st cardStatus) AssetResult {
	res := AssetResult{Asset: display, OK: st == statusOK}
	if !res.OK {
		r := st.reason()
		res.Reason = &r
	}
	return res
}

// MomentumCard is the default multi-asset card: BTC + ETH (Binance 4h) and
// XAUUSD (Yahoo 1h, GC=F fallback).
func (a *Agents) MomentumCard(ctx context.Context) Card {
	c := Card{
		Agent:      "Momentum Agent",
		ShortName:  "Momentum",
		Asset:      "BTC/ETH/XAUUSD",
		Command:    keyMomentum,
		HowItWorks: howTexts[keyMomentum],
		DataTime:   time.Now().UTC(), // narrowed below to the OLDEST closed bar used
	}
	var reads []momentumRead
	var insufficientLines []string
	var assetResults []AssetResult
	var btcCandles []types.OHLCVCandle // the exact series the BTC read used
	for _, key := range []string{"btc", "eth"} {
		spec := assetTable[key]
		candles, err := a.candlesFor(ctx, spec)
		if err != nil {
			// Hard fetch failure — no fact line (pre-existing contract), but
			// the machine results array states it (review fix 3).
			assetResults = append(assetResults, assetResult(spec.Display, statusSourceOffline))
			continue
		}
		if key == "btc" {
			btcCandles = candles
		}
		r, rerr := momentumReadFromCandles(spec, candles)
		switch {
		case rerr == nil:
			reads = append(reads, r)
			assetResults = append(assetResults, assetResult(spec.Display, statusOK))
		case errors.Is(rerr, errInsufficientHistory):
			insufficientLines = append(insufficientLines, r.name+": insufficient history for RSI/MACD")
			assetResults = append(assetResults, assetResult(spec.Display, statusInsufficientHistory))
		}
	}
	xau, xauErr := a.momentumReadFor(ctx, xauSpec)
	switch {
	case xauErr == nil:
		assetResults = append(assetResults, assetResult(xauSpec.Display, statusOK))
	case errors.Is(xauErr, errInsufficientHistory):
		assetResults = append(assetResults, assetResult(xauSpec.Display, statusInsufficientHistory))
	default:
		assetResults = append(assetResults, assetResult(xauSpec.Display, statusSourceOffline))
	}
	c.Results = assetResults
	if len(reads) == 0 && xauErr != nil {
		// Zero real readings — same reason split as the scan card (review
		// fix 3): any answered-but-short source → insufficient_history; only
		// an all-dead sweep is source_offline.
		if len(insufficientLines) > 0 || errors.Is(xauErr, errInsufficientHistory) {
			c.Emoji = emojiNeutral
			c.Verdict = "Insufficient history — no verdict"
			c.Short = "insufficient history"
			c.Offline = true
			c.Status = statusInsufficientHistory
			c.Facts = append(c.Facts, insufficientLines...)
			return c
		}
		off := offlineCard("Momentum Agent", "Momentum", "BTC/ETH/XAUUSD", keyMomentum, howTexts[keyMomentum])
		off.Results = assetResults
		return off
	}

	var verdictParts []string
	maxDev := 0
	all := reads
	if xauErr == nil {
		all = append(all, xau)
	}
	fxOpen := isForexOpen(time.Now())
	for _, r := range all {
		verdictParts = append(verdictParts, fmt.Sprintf("%s: %s", r.name, strings.ToUpper(r.verdict)))
		// %g: gold's tiny histogram must not render as a misleading "+0.0".
		line := fmt.Sprintf("%s: RSI(14) %.1f · MACD hist %+.3g → %s", r.name, r.rsi, r.hist, r.verdict)
		if r.source == srcYahoo && !fxOpen {
			line += " (market closed)"
		}
		c.Facts = append(c.Facts, line)
		// Footer time = OLDEST closed bar used across assets: "every number
		// on this card is at least this fresh" (item 7).
		if !r.closeAt.IsZero() && r.closeAt.Before(c.DataTime) {
			c.DataTime = r.closeAt
		}
		// Deviation drives the /digest priority rule and is CRYPTO-ONLY in
		// v1 — FX reads never push momentum to the top slot (see priority.go).
		// No Confidence bar either: the card contract shows one only where an
		// API supplies confidence, and this read is computed locally.
		if r.source == srcBinance {
			if d := int(math.Abs(r.rsi-50) * 2); d > maxDev {
				maxDev = clampInt(d, 0, 100)
			}
		}
	}
	c.Facts = append(c.Facts, insufficientLines...)
	// The driving thresholds, once for all assets (Volatility-model style: the
	// number lines stay compact, one rule line documents what flips them).
	if len(all) > 0 {
		c.Facts = append(c.Facts, "Rule: RSI 55+/45- with matching MACD sign")
	}
	switch {
	case xauErr == nil:
		c.SourceNote = "XAUUSD data: Yahoo Finance"
	case errors.Is(xauErr, errInsufficientHistory):
		c.Facts = append(c.Facts, "XAUUSD: insufficient history for RSI/MACD")
	default:
		c.Facts = append(c.Facts, "XAUUSD: "+fxOfflineVerdict)
	}
	c.Verdict = strings.Join(verdictParts, " · ")
	lead := "neutral"
	if len(all) > 0 {
		lead = all[0].verdict // BTC leads the semaphore when present
	}
	c.Emoji = momentumEmoji(lead)
	c.Short = strings.ToLower(lead)
	c.Deviation = maxDev

	// Volume read from the SAME series the BTC read used — no refetch, so
	// the volume line can never contradict a missing BTC line.
	if ratio, ok := volRatio20(btcCandles); ok {
		c.Facts = append(c.Facts, fmt.Sprintf("BTC 4h volume: %.2f× its 20-bar average", ratio))
	}

	// Relative strength vs BTC from the backend (secondary, best-effort).
	if rs, err := a.api.MomentumRS(ctx, []string{"ETH"}); err == nil {
		if item, ok := rs.Items["ETH"]; ok {
			var parts []string
			if item.RS7D != nil {
				parts = append(parts, fmt.Sprintf("7d %+.1f%%", *item.RS7D))
			}
			if item.RS30D != nil {
				parts = append(parts, fmt.Sprintf("30d %+.1f%%", *item.RS30D))
			}
			if len(parts) > 0 {
				c.Facts = append(c.Facts, "ETH vs BTC relative strength: "+strings.Join(parts, " · "))
			}
		}
	}
	return c
}

// volRatio20 compares the last closed bar's volume to the mean of the 20
// bars before it. ok=false under 21 bars or on a dead baseline — a short or
// zero series is stated by omission, never rendered as a fake ratio.
func volRatio20(candles []types.OHLCVCandle) (float64, bool) {
	n := len(candles)
	if n < 21 {
		return 0, false
	}
	var sum float64
	for _, c := range candles[n-21 : n-1] {
		sum += c.Volume
	}
	avg := sum / 20
	if avg <= 0 {
		return 0, false
	}
	return candles[n-1].Volume / avg, true
}

// MomentumAssetCard is the single-asset form: /momentum eurusd.
func (a *Agents) MomentumAssetCard(ctx context.Context, spec assetSpec) Card {
	candles, err := a.candlesFor(ctx, spec)
	if err != nil {
		return assetOffline(spec, "Momentum Agent", "Momentum", keyMomentum, howTexts[keyMomentum])
	}
	r, err := momentumReadFromCandles(spec, candles)
	if errors.Is(err, errInsufficientHistory) {
		return insufficientCard(spec, "Momentum Agent", "Momentum", keyMomentum, howTexts[keyMomentum], "RSI(14)/MACD")
	}
	if err != nil {
		return assetOffline(spec, "Momentum Agent", "Momentum", keyMomentum, howTexts[keyMomentum])
	}
	c := Card{
		Agent:      "Momentum Agent",
		ShortName:  "Momentum",
		Asset:      spec.Display,
		Command:    keyMomentum,
		HowItWorks: howTexts[keyMomentum],
		DataTime:   r.closeAt,
		Emoji:      momentumEmoji(r.verdict),
		Verdict:    fmt.Sprintf("%s: %s", r.name, strings.ToUpper(r.verdict)),
		Short:      r.verdict,
		Deviation:  clampInt(int(math.Abs(r.rsi-50)*2), 0, 100),
	}
	c.Facts = append(c.Facts,
		fmt.Sprintf("RSI(14): %.1f", r.rsi),
		// %g keeps FX-scale histograms (~0.0005) readable without padding
		// BTC-scale ones (~180) with useless decimals.
		fmt.Sprintf("MACD histogram: %+.4g", r.hist),
		fmt.Sprintf("Rule: RSI 55+/45- with matching MACD sign · %s candles", spec.Interval),
	)
	if spec.Source == srcYahoo && spec.Interval == "4h" {
		c.Facts = append(c.Facts, yahooAgg4hNote) // B1: disclose the 1h→4h merge
	}
	// Binance assets get the volume read from the SAME series as the RSI/MACD
	// math. Yahoo FX volume is null throughout, so no line over a fake 0×.
	if spec.Source == srcBinance {
		if ratio, ok := volRatio20(candles); ok {
			c.Facts = append(c.Facts, fmt.Sprintf("Volume: %.2f× its 20-bar average (%s)", ratio, spec.Interval))
		}
	}
	if spec.Source == srcYahoo {
		decorateFX(&c)
	}
	return c
}

// yahooAgg4hNote discloses the B1 aggregation whenever a Yahoo asset runs on
// 4h bars — Yahoo has no native 4h interval, so the bars are merged 1h groups
// (see aggregate1hTo4h for the exact rules).
const yahooAgg4hNote = "Note: 4h FX/gold bars aggregated from Yahoo 1h (Yahoo serves no native 4h)"

// MomentumScanCard is the user-configured scan (B1): up to scanMaxAssets
// registry assets, optionally re-based on one shared timeframe (tf "" keeps
// each asset's native interval). keys must be pre-validated registry keys
// (parseAssetList); per-asset failures degrade to honest fact lines, and the
// card goes offline only when EVERY asset failed.
func (a *Agents) MomentumScanCard(ctx context.Context, keys []string, tf string) Card {
	specs := make([]assetSpec, 0, len(keys))
	displays := make([]string, 0, len(keys))
	for _, k := range keys {
		spec := assetTable[k]
		if tf != "" {
			spec.Interval = tf // validated upstream via specWithTF/momentumTFs
		}
		specs = append(specs, spec)
		displays = append(displays, spec.Display)
	}

	type scanResult struct {
		read         momentumRead
		insufficient bool
		failed       bool
	}
	results := make([]scanResult, len(specs))
	var wg sync.WaitGroup
	for i, spec := range specs {
		wg.Add(1)
		go func(i int, spec assetSpec) {
			defer wg.Done()
			r, err := a.momentumReadFor(ctx, spec)
			switch {
			case err == nil:
				results[i] = scanResult{read: r}
			case errors.Is(err, errInsufficientHistory):
				results[i] = scanResult{read: r, insufficient: true}
			default:
				results[i] = scanResult{failed: true}
			}
		}(i, spec)
	}
	wg.Wait()

	c := Card{
		Agent:      "Momentum Agent",
		ShortName:  "Momentum",
		Asset:      strings.Join(displays, "/"),
		Command:    keyMomentum,
		HowItWorks: howTexts[keyMomentum],
		DataTime:   time.Now().UTC(), // narrowed below to the OLDEST closed bar used
	}

	anyOK := false
	anyInsufficient := false
	anyYahoo := false
	agg4h := false
	fxOpen := isForexOpen(time.Now())
	var verdictParts []string
	var assetResults []AssetResult
	maxDev := 0
	lead := ""
	for i, res := range results {
		spec := specs[i]
		if spec.Source == srcYahoo {
			anyYahoo = true
			if spec.Interval == "4h" {
				agg4h = true
			}
		}
		name := spec.Display
		if tf == "" {
			// Mixed native intervals — each line names its own.
			name = fmt.Sprintf("%s (%s)", spec.Display, spec.Interval)
		}
		switch {
		case res.failed:
			c.Facts = append(c.Facts, name+": data unavailable right now")
			assetResults = append(assetResults, assetResult(spec.Display, statusSourceOffline))
		case res.insufficient:
			anyInsufficient = true
			c.Facts = append(c.Facts, name+": insufficient history for RSI/MACD")
			assetResults = append(assetResults, assetResult(spec.Display, statusInsufficientHistory))
		default:
			anyOK = true
			assetResults = append(assetResults, assetResult(spec.Display, statusOK))
			r := res.read
			if lead == "" {
				lead = r.verdict // first requested asset leads the semaphore
			}
			verdictParts = append(verdictParts, fmt.Sprintf("%s: %s", spec.Display, strings.ToUpper(r.verdict)))
			line := fmt.Sprintf("%s: RSI(14) %.1f · MACD hist %+.3g → %s", name, r.rsi, r.hist, r.verdict)
			if spec.Source == srcYahoo && !fxOpen {
				line += " (market closed)"
			}
			c.Facts = append(c.Facts, line)
			if !r.closeAt.IsZero() && r.closeAt.Before(c.DataTime) {
				c.DataTime = r.closeAt
			}
			// Deviation stays CRYPTO-ONLY (v1 priority rule — see priority.go).
			if spec.Source == srcBinance {
				if d := clampInt(int(math.Abs(r.rsi-50)*2), 0, 100); d > maxDev {
					maxDev = d
				}
			}
		}
	}
	c.Results = assetResults
	if !anyOK {
		// Zero real readings. The WHY must be honest (review fix 3): when at
		// least one asset answered but was too short, the scan degrades as
		// insufficient_history; only an all-sources-dead sweep is
		// source_offline.
		if anyInsufficient {
			c.Emoji = emojiNeutral
			c.Verdict = "Insufficient history — no verdict"
			c.Short = "insufficient history"
			c.Offline = true
			c.Status = statusInsufficientHistory
			return c
		}
		off := offlineCard("Momentum Agent", "Momentum", c.Asset, keyMomentum, howTexts[keyMomentum])
		off.Results = assetResults
		return off
	}

	rule := "Rule: RSI 55+/45- with matching MACD sign"
	if tf != "" {
		rule += " · " + tf + " candles"
	}
	c.Facts = append(c.Facts, rule)
	if agg4h {
		c.Facts = append(c.Facts, yahooAgg4hNote)
	}
	if anyYahoo {
		c.SourceNote = "FX/gold data: Yahoo Finance"
	}
	c.Verdict = strings.Join(verdictParts, " · ")
	if lead == "" {
		lead = "neutral"
	}
	c.Emoji = momentumEmoji(lead)
	c.Short = strings.ToLower(lead)
	c.Deviation = maxDev
	return c
}

// ── FX overview ──────────────────────────────────────────────────────────────

// fxReads computes the /fx snapshot for all pairs concurrently (closed bars
// only — candlesFor drops the forming bar). Direction: EMA50 vs EMA200 on
// 1h. Day change: last close vs the latest bar at least 24h older (session
// gaps mean "previous trading day" on weekends). Pairs with too little
// history for EMA200/RSI report it explicitly instead of a fake flat.
func (a *Agents) fxReads(ctx context.Context) []fxRead {
	reads := make([]fxRead, len(fxPairs))
	var wg sync.WaitGroup
	for i, key := range fxPairs {
		wg.Add(1)
		go func(i int, spec assetSpec) {
			defer wg.Done()
			r := fxRead{Pair: spec.Display}
			candles, err := a.candlesFor(ctx, spec)
			if err != nil {
				reads[i] = r // OK=false → "data unavailable"
				return
			}
			closes := closesOf(candles)
			ema50, ok50 := emaLast(closes, 50)
			ema200, ok200 := emaLast(closes, 200)
			rsi, okRSI := rsiWilder(closes, 14)
			if !ok50 || !ok200 || !okRSI {
				r.Insufficient = true // explicit, never a confident flat
				reads[i] = r
				return
			}
			r.OK = true
			r.RSI = rsi
			r.CloseAt = closeTimeOf(candles, spec.Interval)
			switch {
			case ema50 > ema200:
				r.Dir = "up"
			case ema50 < ema200:
				r.Dir = "down"
			default:
				r.Dir = "flat"
			}
			if pos, ok := dayRange(candles); ok {
				r.DayPos, r.HasRange = pos, true
			}
			lastBar := candles[len(candles)-1]
			for j := len(candles) - 2; j >= 0; j-- {
				if candles[j].Time <= lastBar.Time-86400 {
					if candles[j].Close != 0 {
						r.DayChangePct = (lastBar.Close - candles[j].Close) / candles[j].Close * 100
						r.HasDay = true
					}
					break
				}
			}
			reads[i] = r
		}(i, assetTable[key])
	}
	wg.Wait()
	return reads
}

// FXCard builds the /fx overview from live reads. Footer time = the newest
// closed bar among the pairs that produced data.
func (a *Agents) FXCard(ctx context.Context) Card {
	reads := a.fxReads(ctx)
	anyOK := false
	var latest time.Time
	for _, r := range reads {
		if !r.OK {
			continue
		}
		anyOK = true
		if r.CloseAt.After(latest) {
			latest = r.CloseAt
		}
	}
	if !anyOK {
		c := offlineCard("FX Agent", "FX", "", keyFX, howTexts[keyFX])
		c.Verdict = fxOfflineVerdict
		c.SourceNote = "data: Yahoo Finance"
		return c
	}
	if latest.IsZero() {
		latest = time.Now().UTC()
	}
	return fxOverviewCard(reads, isForexOpen(time.Now()), latest)
}

// ── Trend ────────────────────────────────────────────────────────────────────

const (
	trendFlat     = "flat"
	trendGrey     = "grey"
	trendUp       = "up"
	trendDown     = "down"
	trendConflict = "conflict"
)

// classifyTrend is the /trend state machine:
//
//	ADX < 20            → flat (trading not advised)
//	20 ≤ ADX < 25       → grey zone
//	ADX ≥ 25, EMA50>EMA200, close>EMA50 → confirmed uptrend
//	ADX ≥ 25, EMA50<EMA200, close<EMA50 → confirmed downtrend
//	ADX ≥ 25, otherwise → indicator conflict
func classifyTrend(adx, ema50, ema200, closePrice float64) string {
	switch {
	case adx < 20:
		return trendFlat
	case adx < 25:
		return trendGrey
	case ema50 > ema200 && closePrice > ema50:
		return trendUp
	case ema50 < ema200 && closePrice < ema50:
		return trendDown
	default:
		return trendConflict
	}
}

// invalidationLevel is the below-form invalidation: 1 ATR(14) below the lower
// edge of the EMA cluster (min(EMA50, EMA200) − 1×ATR). One ATR of slack
// keeps ordinary noise from reading as a break; a close beyond it means the
// EMA structure the verdict stands on is gone.
func invalidationLevel(ema50, ema200, atr float64) float64 {
	return math.Min(ema50, ema200) - atr
}

// invalidationFor picks the invalidation level AND direction per state
// (review fix 8): a DOWNTREND's structure breaks UPWARD — max(EMA50, EMA200)
// + 1 ATR, worded "above" — while the uptrend and the non-directional states
// keep the below-form (for flat/grey/conflict the below-form marks where the
// bearish repricing of the EMA cluster completes; the direction ships
// machine-readably as invalidation_side so no consumer has to parse wording).
func invalidationFor(state string, ema50, ema200, atr float64) (float64, string) {
	if state == trendDown {
		return math.Max(ema50, ema200) + atr, "above"
	}
	return invalidationLevel(ema50, ema200, atr), "below"
}

// pullbackZoneFor gates the EMA20-EMA50 pullback band (B3): present ONLY in
// confirmed-trend states — flat/grey/conflict have no trend to pull back
// within, so offering an "entry band" there would be an invented signal. See
// the PullbackZone type for why the band is the pullback target.
func pullbackZoneFor(state string, ema20, ema50 float64) *PullbackZone {
	if state != trendUp && state != trendDown {
		return nil
	}
	return &PullbackZone{From: ema20, To: ema50}
}

func trendVerdict(state string, adx float64) string {
	switch state {
	case trendFlat:
		return fmt.Sprintf("Flat. Trading not advised (ADX %.0f < 20)", adx)
	case trendGrey:
		return "Grey zone — trend forming, not confirmed"
	case trendUp:
		return "Confirmed UPTREND"
	case trendDown:
		return "Confirmed DOWNTREND"
	default:
		return "Indicator conflict — stand aside"
	}
}

func (a *Agents) TrendCard(ctx context.Context, spec assetSpec) Card {
	candles, err := a.candlesFor(ctx, spec)
	if err != nil {
		return assetOffline(spec, "Trend Agent", "Trend", keyTrend, howTexts[keyTrend])
	}
	closes := closesOf(candles)
	highs, lows := highsLowsOf(candles)
	ema20, ok20 := emaLast(closes, 20)
	ema50, ok50 := emaLast(closes, 50)
	ema200, ok200 := emaLast(closes, 200)
	adx, okADX := adxWilder(highs, lows, closes, 14)
	rsi, okRSI := rsiWilder(closes, 14)
	if !ok20 || !ok50 || !ok200 || !okADX || !okRSI {
		c := insufficientCard(spec, "Trend Agent", "Trend", keyTrend, howTexts[keyTrend], "EMA200/ADX(14)")
		c.DataTime = closeTimeOf(candles, spec.Interval)
		return c
	}
	last := closes[len(closes)-1]

	// HH/HL swing structure (B3, review fix 7) — the SAME swing points the
	// S/R agent clusters (wing 3), classified over the last six ALTERNATING
	// pivots. It PARTICIPATES in the state machine: an EMA/ADX-confirmed
	// direction with a contradicting or mixed pivot structure demotes to the
	// grey zone — the card must never say "Confirmed UPTREND" over LH/LL
	// swings. An empty read ("" — too few pivots, e.g. a monotone series)
	// does NOT demote: there is no structure evidence against the trend, and
	// a clean monotone rise is the strongest trend there is.
	swingHighs, swingLows := swingPointsIdx(highs, lows, 3)
	structure := hhhlStructure(swingHighs, swingLows)

	state := classifyTrend(adx, ema50, ema200, last)
	structureDemoted := ""
	switch {
	case state == trendUp && structure == "lh_ll":
		structureDemoted = "Structure disagrees (LH/LL) — trend not confirmed"
	case state == trendUp && structure == "mixed":
		structureDemoted = "Structure mixed — trend not confirmed"
	case state == trendDown && structure == "hh_hl":
		structureDemoted = "Structure disagrees (HH/HL) — trend not confirmed"
	case state == trendDown && structure == "mixed":
		structureDemoted = "Structure mixed — trend not confirmed"
	}
	if structureDemoted != "" {
		state = trendGrey
	}

	c := Card{
		Agent:      "Trend Agent",
		ShortName:  "Trend",
		Asset:      spec.Display,
		Command:    keyTrend,
		HowItWorks: howTexts[keyTrend],
		DataTime:   closeTimeOf(candles, spec.Interval),
		Verdict:    trendVerdict(state, adx),
	}
	emaStructure := "bearish structure"
	if ema50 > ema200 {
		emaStructure = "bullish structure"
	}
	// ADX drives the verdict → its confirm threshold rides beside the number
	// (batch-2 rule: thresholds in parentheses only where the number drives
	// the verdict — RSI here is informational, so it carries none).
	c.Facts = append(c.Facts,
		fmt.Sprintf("EMA50 %s vs EMA200 %s — %s", trimFloat(ema50), trimFloat(ema200), emaStructure),
	)
	// One structure fact: the demotion wording when the gate fired, the plain
	// read otherwise. Silent when the window has too few pivots to classify —
	// no claim beats an invented one.
	switch {
	case structureDemoted != "":
		c.Facts = append(c.Facts, structureDemoted)
	case structure == "hh_hl":
		c.Facts = append(c.Facts, "Structure: HH/HL confirmed (last 3 swing highs and lows rising)")
	case structure == "lh_ll":
		c.Facts = append(c.Facts, "Structure: LH/LL (last 3 swing highs and lows falling)")
	case structure == "mixed":
		c.Facts = append(c.Facts, "Structure: mixed (last swings not aligned)")
	}
	c.Facts = append(c.Facts,
		fmt.Sprintf("ADX(14): %.1f (trend confirms above 25) · RSI(14): %.1f", adx, rsi),
		fmt.Sprintf("Last %s close: %s", spec.Interval, trimFloat(last)),
	)
	// Pullback zone (B3): the EMA20-EMA50 band, confirmed trends only —
	// rendered low-high for reading, raw EMA20/EMA50 in levels. A structure-
	// demoted state is NOT confirmed, so it offers no zone.
	zone := pullbackZoneFor(state, ema20, ema50)
	if zone != nil {
		lo, hi := math.Min(ema20, ema50), math.Max(ema20, ema50)
		c.Facts = append(c.Facts, fmt.Sprintf("Pullback zone: %s-%s (EMA20-EMA50 band)",
			trimFloat(lo), trimFloat(hi)))
	}
	// Invalidation: the price that breaks the structure this card reads —
	// direction-aware (review fix 8: downtrends break UPWARD, above the EMA
	// cluster). ATR(14) is guaranteed by the EMA200 history gate above; the
	// >0 guard keeps a degenerate flat series from rendering a fake level.
	if atrSeries := atrSeriesWilder(highs, lows, closes, 14); len(atrSeries) > 0 {
		if atr := atrSeries[len(atrSeries)-1]; atr > 0 {
			inv, side := invalidationFor(state, ema50, ema200, atr)
			prep := "under"
			if side == "above" {
				prep = "over"
			}
			c.Facts = append(c.Facts, fmt.Sprintf(
				"Invalidation: %s %s the structure is broken (1 ATR %s the EMA cluster)",
				side, trimFloat(inv), prep))
			c.Levels = TrendLevels{Invalidation: inv, InvalidationSide: side, PullbackZone: zone}
		}
	}
	if spec.Source == srcYahoo {
		decorateFX(&c)
	}
	switch state {
	case trendUp:
		c.Emoji, c.Short = emojiBull, "confirmed uptrend"
	case trendDown:
		c.Emoji, c.Short = emojiBear, "confirmed downtrend"
	case trendGrey:
		c.Emoji, c.Short = emojiNeutral, "grey zone"
	case trendConflict:
		c.Emoji, c.Short = emojiNeutral, "indicator conflict"
	default:
		c.Emoji, c.Short = emojiNeutral, "flat — no trade"
	}
	// Confirmed trends count double toward the priority rule; unconfirmed
	// states carry only the raw ADX (documented in pickTop's rule 2).
	if state == trendUp || state == trendDown {
		c.Deviation = clampInt(int(adx)*2, 0, 100)
	} else {
		c.Deviation = clampInt(int(adx), 0, 100)
	}
	return c
}

// ── Support / Resistance ─────────────────────────────────────────────────────

// srStrongTouches is the touch count from which a clustered level counts as
// STRONG. Below it a level is a candidate; at 7+ distinct swing touches the
// market has respected the price often enough to call the level established.
// Surfaced beside the touch numbers on the card (batch-2 thresholds rule).
const srStrongTouches = 7

func (a *Agents) SRCard(ctx context.Context, spec assetSpec) Card {
	candles, err := a.candlesFor(ctx, spec)
	if err != nil {
		return assetOffline(spec, "S/R Agent", "S/R", keySR, howTexts[keySR])
	}
	if len(candles) < 20 { // too few closed bars for meaningful swings
		c := insufficientCard(spec, "S/R Agent", "S/R", keySR, howTexts[keySR], "swing detection")
		c.DataTime = closeTimeOf(candles, spec.Interval)
		return c
	}
	sup, res := supportResistance(candles, 3, 0.5)
	last := candles[len(candles)-1].Close
	// Both sides empty is never "Key levels around …" (review fix 2). Two
	// distinct causes, two honest states:
	//   - ZERO swing points in the whole window (monotone/flat tape): there is
	//     no structure to read at all — same class as a too-short series →
	//     insufficient_history.
	//   - swings exist but no cluster sits strictly on either side of the last
	//     price: that IS a real finding — an explicit ok "no significant
	//     levels" statement with empty arrays.
	if len(sup) == 0 && len(res) == 0 {
		hs, ls := highsLowsOf(candles)
		sh, sl := swingPointsIdx(hs, ls, 3)
		if len(sh)+len(sl) == 0 {
			c := insufficientCard(spec, "S/R Agent", "S/R", keySR, howTexts[keySR], "swing structure")
			c.DataTime = closeTimeOf(candles, spec.Interval)
			return c
		}
		c := Card{
			Emoji:      emojiNeutral,
			Agent:      "S/R Agent",
			ShortName:  "S/R",
			Asset:      spec.Display,
			Command:    keySR,
			HowItWorks: howTexts[keySR],
			DataTime:   closeTimeOf(candles, spec.Interval),
			Verdict:    "No significant levels detected in the window",
			Short:      "no significant levels",
			Levels:     SRLevels{Supports: srPoints(nil), Resistances: srPoints(nil)},
		}
		c.Facts = append(c.Facts,
			fmt.Sprintf("Swing points exist (%d), but no cluster sits clear of the last price %s", len(sh)+len(sl), trimFloat(last)),
			fmt.Sprintf("Method: swing clusters over %d×%s candles · strength = touches + 0.5 per above-median-volume touch (strong from %d touches)",
				len(candles), spec.Interval, srStrongTouches),
		)
		if spec.Source == srcYahoo {
			decorateFX(&c)
		}
		return c
	}
	c := Card{
		Emoji:      emojiNeutral,
		Agent:      "S/R Agent",
		ShortName:  "S/R",
		Asset:      spec.Display,
		Command:    keySR,
		HowItWorks: howTexts[keySR],
		DataTime:   closeTimeOf(candles, spec.Interval),
		Verdict:    fmt.Sprintf("Key levels around %s", trimFloat(last)),
	}
	c.Facts = append(c.Facts, "Resistance: "+srLine(res), "Support: "+srLine(sup))
	// Machine-readable twin of the two lines above: raw cluster means, not the
	// display-rounded labels.
	c.Levels = SRLevels{Supports: srPoints(sup), Resistances: srPoints(res)}
	if nl := nearestLevelLine(sup, res, last); nl != "" {
		c.Facts = append(c.Facts, nl)
	}
	// Strength threshold beside the numbers (batch-2 rule) + the B4 formulas,
	// stated on the card so the numbers are auditable. Frequency wording only.
	c.Facts = append(c.Facts, fmt.Sprintf(
		"Method: swing clusters over %d×%s candles · strength = touches + 0.5 per above-median-volume touch (strong from %d touches)",
		len(candles), spec.Interval, srStrongTouches))
	c.Facts = append(c.Facts,
		"Tests: close within 0.25×ATR of a level = test · close beyond by >0.25×ATR within 3 bars = break, else held. Counts are frequencies over this window, not probabilities")
	short := "no clear levels"
	if len(res) > 0 && len(sup) > 0 {
		short = fmt.Sprintf("R %s / S %s", srLevelLabel(res[0]), srLevelLabel(sup[0]))
	}
	c.Short = short
	if spec.Source == srcYahoo {
		decorateFX(&c)
	}
	return c
}

// srPoints converts clustered levels to the machine-readable envelope form:
// raw cluster means at full precision, never the display-rounded labels.
// Always non-nil so an empty side serializes as [].
func srPoints(levels []SRLevel) []SRPoint {
	pts := make([]SRPoint, 0, len(levels))
	for _, l := range levels {
		p := SRPoint{
			Level:     l.Raw,
			Touches:   l.Touches,
			Strength:  l.Strength,
			Weakening: l.Weakening,
			Breaks:    l.Breaks,
			Holds:     l.Holds,
		}
		if !l.LastTouch.IsZero() {
			p.LastTouch = l.LastTouch.Format(time.RFC3339)
		}
		pts = append(pts, p)
	}
	return pts
}

// srLevelLabel: integer levels for BTC/gold-scale prices (original card
// contract); full pip precision for FX-scale prices where an integer would
// destroy the level.
func srLevelLabel(l SRLevel) string {
	switch {
	case l.Raw >= 100:
		return fmt.Sprintf("%d", l.Level)
	case l.Raw >= 10:
		return fmt.Sprintf("%.2f", l.Raw)
	default:
		return fmt.Sprintf("%.4f", l.Raw)
	}
}

// nearestLevelLine names the level closest to the last price with its
// conventional label (S1..S3 / R1..R3 = position in the strength-sorted
// list) and the percent distance: "Price 1.3% above S1 (61200)". "" when no
// levels or no price. Supports sit below price, resistances above, so the
// direction word is fixed per side.
func nearestLevelLine(sup, res []SRLevel, last float64) string {
	if last <= 0 {
		return ""
	}
	best := ""
	bestDist := math.MaxFloat64
	consider := func(prefix, rel string, levels []SRLevel) {
		for i, l := range levels {
			if d := math.Abs(last - l.Raw); d < bestDist {
				bestDist = d
				best = fmt.Sprintf("Price %.1f%% %s %s%d (%s)", d/last*100, rel, prefix, i+1, srLevelLabel(l))
			}
		}
	}
	consider("S", "above", sup)
	consider("R", "below", res)
	return best
}

func srLine(levels []SRLevel) string {
	if len(levels) == 0 {
		return "no clustered levels in window"
	}
	parts := make([]string, 0, len(levels))
	for _, l := range levels {
		word := "touches"
		if l.Touches == 1 {
			word = "touch"
		}
		entry := fmt.Sprintf("%s (%d %s", srLevelLabel(l), l.Touches, word)
		// B4: how the level actually behaved when tested — frequency counts
		// over this window, never a probability.
		if tests := l.Breaks + l.Holds; tests > 0 {
			entry += fmt.Sprintf(", held %d of %d tests", l.Holds, tests)
		}
		if l.Weakening {
			entry += ", weakening: volume fading"
		}
		parts = append(parts, entry+")")
	}
	return strings.Join(parts, " · ")
}

// ── Volatility ───────────────────────────────────────────────────────────────

const (
	volExpanding  = "expanding"
	volCompressed = "compressed"
	volNormal     = "normal"
)

// volState: ATR(14) now vs its 30-bar average.
func volState(ratio float64) string {
	switch {
	case ratio >= 1.25:
		return volExpanding
	case ratio <= 0.8:
		return volCompressed
	default:
		return volNormal
	}
}

func (a *Agents) VolCard(ctx context.Context, spec assetSpec) Card {
	candles, err := a.candlesFor(ctx, spec)
	if err != nil {
		return assetOffline(spec, "Volatility Agent", "Volatility", keyVol, howTexts[keyVol])
	}
	closes := closesOf(candles)
	highs, lows := highsLowsOf(candles)
	series := atrSeriesWilder(highs, lows, closes, 14)
	n := len(series)
	if n < 45 { // 14 warmup + 30 window + current — short data is stated, not faked
		c := insufficientCard(spec, "Volatility Agent", "Volatility", keyVol, howTexts[keyVol], "ATR(14) 30-bar baseline")
		c.DataTime = closeTimeOf(candles, spec.Interval)
		return c
	}
	now := series[n-1]
	var sum float64
	for _, v := range series[n-31 : n-1] {
		sum += v
	}
	avg := sum / 30
	ratio := 0.0
	if avg > 0 {
		ratio = now / avg
	}
	state := volState(ratio)
	c := Card{
		Emoji:      emojiNeutral,
		Agent:      "Volatility Agent",
		ShortName:  "Volatility",
		Asset:      spec.Display,
		Command:    keyVol,
		HowItWorks: howTexts[keyVol],
		DataTime:   closeTimeOf(candles, spec.Interval),
		Levels:     VolLevels{ExpansionRatio: ratio}, // unrounded, envelope "levels"
	}
	switch state {
	case volExpanding:
		c.Verdict, c.Short = "Volatility EXPANDING — breakout conditions", fmt.Sprintf("expanding %.2f×", ratio)
	case volCompressed:
		c.Verdict, c.Short = "Volatility COMPRESSED — range conditions, expansion often follows", fmt.Sprintf("compressed %.2f×", ratio)
	default:
		c.Verdict, c.Short = "Volatility NORMAL — no expansion signal", fmt.Sprintf("normal %.2f×", ratio)
	}
	last := closes[len(closes)-1]
	c.Facts = append(c.Facts,
		fmt.Sprintf("ATR(14) now: %s (%.2f%% of price)", trimFloat(now), now/last*100),
		fmt.Sprintf("30-bar ATR average: %s", trimFloat(avg)),
		fmt.Sprintf("Ratio: %.2f× (expansion at 1.25×, compression at 0.80×)", ratio),
	)
	if spec.Source == srcYahoo {
		decorateFX(&c)
	}
	return c
}

// ── Risk calculator ──────────────────────────────────────────────────────────

// riskResult is pure position-sizing math. It deliberately carries NO
// direction label (team review batch 2): "LONG"/"SHORT" read as a trade
// suggestion, and the calculator's only claim is arithmetic — size, max loss,
// notional. The math is direction-agnostic (|entry − stop|) anyway.
type riskResult struct {
	RiskAmount float64
	PerUnit    float64
	Size       float64
	Notional   float64
}

func calcRisk(balance, riskPct, entry, stop float64) (riskResult, error) {
	switch {
	case balance <= 0:
		return riskResult{}, fmt.Errorf("balance must be positive")
	case riskPct <= 0 || riskPct > 100:
		return riskResult{}, fmt.Errorf("risk%% must be in (0, 100]")
	case entry <= 0 || stop <= 0:
		return riskResult{}, fmt.Errorf("entry and stop must be positive")
	case entry == stop:
		return riskResult{}, fmt.Errorf("stop must differ from entry")
	}
	r := riskResult{
		RiskAmount: balance * riskPct / 100,
		PerUnit:    math.Abs(entry - stop),
	}
	r.Size = r.RiskAmount / r.PerUnit
	r.Notional = r.Size * entry
	return r, nil
}

const riskUsage = "Usage: /risk <balance> <risk%> <entry> <stop>"

// RiskCard renders the calculator. With no args it shows a worked example —
// clearly labeled, never pretending to be live data.
func (a *Agents) RiskCard(args []float64, isExample bool, parseErr error) Card {
	c := Card{
		Emoji:      emojiNeutral,
		Agent:      "Risk Calculator",
		ShortName:  "Risk",
		Command:    keyRisk,
		HowItWorks: howTexts[keyRisk],
		DataTime:   time.Now().UTC(),
	}
	if parseErr != nil {
		c.Verdict = "Could not parse that"
		c.Facts = []string{parseErr.Error(), riskUsage, "Example: /risk 10000 1 100000 98000"}
		return c
	}
	r, err := calcRisk(args[0], args[1], args[2], args[3])
	if err != nil {
		c.Verdict = "Those numbers don't work"
		c.Facts = []string{err.Error(), riskUsage}
		return c
	}
	// Pure math, no direction word: sizing is |entry − stop| arithmetic, and a
	// LONG/SHORT label would read as a trade suggestion (batch-2 language rule).
	c.Verdict = fmt.Sprintf("Position size: %s units", trimFloat6(r.Size))
	if isExample {
		c.Verdict = "Example — " + c.Verdict
	}
	c.Facts = append(c.Facts,
		fmt.Sprintf("Max loss at stop: %s (%.4g%% of %s balance)", usd(r.RiskAmount), args[1], usd(args[0])),
		fmt.Sprintf("Entry %s / stop %s → %s risk per unit", trimFloat(args[2]), trimFloat(args[3]), trimFloat(r.PerUnit)),
		fmt.Sprintf("Position notional: %s", usd(r.Notional)),
		"Position sizing math only — not a trade suggestion.",
	)
	if isExample {
		c.Facts = append(c.Facts, riskUsage)
	}
	return c
}

// ── formatting helpers ───────────────────────────────────────────────────────

func closesOf(candles []types.OHLCVCandle) []float64 {
	out := make([]float64, len(candles))
	for i, c := range candles {
		out[i] = c.Close
	}
	return out
}

func highsLowsOf(candles []types.OHLCVCandle) (highs, lows []float64) {
	highs = make([]float64, len(candles))
	lows = make([]float64, len(candles))
	for i, c := range candles {
		highs[i] = c.High
		lows[i] = c.Low
	}
	return highs, lows
}

// usd renders a compact dollar amount: $6.96M, $412K, $99.50, -$1.2B.
func usd(v float64) string {
	sign := ""
	if v < 0 {
		sign = "-"
		v = -v
	}
	switch {
	case v >= 1e9:
		return fmt.Sprintf("%s$%.2fB", sign, v/1e9)
	case v >= 1e6:
		return fmt.Sprintf("%s$%.2fM", sign, v/1e6)
	case v >= 1e3:
		return fmt.Sprintf("%s$%.1fK", sign, v/1e3)
	default:
		return fmt.Sprintf("%s$%.2f", sign, v)
	}
}

// trimFloat renders a price-like float at sensible precision. FX majors
// live in the 4th decimal, so anything under 10 keeps pip precision.
func trimFloat(v float64) string {
	a := math.Abs(v)
	switch {
	case a >= 1000:
		return fmt.Sprintf("%.0f", v)
	case a >= 10:
		return fmt.Sprintf("%.2f", v)
	default:
		return fmt.Sprintf("%.4f", v)
	}
}

func trimFloat6(v float64) string { return fmt.Sprintf("%.6g", v) }

func truncate(s string, max int) string {
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return strings.TrimSpace(string(r[:max-1])) + "…"
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
