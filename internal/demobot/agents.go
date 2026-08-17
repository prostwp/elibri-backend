package demobot

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
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
		key := "yahoo|" + spec.Symbol
		candles, err = a.klines.cached(key, func() ([]types.OHLCVCandle, error) {
			c, ferr := fetchYahooCandles(ctx, spec.Symbol)
			if (ferr != nil || len(c) == 0) && spec.Fallback != "" {
				return fetchYahooCandles(ctx, spec.Fallback)
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
	keyMomentum: "RSI(14) + MACD histogram on 4h Binance candles. RSI 55+ with positive MACD = buy; RSI 45- with negative = sell; anything else = neutral.",
	keyTrend:    "State machine on 4h candles: ADX<20 = flat (no trade), 20-25 = grey zone, ADX 25+ with price and EMA50/EMA200 aligned = confirmed trend; misaligned = conflict.",
	keySR:       "Finds swing highs/lows on 4h candles and clusters levels within 0.5%. Strength = number of touches. Levels are rounded to whole numbers.",
	keyVol:      "ATR(14) now vs its 30-bar average. Ratio 1.25+ = volatility expanding (breakout regime); 0.8- = compressed (range regime).",
	keyRisk:     "Position size = (balance × risk%) ÷ |entry − stop|. Keeps one losing trade at a fixed fraction of the account. Works for any asset.",
	keyFX:       "EMA50 vs EMA200 direction, RSI(14) and 24h change on 1h Yahoo Finance data for EURUSD, GBPUSD, USDJPY, XAUUSD. Weekend closures (Fri 21:00–Sun 21:00 UTC) are flagged, never hidden.",
	keyDigest:   "Deterministic priority: RISK-OFF macro always tops; otherwise the strongest deviation from neutral among funding, momentum, trend. Ties break funding > momentum > trend.",
	keyTop:      "Deterministic priority: RISK-OFF macro always tops; otherwise the strongest deviation from neutral among funding, momentum, trend. Ties break funding > momentum > trend.",
}

// ── Macro ────────────────────────────────────────────────────────────────────

// MacroCard returns the card plus the raw regime for the priority rule
// ("" when the source is offline).
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
	switch m.Regime {
	case "risk_on":
		c.Emoji, c.Verdict, c.Short = emojiBull, "RISK-ON — big money leaning into risk", "risk-on"
	case "risk_off":
		c.Emoji, c.Verdict, c.Short = emojiBear, "RISK-OFF — big money defensive", "risk-off"
	default:
		c.Emoji, c.Verdict, c.Short = emojiNeutral, "MIXED — no single regime in control", "mixed"
	}

	var tail, head, neut, dark int
	for _, l := range m.Lamps {
		switch l.Status {
		case "tailwind":
			tail++
		case "headwind":
			head++
		case "neutral":
			neut++
		default:
			dark++
		}
	}
	if dark == len(m.Lamps) && len(m.Lamps) > 0 {
		note := "no tradfin data right now"
		if !m.TradfinOpen {
			note += " (market closed)"
		}
		c.Facts = append(c.Facts, "Lamps: "+note)
	} else {
		c.Facts = append(c.Facts, fmt.Sprintf("Lamps: %d tailwind / %d headwind / %d neutral", tail, head, neut))
	}
	if m.FNG != nil && m.FNG.OK {
		c.Facts = append(c.Facts, fmt.Sprintf("Crypto Fear & Greed: %d — %s", m.FNG.Value, m.FNG.Label))
	}
	if idea := strings.TrimSpace(m.GeneratedIdea); idea != "" {
		c.Facts = append(c.Facts, truncate(idea, 180))
	}
	if m.Composite != nil {
		c.Confidence = m.Composite
		c.Deviation = clampInt(abs(*m.Composite-50)*2, 0, 100)
	}
	return c, m.Regime
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
		if btc.Confidence > 0 {
			conf := btc.Confidence
			c.Confidence = &conf
		}
	}

	// Top BTC transfers of the last 24h by USD size.
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
		if i == 2 {
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
		c.Facts = append(c.Facts, fmt.Sprintf("Liquidations 1h: %s longs vs %s shorts", usd(longUSD), usd(shortUSD)))
		if len(liq.Zones) > 0 {
			z := liq.Zones[0]
			c.Facts = append(c.Facts, fmt.Sprintf("Magnet zone: %s %s (%s, %d hits)", z.Symbol, z.PriceBand, usd(z.TotalUSD), z.Count))
		}
	}
	return c
}

// ── Momentum ─────────────────────────────────────────────────────────────────

// Momentum verdict rule (documented, deterministic): RSI≥55 with positive
// MACD histogram = buy; RSI≤45 with negative histogram = sell; else neutral.
func momentumVerdict(rsi, macdHist float64) string {
	switch {
	case rsi >= 55 && macdHist > 0:
		return "buy"
	case rsi <= 45 && macdHist < 0:
		return "sell"
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

func (a *Agents) momentumReadFor(ctx context.Context, spec assetSpec) (momentumRead, error) {
	candles, err := a.candlesFor(ctx, spec)
	if err != nil {
		return momentumRead{}, err
	}
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

func momentumEmoji(verdict string) string {
	switch verdict {
	case "buy":
		return emojiBull
	case "sell":
		return emojiBear
	default:
		return emojiNeutral
	}
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
	for _, key := range []string{"btc", "eth"} {
		r, err := a.momentumReadFor(ctx, assetTable[key])
		switch {
		case err == nil:
			reads = append(reads, r)
		case errors.Is(err, errInsufficientHistory):
			insufficientLines = append(insufficientLines, r.name+": insufficient history for RSI/MACD")
		}
	}
	xau, xauErr := a.momentumReadFor(ctx, xauSpec)
	if len(reads) == 0 && xauErr != nil {
		return offlineCard("Momentum Agent", "Momentum", "BTC/ETH/XAUUSD", keyMomentum, howTexts[keyMomentum])
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

	// Relative strength vs BTC from the backend (secondary, best-effort).
	if rs, err := a.api.MomentumRS(ctx, []string{"ETH"}); err == nil {
		if item, ok := rs.Items["ETH"]; ok && item.RS7D != nil {
			c.Facts = append(c.Facts, fmt.Sprintf("ETH vs BTC 7d: %+.1f%%", *item.RS7D))
		}
	}
	return c
}

// MomentumAssetCard is the single-asset form: /momentum eurusd.
func (a *Agents) MomentumAssetCard(ctx context.Context, spec assetSpec) Card {
	r, err := a.momentumReadFor(ctx, spec)
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
	if spec.Source == srcYahoo {
		decorateFX(&c)
	}
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
	ema50, ok50 := emaLast(closes, 50)
	ema200, ok200 := emaLast(closes, 200)
	adx, okADX := adxWilder(highs, lows, closes, 14)
	rsi, okRSI := rsiWilder(closes, 14)
	if !ok50 || !ok200 || !okADX || !okRSI {
		c := insufficientCard(spec, "Trend Agent", "Trend", keyTrend, howTexts[keyTrend], "EMA200/ADX(14)")
		c.DataTime = closeTimeOf(candles, spec.Interval)
		return c
	}
	last := closes[len(closes)-1]

	state := classifyTrend(adx, ema50, ema200, last)
	c := Card{
		Agent:      "Trend Agent",
		ShortName:  "Trend",
		Asset:      spec.Display,
		Command:    keyTrend,
		HowItWorks: howTexts[keyTrend],
		DataTime:   closeTimeOf(candles, spec.Interval),
		Verdict:    trendVerdict(state, adx),
	}
	structure := "bearish structure"
	if ema50 > ema200 {
		structure = "bullish structure"
	}
	c.Facts = append(c.Facts,
		fmt.Sprintf("EMA50 %s vs EMA200 %s — %s", trimFloat(ema50), trimFloat(ema200), structure),
		fmt.Sprintf("ADX(14): %.1f · RSI(14): %.1f", adx, rsi),
		fmt.Sprintf("Last %s close: %s", spec.Interval, trimFloat(last)),
	)
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
	c.Facts = append(c.Facts, fmt.Sprintf("Method: swing clusters over %d×%s candles", len(candles), spec.Interval))
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
		parts = append(parts, fmt.Sprintf("%s (%d %s)", srLevelLabel(l), l.Touches, word))
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

type riskResult struct {
	RiskAmount float64
	PerUnit    float64
	Size       float64
	Notional   float64
	Direction  string
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
		Direction:  "SHORT",
	}
	if entry > stop {
		r.Direction = "LONG"
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
	c.Verdict = fmt.Sprintf("%s · size %s units", r.Direction, trimFloat6(r.Size))
	if isExample {
		c.Verdict = "Example — " + c.Verdict
	}
	c.Facts = append(c.Facts,
		fmt.Sprintf("Risk: %s (%.4g%% of %s balance)", usd(r.RiskAmount), args[1], usd(args[0])),
		fmt.Sprintf("Entry %s / stop %s → %s risk per unit", trimFloat(args[2]), trimFloat(args[3]), trimFloat(r.PerUnit)),
		fmt.Sprintf("Position notional: %s", usd(r.Notional)),
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
