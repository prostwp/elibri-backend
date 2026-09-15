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
	// now is the clock for the showcase sweep time (b.at — generated_at and
	// the /showcase Last-Modified), for the FX market-state wording and its
	// stamp (decorateFXAt, the momentum per-asset freshness) and for the
	// whale 24h window when the snapshot has no captured_at. nil = wall
	// clock; tests set it to step sweeps and weekends deterministically.
	now func() time.Time
}

// clock is the composite build time (UTC).
func (a *Agents) clock() time.Time {
	if a.now != nil {
		return a.now().UTC()
	}
	return time.Now().UTC()
}

func NewAgents(api *BackendClient) *Agents {
	return &Agents{api: api, klines: newKlineCache()}
}

const (
	klineInterval = "4h"
	// klineLimit is the Binance window of every agent EXCEPT trend: 250 raw
	// bars → 249 CLOSED bars after the forming bar is dropped, keeping EMA200
	// above its minimum-history requirement.
	klineLimit = 250
	// trendKlineLimit is the Binance window of the Trend Agent and its chart
	// only: 1000 raw → 999 closed. EMA200 is recursive and seeded by an SMA;
	// on 249 bars that seed still weighs in visibly, on 999 it has decayed
	// away, so the value matches the converged EMA200 other tools show.
	// Momentum/S-R/vol keep klineLimit — their outputs must not move.
	trendKlineLimit = 1000
)

// errInsufficientHistory marks a fetch that succeeded but returned too few
// closed bars for the requested indicator set.
var errInsufficientHistory = errors.New("insufficient history")

var fundingSymbols = []string{"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"}

// candlesFor dispatches to the right candle source for an asset and returns
// CLOSED bars only — every bar not yet closed when the candles were FETCHED
// is dropped (item 7: signals must not flicker mid-bar; see candlesWindow for
// why the fetch time, not the request clock). Yahoo symbols go through the
// same 60s cache as Binance ones; the gold fallback (GC=F) kicks in when the
// primary Yahoo symbol fails or comes back empty.
func (a *Agents) candlesFor(ctx context.Context, spec assetSpec) ([]types.OHLCVCandle, error) {
	candles, _, err := a.candlesWindow(ctx, spec, klineLimit)
	return candles, err
}

// trendCandlesFor is candlesFor with the Trend Agent's Binance window
// (trendKlineLimit) — used by TrendCard and /agents/trend/chart only, so the
// card and the chart read the same bars. Yahoo series are unchanged: the
// window only affects Binance-fed trend reads.
func (a *Agents) trendCandlesFor(ctx context.Context, spec assetSpec) ([]types.OHLCVCandle, error) {
	candles, _, err := a.candlesWindow(ctx, spec, trendKlineLimit)
	return candles, err
}

// candlesWindow is the shared body: limit is the number of RAW Binance bars
// (before the forming one is dropped); Yahoo ignores it. Both windows come
// out of the same cached upstream response (klineCache.fetch).
//
// Binance serves a FIXED window: the newest limit−1 closed bars — the raw
// window minus its LAST ROW (250 raw → 249, 1000 raw → 999). A klines answer
// always ends with the bar still forming, so a bar counts as closed only once
// Binance has returned the bar after it. The fetch-time cut alone could not
// see a REST answer that lags the close (fetched after T, bar T still
// forming) or a fast local clock, and a fetch right at a close (no forming
// row) would otherwise read one bar more than the next fetch under the same
// last close. The limit−1 trim is a guard on top.
//
// complete is true only when exactly limit−1 closed bars remain and they are
// contiguous (each opening one interval after the previous): no row skipped
// by the parser, no hole in the source, no short answer. Only then is the
// last close a sound Last-Modified; otherwise the card serves the same body
// without a validator (Card.noValidator). Always false for Yahoo, which keeps
// its untrimmed series and never carries a validator. Builders that serve a
// stamped card read it; the rest use candlesFor.
func (a *Agents) candlesWindow(ctx context.Context, spec assetSpec, limit int) ([]types.OHLCVCandle, bool, error) {
	var candles []types.OHLCVCandle
	var fetchedAt time.Time
	var err error
	if spec.Source == srcYahoo {
		// The cache key carries the interval (B1): a 4h-rebased EURUSD must
		// never collide with the native 1h series.
		key := "yahoo|" + spec.Symbol + "|" + spec.Interval
		candles, fetchedAt, err = a.klines.cached(key, func() ([]types.OHLCVCandle, error) {
			c, ferr := fetchYahooCandlesTF(ctx, spec.Symbol, spec.Interval)
			if (ferr != nil || len(c) == 0) && spec.Fallback != "" {
				return fetchYahooCandlesTF(ctx, spec.Fallback, spec.Interval)
			}
			return c, ferr
		})
	} else {
		candles, fetchedAt, err = a.klines.fetch(ctx, spec.Symbol, spec.Interval, limit)
		// The last row is never read: it is the bar still forming, and a bar
		// is closed only once Binance has returned the NEXT one.
		if n := len(candles); n > 0 {
			candles = candles[: n-1 : n-1]
		}
	}
	if err != nil {
		return nil, false, err
	}
	// Closed = closed when the candles were FETCHED, not by the request
	// clock (see klineCache.cached): a bar that closed after the fetch holds
	// intermediate OHLC in the cache, so it waits for the next fetch (≤ 60s).
	closed := dropUnclosedBars(candles, spec.Interval, fetchedAt)
	if spec.Source == srcYahoo {
		return closed, false, nil
	}
	want := limit - 1
	if n := len(closed); n > want {
		closed = closed[n-want : n : n]
	}
	complete := len(closed) == want && klinesContiguous(closed, spec.Interval)
	return closed, complete, nil
}

// sourceCard tailors the degraded/asset presentation per source: FX cards
// carry the Yahoo footer note and the FX-specific offline wording.
func assetOffline(spec assetSpec, agent, shortName, command, how string) Card {
	c := offlineCard(agent, shortName, spec.Display, command, how)
	c.AssetKey = spec.Key
	if spec.Source == srcYahoo {
		// Per-asset wording: a card headed "GOLD · COMEX GC=F" must not report
		// that the "FX data source" failed, and it must keep the instrument
		// disclosure the spec requires on EVERY card — degraded ones included,
		// where the bare Yahoo credit used to silently drop "not spot XAUUSD".
		c.Verdict = spec.offlineVerdict()
		c.SourceNote = spec.sourceNote()
	}
	return c
}

// decorateFXAt applies the shared FX card furniture at the card's clock
// (Agents.clock — the seam that lets tests render the weekend banner without
// waiting for a weekend): source note and the weekend banner as the first
// fact. DataTime is the caller's job — it must be the close time of the last
// closed bar actually used (item 7).
//
// Every Yahoo card carries no validator (noValidator). No stamp versions its
// body: the market-state wording follows the clock, not the bars; Yahoo can
// publish a bar late or revise the OHLC of a bar already served under the
// same timestamp; and the closed-bar cut runs on its own clock. Any of these
// changes the body under an unchanged stamp — a false 304.
func decorateFXAt(c *Card, now time.Time) {
	c.SourceNote = "data: Yahoo Finance"
	if !isForexOpen(now) {
		c.Facts = append([]string{fxClosedBanner}, c.Facts...)
	}
	c.noValidator = true
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
		AssetKey:   spec.Key,
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
		c.SourceNote = spec.sourceNote()
	}
	return c
}

// howTexts back the [ℹ️ How it works] button. Telegram caps callback alerts
// at 200 characters — keep every entry under that.
var howTexts = map[string]string{
	keyMacro:    "Fixed rules score 5 tradfin lamps (DXY, US 10Y, VIX, S&P 500, Gold) into a 0-100 rule score: above 65 risk-on, below 35 risk-off. A backdrop, not a forecast. RISK-OFF tops the digest.",
	keyWhale:    "Tracks large BTC transfers from the public mempool. These wallets carry no exchange labels, so the card reports transfer activity; exchange in/outflow appears only when labeled.",
	keyFunding:  "Compares perp funding rates across majors. High positive funding = crowded longs (squeeze risk); negative = crowded shorts. Liquidation feed shows where forced exits cluster.",
	keyMomentum: "RSI(14) + MACD histogram. RSI 55+ with positive MACD = bullish; RSI 45- with negative = bearish; else neutral. Crypto on 4h bars, FX and gold on 1h; a 1d scan is available.",
	keyTrend:    "State machine on 4h bars (1h for FX/gold): ADX<20 flat, 20-25 grey zone, ADX 25+ with price and EMA50/200 aligned = confirmed unless swing structure disagrees; else conflict.",
	keySR:       "Clusters swing highs/lows within 0.5% on closed 4h candles (1h FX/gold), 3 strongest per side. Test = a close within 0.25 ATR; reaction or break within 3 candles. 7+ pivots = established.",
	keyVol:      "ATR(14) now vs its 30-bar average. Ratio 1.25+ = volatility expanding; 0.8- = compressed. Measures how far price moves per bar, not which way, and does not confirm a breakout.",
	keyRisk:     "Position size = (balance × risk%) ÷ |entry − stop|. Valid when a 1.0 price move changes one unit's value by 1.0 in account currency (spot); FX lots, futures, CFDs differ.",
	keyFX:       "EMA50 vs EMA200 trend on 1h bars, RSI(14) and 24h change from Yahoo Finance for EURUSD, GBPUSD, USDJPY and gold (COMEX GC=F futures). Weekend closures are flagged.",
	keyDigest:   "Fixed rule: a fresh, fully lit RISK-OFF macro tops; else the strongest fresh CONFIRMED reading among funding, momentum, trend. Their scales are not calibrated.",
	keyTop:      "Fixed rule: a fresh, fully lit RISK-OFF macro tops; else the strongest fresh CONFIRMED reading among funding, momentum, trend. Their scales are not calibrated.",
	keyGold:     goldHow,
	keyNews:     "Crypto themes in CoinDesk and CoinTelegraph headlines plus Reddit posts when reachable. Score 0-100 blends mention growth, volume, sentiment, impact, source spread. From 5 mentions/24h.",
}

// ── Macro ────────────────────────────────────────────────────────────────────

// MacroCard returns the card plus the effective regime for the priority rule
// ("" when the source is offline).
//
// Honesty contract (team-testing defect 2026-08): a regime verdict is a
// knowledge claim, so it needs at least one real lamp behind it. Regime
// "unknown" (or any regime served with zero real lamps — reclassified in
// effectiveMacroRegime) renders an explicit no-data card: UNKNOWN verdict,
// neutral semaphore, facts limited to what IS known (last data date, crypto
// F&G).
//
// Everything the card says is worded in macro_text.go (readability plan,
// 2026-09-15): regime and rule score → main factors → what holds it → data
// dates → breakdown → BTC / gold context → F&G. The backend's generated_idea
// is no longer rendered: the card words the same rule itself, and an older
// backend's causal sentence ("tends to favor crypto") must not leak through a
// version skew.
func (a *Agents) MacroCard(ctx context.Context) (Card, string) {
	m, err := a.api.Macro(ctx)
	if err != nil {
		return offlineCard("Macro Agent", "Macro", "", keyMacro, howTexts[keyMacro]), ""
	}
	return macroCardFrom(m)
}

// macroCardFrom is MacroCard's pure half: one backend payload → the card and
// the effective regime. No network and no clock (every time on the card comes
// from the payload), so tests can sweep any payload.
func macroCardFrom(m *MacroResp) (Card, string) {
	c := macroBaseCard(m, "")

	// A lamp is REAL when the payload carries a value for it; zero real lamps
	// reclassify any regime to unknown. Shared with the asset views — see
	// effectiveMacroRegime.
	regime, _ := effectiveMacroRegime(m)
	c.State = regime // authoritative context for the AI layer (see Card.State)

	if regime == "unknown" {
		// No tradfin inputs at all — say so instead of claiming a regime read.
		// Machine status splits the two absences templates must distinguish:
		// outside the clock week = market_closed, inside it = no_data. The
		// words never say "market closed": the week window knows no holidays.
		c.Emoji, c.Short = emojiNeutral, "unknown (no data)"
		c.Verdict = "UNKNOWN — " + macroNoDataNote(m.TradfinOpen)
		c.Status = macroUnknownStatus(m.TradfinOpen)
		c.Facts = macroUnknownFacts(m, riskModel.scoreName)
		return c, regime
	}

	newMacroView(m, regime).fillGlobal(&c)
	// No AI mood read on this card. The mood read is written from the
	// CoinMarketCap Fear & Greed index and the narrative themes, while this
	// card shows a different F&G feed: a live card printed "Crypto Fear &
	// Greed: 57" directly above an AI paragraph about "a 70/100 reading", and
	// the paragraph was about news themes, not macro. Two numbers under one
	// name on one card is a contradiction a reader cannot resolve.
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
	// No validator, ever: the transfers come from the backend's live table
	// (the newest `limit` rows), not from the snapshot captured_at names — a
	// new transfer pushes an old one out of the list under the same
	// captured_at. The top-3 window is the snapshot's 24h, like the backend's
	// own counters, and a transfer stamped after captured_at is not shown:
	// the body must not carry data newer than its data_as_of. Without a
	// parseable captured_at the window is the 24h up to the card's clock,
	// with the same upper bound: nothing stamped after it is shown.
	c.noValidator = true
	windowEnd, capErr := time.Parse(time.RFC3339, w.CapturedAt)
	if capErr != nil {
		windowEnd = a.clock()
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
		// "potential", like the inflow branch: an exchange outflow can be
		// custody migration or settlement, not only accumulation.
		c.Emoji, c.Verdict, c.Short = emojiBull, "Net OUTFLOW from exchanges — potential accumulation", "outflow (potential accumulation)"
	case btc.Partial && btc.TxCount24h == 0:
		c.Emoji, c.Verdict, c.Short = emojiNeutral, "No large BTC transfers in 24h", "no large transfers"
	case btc.Partial:
		// The BTC feed (public mempool) never labels exchange wallets, so the
		// backend's net flow is structurally $0 and its direction "neutral".
		// That is "not measurable", not "balanced": the old verdict "Flows
		// balanced over 24h" asserted a finding the data cannot produce.
		c.Emoji = emojiNeutral
		c.Verdict = fmt.Sprintf("%d large BTC transfers in 24h — exchange direction not measurable", btc.TxCount24h)
		c.Short = fmt.Sprintf("%d large tx, direction n/a", btc.TxCount24h)
	default:
		c.Emoji, c.Verdict, c.Short = emojiNeutral, "Flows balanced over 24h", "balanced"
	}
	directional := btc != nil && (btc.Direction == "inflow" || btc.Direction == "outflow")
	if btc != nil {
		if btc.Partial && !directional {
			c.Facts = append(c.Facts, "Net exchange flow: not measurable (these BTC wallets carry no exchange labels)")
		} else {
			flowLine := fmt.Sprintf("Net flow 24h: %s (%d large tx)", usd(btc.NetFlowUSD24h), btc.TxCount24h)
			if btc.Partial {
				flowLine += " — partial data, labeled wallets only"
			}
			c.Facts = append(c.Facts, flowLine)
		}
		// Baseline comparison — only when the payload actually carries a
		// prior-24h figure (a zero baseline means "no snapshot to compare").
		if btc.NetFlowPrev24h != nil && *btc.NetFlowPrev24h != 0 {
			base := fmt.Sprintf("Prior 24h net flow: %s", usd(*btc.NetFlowPrev24h))
			if btc.FlowPct != nil {
				base += fmt.Sprintf(" → %+.0f%% change", *btc.FlowPct)
			}
			c.Facts = append(c.Facts, base)
		}
		if btc.Confidence > 0 && directional {
			conf := btc.Confidence
			c.Confidence = &conf
		}
	}

	// Top-3 BTC transfers of the snapshot's last 24h by USD size.
	cutoff := windowEnd.Add(-24 * time.Hour)
	var recent []WhaleTransfer
	for _, t := range w.Transfers {
		if t.Chain == "BTC" && t.Timestamp.After(cutoff) && !t.Timestamp.After(windowEnd) {
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
		// "seen", not a transaction time: the mempool feed carries no tx
		// timestamp, so the backend stamps each unconfirmed tx with its poll
		// time (whale/source_mempool.go). That is when we saw it broadcast —
		// still more usable than "neutral · unlabeled wallet" on every line.
		line := fmt.Sprintf("%s BTC ≈ %s · seen %s UTC",
			trimFloat(t.AmountNative), usd(t.AmountUSD), t.Timestamp.UTC().Format("Jan 2 15:04"))
		if t.Exchange != "" && (t.Direction == "inflow" || t.Direction == "outflow") {
			line += " · " + t.Direction + " · " + t.Exchange
		}
		c.Facts = append(c.Facts, line)
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
		// The AI idea is generated and cached by the backend per narrative
		// and hour, failures uncached — it can appear or change under the
		// same captured_at. No validator.
		noValidator: true,
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
			c.Facts = append(c.Facts, fmt.Sprintf("%d. %s — %s/24h", i+1, item.Narrative, mentionsWord(item.MentionCount)))
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
		line := fmt.Sprintf("%d. %s — %s · score %d · %s/24h", i+1, item.Narrative, item.Stage, item.TrendScore, mentionsWord(item.MentionCount))
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

// mentionsWord renders "1 mention" / "N mentions" — the radar's lists
// printed "1 mentions/24h" on most live days.
func mentionsWord(n int) string {
	if n == 1 {
		return "1 mention"
	}
	return fmt.Sprintf("%d mentions", n)
}

// ── Funding ──────────────────────────────────────────────────────────────────

// Funding-rate verdict thresholds (8h rate, absolute):
// +0.03% and above = longs crowded; -0.01% and below = shorts crowded
// (negative funding is rarer, so its threshold is tighter). In between =
// balanced. Deviation: see fundingDeviation.
const (
	fundingLongsCrowded  = 0.0003
	fundingShortsCrowded = -0.0001
	// fundingThresholdScore is the digest score of a rate sitting exactly on
	// its side's threshold. 30 keeps the positive side identical to the old
	// |rate|/0.10% scale (+0.03% → 30, +0.10% → 100).
	fundingThresholdScore = 30
)

// fundingDeviation is the digest ranking score of the widest rate, measured
// against the threshold of ITS OWN side, so crossing either threshold scores
// the same 30. The old symmetric |rate|/0.10% scale ignored the asymmetric
// thresholds: −0.010% ("shorts crowded") scored 10 while +0.029% ("balanced")
// scored 29. Ranking only — the verdict switch in FundingCard is unchanged.
func fundingDeviation(widest float64) int {
	th := fundingLongsCrowded
	if widest < 0 {
		th = -fundingShortsCrowded
	}
	return clampInt(int(math.Round(math.Abs(widest)/th*fundingThresholdScore)), 0, 100)
}

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
		// Rates, the liquidation feed, the 1h window from now and the BTC
		// magnet price are separate reads; the request-time DataTime (one
		// second of header resolution) is not a version of that body.
		noValidator: true,
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
			c.confirmed = true
		case widest <= fundingShortsCrowded:
			c.Emoji, c.Verdict, c.Short = emojiBull, "Shorts crowded — squeeze fuel above", "shorts crowded"
			c.confirmed = true
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
		c.Deviation = fundingDeviation(widest)
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
// neutral. The thresholds are momentumBullRSI / momentumBearRSI; everything
// the card says about the read is worded in momentum_text.go.
//
// REGULATORY LANGUAGE (team review batch 2): verdicts are analytical READINGS,
// never trade instructions — "bullish"/"bearish", NOT "buy"/"sell". The
// product must read as analytics; advice-words are banned from every card,
// one-liner and envelope. Factual market-mechanics wording ("longs pay
// shorts", "crowded longs", "sell pressure" as a flow description) stays.
func momentumVerdict(rsi, macdHist float64) string {
	switch {
	case rsi >= momentumBullRSI && macdHist > 0:
		return momentumBullish
	case rsi <= momentumBearRSI && macdHist < 0:
		return momentumBearish
	default:
		return "neutral"
	}
}

// momentumRead computes one asset's RSI/MACD snapshot.
type momentumRead struct {
	name     string
	verdict  string
	rsi      float64
	hist     float64
	source   string
	interval string    // bar size the read was taken on ("4h", "1h")
	closeAt  time.Time // close time of the last closed bar used
	// complete: the read came from a full, contiguous Binance window (see
	// candlesWindow) — set by momentumReadFor, validator use only.
	complete bool
}

// momentumReadFromCandles computes one asset's snapshot from an already
// fetched series, so a read and any derived facts (volume line) always come
// from the SAME bars — a transient refetch can't produce a half-coherent card.
// Non-finite indicator output (overflowing or Inf prices) degrades as
// insufficient history instead of reaching the text and the JSON encoder.
func momentumReadFromCandles(spec assetSpec, candles []types.OHLCVCandle) (momentumRead, error) {
	closes := closesOf(candles)
	rsi, okRSI := rsiWilder(closes, 14)
	_, _, hist, okMACD := macdLast(closes)
	if !okRSI || !okMACD || !isFinite(rsi) || !isFinite(hist) {
		return momentumRead{name: spec.Display, source: spec.Source}, errInsufficientHistory
	}
	return momentumRead{
		name:     spec.Display,
		verdict:  momentumVerdict(rsi, hist),
		rsi:      rsi,
		hist:     hist,
		source:   spec.Source,
		interval: spec.Interval,
		closeAt:  closeTimeOf(candles, spec.Interval),
	}, nil
}

func (a *Agents) momentumReadFor(ctx context.Context, spec assetSpec) (momentumRead, error) {
	candles, complete, err := a.candlesWindow(ctx, spec, klineLimit)
	if err != nil {
		return momentumRead{}, err
	}
	r, err := momentumReadFromCandles(spec, candles)
	r.complete = complete
	return r, err
}

// momentumRankScore is one read's digest ranking score: |RSI−50|×2 for a
// CONFIRMED bullish/bearish read, 0 otherwise. A neutral read (RSI and MACD
// disagree, or RSI inside 45–55) used to score on RSI distance alone and took
// the top slot with a card where every asset said NEUTRAL. Ranking only — the
// verdict rule (momentumVerdict) is unchanged.
func momentumRankScore(r momentumRead) int {
	if r.verdict != momentumBullish && r.verdict != momentumBearish {
		return 0
	}
	return clampInt(int(math.Abs(r.rsi-50)*2), 0, 100)
}

func momentumEmoji(verdict string) string {
	switch verdict {
	case momentumBullish:
		return emojiBull
	case momentumBearish:
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

// momentumAssetFrom classifies one fetched asset for a multi-asset card.
func momentumAssetFrom(spec assetSpec, r momentumRead, err error) momentumAsset {
	switch {
	case err == nil:
		return momentumAsset{spec: spec, read: r, status: statusOK}
	case errors.Is(err, errInsufficientHistory):
		return momentumAsset{spec: spec, read: r, status: statusInsufficientHistory}
	default:
		return momentumAsset{spec: spec, status: statusSourceOffline}
	}
}

// momentumNoReading applies the zero-readings outcome shared by the overview
// and the scans (review fix 3): any answered-but-short source →
// insufficient_history with one line per asset; only an all-dead sweep is
// source_offline. ok=false when at least one asset read.
func momentumNoReading(c Card, assets []momentumAsset) (Card, bool) {
	insufficient := false
	for _, a := range assets {
		switch a.status {
		case statusOK:
			return c, false
		case statusInsufficientHistory:
			insufficient = true
		}
	}
	if insufficient {
		c.Emoji = emojiNeutral
		c.Verdict = "Insufficient history — no verdict"
		c.Short = "insufficient history"
		c.Offline = true
		c.Status = statusInsufficientHistory
		momentumDegraded(&c, assets)
		return c, true
	}
	off := offlineCard("Momentum Agent", "Momentum", c.Asset, keyMomentum, howTexts[keyMomentum])
	for _, a := range assets {
		off.Results = append(off.Results, assetResult(a.spec.Display, a.status))
	}
	return off, true
}

// MomentumCard is the default multi-asset card: BTC + ETH (Binance 4h) and
// XAUUSD (Yahoo 1h, GC=F fallback). The header is a counter over the three
// reads (momentumHeader), each asset carries its own timeframe, bar time and
// freshness, and volume / ETH-vs-BTC follow as labelled context.
func (a *Agents) MomentumCard(ctx context.Context) Card {
	c := Card{
		Agent:      "Momentum Agent",
		ShortName:  "Momentum",
		Asset:      "BTC/ETH/XAUUSD",
		Command:    keyMomentum,
		HowItWorks: howTexts[keyMomentum],
		DataTime:   time.Now().UTC(), // narrowed below to the OLDEST closed bar used
		// Three series, the backend RS read and the clock-derived freshness:
		// the oldest bar (DataTime) can stay put while the body changes.
		noValidator: true,
	}
	var assets []momentumAsset
	var btcCandles []types.OHLCVCandle // the exact series the BTC read used
	for _, key := range []string{"btc", "eth"} {
		spec := assetTable[key]
		candles, err := a.candlesFor(ctx, spec)
		if err != nil {
			assets = append(assets, momentumAssetFrom(spec, momentumRead{}, err))
			continue
		}
		r, rerr := momentumReadFromCandles(spec, candles)
		if key == "btc" && rerr == nil {
			btcCandles = candles
		}
		assets = append(assets, momentumAssetFrom(spec, r, rerr))
	}
	xau, xauErr := a.momentumReadFor(ctx, xauSpec)
	assets = append(assets, momentumAssetFrom(xauSpec, xau, xauErr))
	if out, none := momentumNoReading(c, assets); none {
		return out
	}

	composeMomentum(&c, assets, "", a.clock())
	if xauErr == nil {
		c.SourceNote = "XAUUSD data: Yahoo Finance"
	}
	// Context, not part of the reading. Volume from the SAME series the BTC
	// read used — no refetch, so it never sits beside a missing BTC read.
	if ratio, ok := volRatio20(btcCandles); ok {
		c.Facts = append(c.Facts, momentumVolumeContext("BTC "+btcSpec.Interval+" volume", ratio))
	}
	// ETH-vs-BTC return gap from the backend (secondary, best-effort).
	if rs, err := a.api.MomentumRS(ctx, []string{"ETH"}); err == nil {
		if item, ok := rs.Items["ETH"]; ok {
			if line := momentumRSContext(item); line != "" {
				c.Facts = append(c.Facts, line)
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
	candles, complete, err := a.candlesWindow(ctx, spec, klineLimit)
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
	// Binance assets get the volume read from the SAME series as the RSI/MACD
	// math. Yahoo FX volume is null throughout, so no line over a fake 0×.
	ratio, volOK := volRatio20(candles)
	c := momentumAssetCardFrom(spec, r, ratio, volOK, a.clock())
	if !complete { // partial Binance window: see candlesWindow
		c.noValidator = true
	}
	return c
}

// momentumAssetCardFrom is MomentumAssetCard's pure half: one read at the
// card's clock → the card. Line order: why (the checklist) → what turns or
// keeps the reading → freshness when late → method → context.
func momentumAssetCardFrom(spec assetSpec, r momentumRead, volRatio float64, volOK bool, now time.Time) Card {
	tf := candleWord(spec.Interval)
	fresh := momentumFreshness(spec.Source, spec.Interval, r.closeAt, now)
	c := Card{
		Agent:      "Momentum Agent",
		ShortName:  "Momentum",
		Asset:      spec.Display,
		AssetKey:   spec.Key,
		Command:    keyMomentum,
		HowItWorks: howTexts[keyMomentum],
		DataTime:   r.closeAt,
		Emoji:      momentumEmoji(r.verdict),
		Verdict:    fmt.Sprintf("%s · %s — %s", strings.ToUpper(momentumWord(r.verdict)), tf, momentumWhy(r.rsi, r.hist)),
		Short:      r.verdict,
		Deviation:  momentumRankScore(r),
		confirmed:  r.verdict == momentumBullish || r.verdict == momentumBearish,
	}
	c.Facts = append(c.Facts, "Why: "+momentumChecklist(r.rsi, r.hist))
	for _, d := range momentumDirOrder(r.rsi, r.hist) {
		c.Facts = append(c.Facts, momentumTurnLine(r.rsi, r.hist, d))
	}
	if fresh == momentumDataDelayed {
		c.Facts = append(c.Facts, fmt.Sprintf("Data delayed: the last closed %s bar (%s) is over 2 bars old", tf, momentumBarTime(r.closeAt)))
		// The wording follows the clock, not the bars: under the same last
		// close the body changes from on time to delayed, so no stamp may
		// version it (a conditional GET would answer 304 for a changed body).
		c.noValidator = true
	}
	c.Facts = append(c.Facts, fmt.Sprintf("Read on closed %s candles: RSI(14) and the MACD(12,26,9) histogram", tf))
	if spec.Source == srcYahoo && spec.Interval == "4h" {
		c.Facts = append(c.Facts, yahooAgg4hNote) // B1: disclose the 1h→4h merge
	}
	if spec.Source == srcBinance && volOK {
		c.Facts = append(c.Facts, momentumVolumeContext("volume", volRatio)+" ("+tf+")")
	}
	c.Results = []AssetResult{momentumResult(spec, r, fresh)}
	c.Blocks = momentumBlocks(spec.Display, spec.Interval, r.rsi, r.hist)
	if spec.Source == srcYahoo {
		decorateFXAt(&c, now)
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

	assets := make([]momentumAsset, len(specs))
	var wg sync.WaitGroup
	for i, spec := range specs {
		wg.Add(1)
		go func(i int, spec assetSpec) {
			defer wg.Done()
			r, err := a.momentumReadFor(ctx, spec)
			assets[i] = momentumAssetFrom(spec, r, err)
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
	if out, none := momentumNoReading(c, assets); none {
		return out
	}

	composeMomentum(&c, assets, tf, a.clock())
	anyYahoo, agg4h := false, false
	for _, spec := range specs {
		if spec.Source == srcYahoo {
			anyYahoo = true
			agg4h = agg4h || spec.Interval == "4h"
		}
	}
	if agg4h {
		c.Facts = append(c.Facts, yahooAgg4hNote)
	}
	if anyYahoo {
		c.SourceNote = "FX/gold data: Yahoo Finance"
	}
	// Validator: only a single Binance asset read from a complete window keeps
	// one — its closed bars version the whole body. Several assets are a
	// composite (the oldest bar can stay put while another asset, or a
	// failure/recovery, changes the body), a Yahoo asset has no sound stamp at
	// all (see decorateFXAt), and a partial Binance window neither (see
	// candlesWindow). A data-delayed read follows the clock, not the bars (see
	// momentumAssetCardFrom). With one asset, a reading means assets[0] is it.
	if len(specs) > 1 || anyYahoo || !assets[0].read.complete || c.Results[0].Freshness == momentumDataDelayed {
		c.noValidator = true
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
	return fxCardFromReads(a.fxReads(ctx))
}

// fxCardFromReads is the pure half of FXCard: reads already computed → the
// card. Split out so a caller that ALREADY holds a sweep's fx reads (the
// landing showcase reuses gather's) rebuilds the exact same card without a
// second round of Yahoo fetches — same builder, no duplicated logic.
func fxCardFromReads(reads []fxRead) Card {
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
	c := fxOverviewCard(reads, isForexOpen(time.Now()), latest)
	// Four series plus the banner: an older pair can update, drop out or
	// recover while the newest close stays put — no validator.
	c.noValidator = true
	return c
}

// ── Trend ────────────────────────────────────────────────────────────────────

const (
	trendFlat     = "flat"
	trendGrey     = "grey"
	trendUp       = "up"
	trendDown     = "down"
	trendConflict = "conflict"
)

// Trend state-machine ADX(14) thresholds. Below trendADXRead there is no trend
// to read (flat); from it up to trendADXConfirm is the grey zone; AT or above
// trendADXConfirm the EMA alignment decides. Named so the card text prints the
// same numbers the rule uses — the card used to say "confirms above 25" about
// a rule that confirms AT 25.
const (
	trendADXRead    = 20
	trendADXConfirm = 25
)

// classifyTrend is the /trend state machine:
//
//	ADX < 20            → flat
//	20 ≤ ADX < 25       → grey zone
//	ADX ≥ 25, EMA50>EMA200, close>EMA50 → confirmed uptrend
//	ADX ≥ 25, EMA50<EMA200, close<EMA50 → confirmed downtrend
//	ADX ≥ 25, otherwise → indicator conflict
func classifyTrend(adx, ema50, ema200, closePrice float64) string {
	switch {
	case adx < trendADXRead:
		return trendFlat
	case adx < trendADXConfirm:
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

// trendInvalidationFact words the invalidation level for the card, or returns
// "" when there is nothing to invalidate.
//
// Only a confirmed trend has a reading that can be invalidated. Unconfirmed
// cards (flat/grey/conflict, structure-demoted included) print no level line
// at all: the old "reference: below X" line sat under a grey gold card whose
// price was already under X, i.e. a level the market had crossed. The number
// itself still ships in levels.invalidation — that is a JSON contract and only
// the card TEXT changed.
//
// Two things the old wording got wrong, fixed here: the level is the EMA
// cluster ± 1 ATR, not swing structure, so what it invalidates is the trend
// READING ("the structure is broken" sat under "structure: no reading"); and
// it is checked on a CLOSED candle of the agent's timeframe — the agent only
// ever reads closed bars, so a wick through the level does not count.
func trendInvalidationFact(state string, inv float64, side, tf string, last float64) string {
	if state != trendUp && state != trendDown {
		return ""
	}
	prep := "under"
	if side == "above" {
		prep = "over"
	}
	return fmt.Sprintf("Invalidated by a closed %s candle %s %s (%s, 1 ATR %s the EMA cluster)",
		candleWord(tf), side, trimFloat(inv), signedPct(last, inv), prep)
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

// trendVerdict words each state.
//
// Analytic statements only, never advice — the Этап 6 review flagged "Trading
// not advised" and "stand aside" as recommendations, and a recommendation to
// abstain is still a recommendation. The disclaimer in the footer does not
// change what the body of the card says. What the agent may state is what it
// READ: there is no trend to read, the indicators do not agree.
//
// tf is the agent's timeframe (spec.Interval), printed beside the state so a
// reader never has to guess which candles the reading is about; "" omits it.
// No verdict prints the ADX number: the card prints it exactly once, in the
// "Why:" checklist, always through adxShown. Flat states its threshold (20)
// here; the checklist states the confirming one (25).
func trendVerdict(r trendRead, tf string) string {
	head := func(s string) string {
		if tf == "" {
			return s
		}
		return s + " · " + tf
	}
	switch r.State {
	case trendFlat:
		return fmt.Sprintf("%s — no trend to read (ADX under %d)", head("Flat"), trendADXRead)
	case trendGrey:
		// A demoted state is grey for a different reason than a forming
		// trend: ADX and the EMAs agree, the swing structure does not. The
		// reason rides in the verdict so it cannot fall off the card.
		if r.StructureDemoted != "" {
			return head("Grey zone") + " — not confirmed: " + r.StructureDemoted
		}
		return head("Grey zone") + " — trend forming, not confirmed"
	case trendUp:
		return head("Confirmed UPTREND")
	case trendDown:
		return head("Confirmed DOWNTREND")
	default:
		return head("Indicator conflict") + " — " + conflictReason(r)
	}
}

// trendRead is the Trend Agent's state machine evaluated over one window of
// CLOSED bars — everything the verdict depends on, and nothing about how it is
// rendered.
//
// It exists so the state machine has exactly ONE definition. Anything that
// needs to know what the agent WOULD say (the gold agent, the history-run
// harness) calls this instead of restating the rules, because a restated rule
// silently stops matching the day someone edits the original. That is not
// hypothetical: the measurement harness for the structure gate did exactly
// that on 2026-08-27 and reported the defect it was built to measure as
// costing nothing.
type trendRead struct {
	OK               bool   // false → window too short for the indicator set
	State            string // flat | grey | up | down | conflict (post-gate)
	Raw              string // classifyTrend's answer BEFORE the structure gate
	Structure        string // "" (unreadable) | hh_hl | lh_ll | mixed
	StructureDemoted string // demotion wording; "" when the gate did not fire
	ADX, RSI         float64
	EMA20, EMA50     float64
	EMA200, Last     float64
	// ATR is ATR(14) at the last closed bar — the one value the invalidation
	// level is built from (invalidationFor). 0 on a degenerate series with no
	// range; the card then shows no level.
	ATR float64
}

// trendReadFinite reports whether every number a reading is built from — and
// the invalidation level derived from them — is finite. A NaN ADX fails every
// comparison in classifyTrend and falls through to "conflict"; an Inf level
// breaks JSON encoding. Neither is a reading, so trendReadOf refuses both.
func trendReadFinite(r trendRead) bool {
	inv, _ := invalidationFor(r.State, r.EMA50, r.EMA200, r.ATR)
	for _, v := range []float64{r.ADX, r.RSI, r.EMA20, r.EMA50, r.EMA200, r.Last, r.ATR, inv} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return false
		}
	}
	return true
}

// trendReadOf runs the state machine over candles (oldest first, CLOSED bars
// only — the caller drops the forming bar). Pure: no clock, no network.
// trendMinBars is the history floor for a regime reading.
//
// EMA200 alone is satisfied at exactly 200 bars, and the agent used to accept
// that — while the history run only ever took decisions from bar 220. States
// produced on 200-219 bars were therefore live-reachable and completely
// unrepresented in the report backing them. One floor now, shared by both:
// 200 for EMA200 plus 14 for ADX plus a small settling margin.
const trendMinBars = 220

func trendReadOf(candles []types.OHLCVCandle) trendRead {
	if len(candles) < trendMinBars {
		return trendRead{}
	}
	closes := closesOf(candles)
	highs, lows := highsLowsOf(candles)
	ema20, ok20 := emaLast(closes, 20)
	ema50, ok50 := emaLast(closes, 50)
	ema200, ok200 := emaLast(closes, 200)
	adx, okADX := adxWilder(highs, lows, closes, 14)
	rsi, okRSI := rsiWilder(closes, 14)
	if !ok20 || !ok50 || !ok200 || !okADX || !okRSI {
		return trendRead{}
	}
	last := closes[len(closes)-1]

	// HH/HL swing structure (B3, review fix 7) — the SAME swing points the
	// S/R agent clusters (wing 3), classified over the last six ALTERNATING
	// pivots. It PARTICIPATES in the state machine: an EMA/ADX-confirmed
	// direction with a contradicting or mixed pivot structure demotes to the
	// grey zone — the card must never say "Confirmed UPTREND" over LH/LL
	// swings.
	//
	// An empty read ("") does NOT demote, and that is the load-bearing half of
	// the rule. "" means the structure could not be READ — too few pivots (a
	// monotone series, the strongest trend there is) or a non-alternating
	// window (a double top, two pushes). Neither is evidence against a trend.
	// Only "mixed" — pivots that alternate and still refuse to line up — is a
	// real disagreement, and only it demotes alongside the explicit opposite.
	//
	// The two were conflated until 2026-08-27, which cost 33 percentage points
	// of trend confirmations platform-wide; see hhhlStructure's comment and
	// structure_gate_measure_test.go.
	swingHighs, swingLows := swingPointsIdx(highs, lows, 3)
	structure := hhhlStructure(swingHighs, swingLows)

	state := classifyTrend(adx, ema50, ema200, last)
	raw := state
	demoted := structureDemotion(state, structure)
	if demoted != "" {
		state = trendGrey
	}
	atr := 0.0
	if atrSeries := atrSeriesWilder(highs, lows, closes, 14); len(atrSeries) > 0 {
		atr = atrSeries[len(atrSeries)-1]
	}
	r := trendRead{
		OK: true, State: state, Raw: raw,
		Structure: structure, StructureDemoted: demoted,
		ADX: adx, RSI: rsi,
		EMA20: ema20, EMA50: ema50, EMA200: ema200, Last: last,
		ATR: atr,
	}
	// Non-finite inputs (overflowing prices) are no reading at all: the card
	// degrades instead of printing "conflict" off a NaN or an Inf level.
	if !trendReadFinite(r) {
		return trendRead{}
	}
	return r
}

// structureDemotion is the structure gate: for an EMA/ADX-confirmed direction,
// the reason the swing pivots demote it to grey, or "" when they do not argue
// against it. Only a readable disagreement demotes — the explicit opposite
// structure or "mixed"; an unreadable window ("") never does (see trendReadOf).
// Any other state has no direction to demote and gets "".
//
// The returned wording is reader-facing: it completes the grey verdict
// "Grey zone · 4h — not confirmed: <reason>".
func structureDemotion(direction, structure string) string {
	switch {
	case direction == trendUp && structure == "lh_ll",
		direction == trendDown && structure == "hh_hl":
		return "swing structure against the trend"
	case (direction == trendUp || direction == trendDown) && structure == "mixed":
		return "swing structure not aligned"
	}
	return ""
}

// Confirmed reports whether the state machine committed to a direction.
func (r trendRead) Confirmed() bool { return r.State == trendUp || r.State == trendDown }

func (a *Agents) TrendCard(ctx context.Context, spec assetSpec) Card {
	candles, complete, err := a.candlesWindow(ctx, spec, trendKlineLimit)
	if err != nil {
		return assetOffline(spec, "Trend Agent", "Trend", keyTrend, howTexts[keyTrend])
	}
	r := trendReadOf(candles)
	if !r.OK {
		c := insufficientCard(spec, "Trend Agent", "Trend", keyTrend, howTexts[keyTrend], "EMA200/ADX(14)")
		c.DataTime = closeTimeOf(candles, spec.Interval)
		return c
	}
	c := trendCardFrom(r, spec, closeTimeOf(candles, spec.Interval))
	if spec.Source == srcYahoo {
		decorateFXAt(&c, a.clock())
	}
	if !complete { // partial Binance window: see candlesWindow
		c.noValidator = true
	}
	return c
}

// trendCardFrom is the pure half of TrendCard: one evaluated read (its ATR
// included) → the card. No clock, no network, so every state renders under
// test exactly as it does live.
//
// Line order: where price is → what keeps or changes the reading → why. Every
// sentence comes from trendView, which derives it from the read's own fields
// (trend_text.go) — no rule is restated in wording.
func trendCardFrom(r trendRead, spec assetSpec, dataTime time.Time) Card {
	state := r.State
	c := Card{
		Agent:      "Trend Agent",
		ShortName:  "Trend",
		Asset:      spec.Display,
		AssetKey:   spec.Key,
		Command:    keyTrend,
		HowItWorks: howTexts[keyTrend],
		DataTime:   dataTime,
		Verdict:    trendVerdict(r, spec.Interval),
		State:      state, // grey/flat/conflict = confirmation WITHHELD
	}
	v := trendView{r: r, tf: spec.Interval}
	// Pullback zone (B3): the EMA20-EMA50 band, confirmed trends only. A
	// structure-demoted state is NOT confirmed, so it offers no zone.
	zone := pullbackZoneFor(state, r.EMA20, r.EMA50)
	// Levels exist ONLY for a confirmed trend: the pullback zone, and the
	// direction-aware invalidation level (review fix 8: downtrends break
	// UPWARD, above the EMA cluster). An unconfirmed reading has nothing to
	// invalidate, so neither the card text nor the JSON carries a level there
	// (product decision 2026-09-15; levels used to ship a number for every
	// state). ATR(14) is guaranteed by the EMA200 history gate; the >0 guard
	// keeps a degenerate flat series from producing a fake level.
	if r.Confirmed() {
		lv := TrendLevels{PullbackZone: zone}
		if r.ATR > 0 {
			v.atr = r.ATR
			v.inv, v.invSide = invalidationFor(state, r.EMA50, r.EMA200, r.ATR)
			inv := v.inv
			lv.Invalidation, lv.InvalidationSide = &inv, v.invSide
		}
		c.Levels = lv
	}
	c.Facts = v.facts()
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
		c.Emoji, c.Short = emojiNeutral, "flat — no trend"
	}
	c.Blocks = v.blocks(c.Short)
	c.trendConclusion = v.neutralConclusion(spec.Display)
	// Confirmed trends count double toward the priority rule. Unconfirmed
	// states keep the raw ADX here, but the digest ranking (and the showcase
	// fallback, which reuses it) scores them 0 and never lets them outrank a
	// confirmed reading (rankScore / pickTop in priority.go).
	c.confirmed = r.Confirmed()
	if state == trendUp || state == trendDown {
		c.Deviation = clampInt(int(r.ADX)*2, 0, 100)
	} else {
		c.Deviation = clampInt(int(r.ADX), 0, 100)
	}
	return c
}

// ── Support / Resistance ─────────────────────────────────────────────────────

// srStrongTouches is the touch count from which a clustered level counts as
// STRONG. Below it a level is a candidate; at 7+ distinct swing touches the
// market has respected the price often enough to call the level established.
// Surfaced beside the touch numbers on the card (batch-2 thresholds rule).
// Swing/cluster parameters, named so the gold agent's untruncated search runs
// on exactly the same clustering as the S/R card rather than a second set of
// magic numbers.
const (
	srWing   = 3
	srTolPct = 0.5
)

const srStrongTouches = 7

func (a *Agents) SRCard(ctx context.Context, spec assetSpec) Card {
	candles, complete, err := a.candlesWindow(ctx, spec, klineLimit)
	if err != nil {
		return assetOffline(spec, "S/R Agent", "S/R", keySR, howTexts[keySR])
	}
	c := srCardOf(spec, candles, a.clock())
	if !complete { // partial Binance window: see candlesWindow
		c.noValidator = true
	}
	return c
}

// srCardOf is SRCard after the fetch: every card path for a set of closed
// candles. now only drives the FX weekend banner and its stamp (decorateFXAt).
func srCardOf(spec assetSpec, candles []types.OHLCVCandle, now time.Time) Card {
	if len(candles) < 20 { // too few closed bars for meaningful swings
		c := insufficientCard(spec, "S/R Agent", "S/R", keySR, howTexts[keySR], "swing detection")
		c.DataTime = closeTimeOf(candles, spec.Interval)
		return c
	}
	sup, res := supportResistance(candles, srWing, srTolPct)
	last := candles[len(candles)-1].Close
	// Rendered lines go nearest first; the JSON levels keep the documented
	// strength order (SRLevels contract), each point carrying display_rank and
	// strength_rank so the two orders never diverge silently (sr_text.go).
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
		c := srNoLevelsCard(spec, len(sh)+len(sl), last, len(candles), closeTimeOf(candles, spec.Interval))
		if spec.Source == srcYahoo {
			decorateFXAt(&c, now)
		}
		return c
	}
	c := srCardFrom(spec, sup, res, last, len(candles), closeTimeOf(candles, spec.Interval),
		srVolumeComparable(candles, srWing, srTolPct))
	if spec.Source == srcYahoo {
		decorateFXAt(&c, now)
	}
	return c
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
	candles, complete, err := a.candlesWindow(ctx, spec, klineLimit)
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
	if avg <= 0 {
		// A zero 30-bar ATR baseline makes the ratio 0/0. It used to be left
		// at 0.0, which volState reads as CONFIRMED COMPRESSION — a claim
		// manufactured out of missing data. There is nothing to compare
		// against, so the card says exactly that.
		c := insufficientCard(spec, "Volatility Agent", "Volatility", keyVol, howTexts[keyVol],
			"ATR(14) 30-bar baseline (the baseline is flat zero — nothing to compare against)")
		c.DataTime = closeTimeOf(candles, spec.Interval)
		return c
	}
	ratio := now / avg
	state := volState(ratio)
	c := Card{
		Emoji:      emojiNeutral,
		Agent:      "Volatility Agent",
		ShortName:  "Volatility",
		Asset:      spec.Display,
		AssetKey:   spec.Key,
		Command:    keyVol,
		HowItWorks: howTexts[keyVol],
		DataTime:   closeTimeOf(candles, spec.Interval),
		Levels:     VolLevels{ExpansionRatio: ratio}, // unrounded, envelope "levels"
		State:      state,
	}
	switch state {
	case volExpanding:
		// "ranges widening", not "breakout conditions": ATR measures bar
		// size, it does not see a level being broken.
		c.Verdict, c.Short = "Volatility EXPANDING — bar ranges widening", fmt.Sprintf("expanding %.2f×", ratio)
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
		decorateFXAt(&c, a.clock())
	}
	if !complete { // partial Binance window: see candlesWindow
		c.noValidator = true
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

// riskExampleValues is the single worked example the calculator shows when it
// has no user numbers — shared by the Telegram /risk with no args, the
// self-test and the landing showcase, so the three can never drift apart. It
// is the parsed twin of bot.go's riskExampleArgs (the [Try: …] button's
// string form); TestRiskExampleValuesMatchButtonArgs pins them together.
var riskExampleValues = []float64{10000, 1, 64000, 62500}

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
