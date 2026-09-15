package demobot

import (
	"html"
	"math"
	"strconv"
	"strings"
	"time"
)

// Semaphore emojis — the only three the card contract allows.
const (
	emojiBull    = "🟢"
	emojiBear    = "🔴"
	emojiNeutral = "⚪"
)

// ── machine-readable status ──────────────────────────────────────────────────

// cardStatus is the machine-readable state behind a card, threaded from the
// builders to the HTTP envelope's ok/reason pair (docs/demobot-http.md) so
// templates can branch on WHY a reading is absent instead of sniffing verdict
// strings. The zero value is statusOK — only degraded paths set it.
type cardStatus int

const (
	statusOK                  cardStatus = iota // real reading produced
	statusSourceOffline                         // data source unreachable (or the headline read's source is)
	statusInsufficientHistory                   // source alive, too few closed bars for the indicator set
	statusBelowThreshold                        // narrative radar warming up: thin mention base / no snapshots
	statusNoData                                // upstream alive but nothing to read: macro unknown in the open window, no flow snapshot
	statusMarketClosed                          // macro unknown because the tradfin market is closed
)

// reason maps a status to its JSON enum value; "" for statusOK (the envelope
// serves null instead).
func (s cardStatus) reason() string {
	switch s {
	case statusSourceOffline:
		return "source_offline"
	case statusInsufficientHistory:
		return "insufficient_history"
	case statusBelowThreshold:
		return "below_threshold"
	case statusNoData:
		return "no_data"
	case statusMarketClosed:
		return "market_closed"
	default:
		return ""
	}
}

// ── machine-readable levels ──────────────────────────────────────────────────
//
// Card.Levels is the numeric companion to the facts strings for agents whose
// reading IS a set of price levels: raw computed float64s at full precision,
// never display-rounded. The HTTP envelope serves it verbatim as "levels";
// Telegram rendering ignores it. nil for agents without levels.

// PullbackZone is the EMA20-EMA50 band served ONLY in confirmed-trend states
// (B3). From = EMA20 (the shallow edge, nearer price), To = EMA50 (the deep,
// trend-defining edge) — raw floats, unordered by size on purpose: in an
// uptrend From > To, in a downtrend From < To.
//
// Why this band is the pullback target: in a confirmed trend price extends
// away from its short averages between impulses and mean-reverts toward them;
// EMA20 is the first dynamic edge a routine pullback touches, and EMA50 is the
// same average the state machine reads — a close beyond it starts breaking
// the structure. The honest "pullback with the trend intact" region is
// therefore bounded by the two. Outside confirmed states there is no trend to
// pull back within, so the field is absent.
type PullbackZone struct {
	From float64 `json:"from"` // EMA20
	To   float64 `json:"to"`   // EMA50
}

// TrendLevels — served ONLY for a confirmed trend (up/down); an unconfirmed
// card carries no levels object at all (product decision 2026-09-15: an
// unconfirmed reading has nothing to invalidate, and a number in the JSON
// reads as a level whatever the docs say).
//
//   - PullbackZone: the EMA20-EMA50 band, always present when confirmed.
//   - Invalidation / InvalidationSide: the level that invalidates the trend
//     idea (see invalidationFor) and its direction — downtrends invalidate
//     ABOVE the EMA cluster, uptrends below it. Present together or not at
//     all: absent only on a degenerate series with no ATR. A pointer, so the
//     JSON can never carry a zero level or a side without a level.
type TrendLevels struct {
	Invalidation     *float64      `json:"invalidation,omitempty"`
	InvalidationSide string        `json:"invalidation_side,omitempty"`
	PullbackZone     *PullbackZone `json:"pullback_zone,omitempty"`
}

// ContentBlocks are ready-made sentences for content writers, one per job,
// so nobody has to reassemble them from the facts list. Additive: served as
// the envelope's "blocks" on trend cards and on S/R cards that show at least
// one level (sr_text.go words the fields for the nearest shown level);
// old consumers ignore it.
//   - what_happened: the verdict and where price is
//   - why_level: what the level the card leans on is made of
//   - scenarios: exactly two "if → then" transitions of the state machine
//   - invalidates: what invalidates a CONFIRMED reading; null otherwise
//   - regime: the local regime in one line (state · timeframe · ADX)
//   - context: macro only (additive, 2026-09-15) — the asset backdrops and
//     Fear & Greed beside the regime; absent on every other agent
//   - state_changes_when: volatility only (additive, 2026-09-15) — what
//     changes the state on the next closed candle; limitations: volatility
//     and funding (funding_text.go) — what the agent does not measure;
//     absent on every other agent
//
// Volatility (vol_text.go) has no price level and no directional idea: its
// why_level explains the ratio thresholds and invalidates is always null.
//
// Macro has no price level, so its why_level is always "" (macro_text.go).
// Momentum (momentum_text.go, 2026-09-15) has none either: its why_level
// explains the indicator thresholds, invalidates is what ends a CONFIRMED
// reading, regime is the asset's LOCAL momentum state. Top-level on the
// single-asset card, per asset in results[].blocks on multi-asset cards.
//
// Gold (gold_text.go, 2026-09-15): what_happened is a SNAPSHOT and says so
// ("Snapshot, not an event: …") — the agent keeps no previous state; scenarios
// are the two day-range closes ([] when the range is undefined); invalidates
// only for a confirmed regime; regime is the local 1d regime; limitations
// names the instrument.
type ContentBlocks struct {
	WhatHappened string   `json:"what_happened"`
	WhyLevel     string   `json:"why_level"`
	Scenarios    []string `json:"scenarios"`
	Invalidates  *string  `json:"invalidates"`
	Regime       string   `json:"regime"`
	Context      string   `json:"context,omitempty"`
	StateChanges string   `json:"state_changes_when,omitempty"`
	Limitations  string   `json:"limitations,omitempty"`
	// Source: whale only (additive, 2026-09-15) — where the observation
	// comes from; absent on every other agent.
	Source string `json:"source,omitempty"`
}

// SRPoint is one clustered level at raw precision (SRLevel.Raw — the cluster
// mean, not the display-rounded integer) with its touch count and the B4
// metrics. Formulas live on SRLevel / the indicator functions; the wire adds:
//   - strength: touches + 0.5 per above-median-volume touch (== touches on
//     volume-less FX series)
//   - weakening: ≥7 touches with the last 3 touches' mean volume below the
//     first 3's
//   - breaks / holds: frequency counts of level tests over the window
//     (test = close within 0.25×ATR; break = close beyond by >0.25×ATR
//     within 3 bars) — frequencies, never probabilities
//   - last_touch: RFC3339 UTC of the newest touch's bar time
//
// Added 2026-09-15 (additive):
//   - label: the level exactly as the card prints it (instrument precision)
//   - class: "established" (≥7 pivots) | "candidate" | "single_swing" (1)
//   - strength_rank: 1-based position in this array (strength order)
//   - display_rank: 1-based position of the level's line on the card, where
//     each side is listed nearest-to-price first
type SRPoint struct {
	Level        float64 `json:"level"`
	Label        string  `json:"label"`
	Class        string  `json:"class"`
	DisplayRank  int     `json:"display_rank"`
	StrengthRank int     `json:"strength_rank"`
	Touches      int     `json:"touches"`
	Strength     float64 `json:"strength"`
	Weakening    bool    `json:"weakening"`
	Breaks       int     `json:"breaks"`
	Holds        int     `json:"holds"`
	LastTouch    string  `json:"last_touch"`
}

// SRLevels — strength-sorted supports/resistances. Both slices are always
// non-nil so an empty side serializes as [] ("we looked, nothing clustered"),
// never null.
type SRLevels struct {
	Supports    []SRPoint `json:"supports"`
	Resistances []SRPoint `json:"resistances"`
}

// VolThresholds are the volatility rule's ratio thresholds: compressed at a
// ratio ≤ Compressed, expanding at ≥ Expanding. Keys match the state values.
type VolThresholds struct {
	Compressed float64 `json:"compressed"`
	Expanding  float64 `json:"expanding"`
}

// VolLevels — the volatility read at raw precision. ExpansionRatio is the
// original field (ATR(14) of the last closed candle over the mean of the 30
// ATR(14) values before it, unrounded). Added 2026-09-15 (additive): state
// (expanding | normal | compressed — the card words expanding "elevated"),
// timeframe, atr, atr_pct (ATR / last close × 100), baseline, ratio (the same
// value as expansion_ratio) and thresholds.
type VolLevels struct {
	ExpansionRatio float64       `json:"expansion_ratio"`
	State          string        `json:"state"`
	Timeframe      string        `json:"timeframe"`
	ATR            float64       `json:"atr"`
	ATRPct         float64       `json:"atr_pct"`
	Baseline       float64       `json:"baseline"`
	Ratio          float64       `json:"ratio"`
	Thresholds     VolThresholds `json:"thresholds"`
}

// AssetResult is one asset's machine-readable outcome inside a momentum card
// (review fix 3): a mixed scan is no longer distinguishable from a full one
// only by reading fact strings. OK mirrors the per-asset read; Reason is nil
// when OK, else "insufficient_history" | "source_offline". Served as the
// envelope's "results" (momentum only).
//
// Added 2026-09-15 (additive, present only when ok — see momentumResult):
// the read at raw precision (rsi, macd_histogram — the card prints only the
// histogram's sign), verdict (bullish|bearish|neutral), state (which neutral),
// why (the card's reason), timeframe, data_as_of (close of the asset's last
// closed bar), freshness (on_time|market_closed|data_delayed) and the asset's
// content blocks.
type AssetResult struct {
	Asset         string         `json:"asset"`
	OK            bool           `json:"ok"`
	Reason        *string        `json:"reason,omitempty"`
	Timeframe     string         `json:"timeframe,omitempty"`
	DataAsOf      string         `json:"data_as_of,omitempty"`
	Freshness     string         `json:"freshness,omitempty"`
	Verdict       string         `json:"verdict,omitempty"`
	State         string         `json:"state,omitempty"`
	Why           string         `json:"why,omitempty"`
	RSI           *float64       `json:"rsi,omitempty"`
	MACDHistogram *float64       `json:"macd_histogram,omitempty"`
	Blocks        *ContentBlocks `json:"blocks,omitempty"`
	// FX rows only (additive, 2026-09-15 — see fxResult): the last close,
	// its change and the window it spans ("24h" | "since_previous_close")
	// with the reference bar's close time, EMA50 vs EMA200 (above | below |
	// equal) and the close's place in the range, 0–100. freshness on an FX
	// row is on_time | market_closed | data_delayed (pairs) or on_time |
	// no_recent_bar (gold).
	Price            *float64 `json:"price,omitempty"`
	ChangePct        *float64 `json:"change_pct,omitempty"`
	ChangeWindow     string   `json:"change_window,omitempty"`
	ChangeFrom       string   `json:"change_from,omitempty"`
	EMARelation      string   `json:"ema_relation,omitempty"`
	RangePositionPct *float64 `json:"range_position_pct,omitempty"`
	// Funding rows only (additive, 2026-09-15 — see fundingResults): the
	// symbol, its last funding rate, the threshold of the rate's own side
	// (+0.0003 at or above zero, -0.0001 below), whether the rate is past
	// it, |rate| ÷ |that threshold|, and whether this is the coin the card
	// shows. Raw precision. A symbol that did not answer carries only
	// asset/symbol/ok=false/reason.
	Symbol           string   `json:"symbol,omitempty"`
	Rate             *float64 `json:"rate,omitempty"`
	SideThreshold    *float64 `json:"side_threshold,omitempty"`
	Crossed          *bool    `json:"crossed,omitempty"`
	RatioToThreshold *float64 `json:"ratio_to_threshold,omitempty"`
	Selected         *bool    `json:"selected,omitempty"`
}

// FundingReadout is the envelope's "funding" object (funding cards only,
// additive 2026-09-15; docs/demobot-http.md "Funding card").
type FundingReadout struct {
	State           string  `json:"state"`            // positive_above_threshold | negative_below_threshold | within_thresholds | partial | rates_offline
	SelectedSymbol  *string `json:"selected_symbol"`  // null on partial / rates_offline
	SelectionReason *string `json:"selection_reason"` // furthest_past_own_threshold | closest_to_own_threshold | null
	RateKind        string  `json:"rate_kind"`        // always "last_funding_rate"
	// FundingInterval is always null: premiumIndex serves no funding interval
	// and the agent does not verify one, so no "/8h" is claimed.
	FundingInterval *string              `json:"funding_interval"`
	Thresholds      FundingThresholds    `json:"thresholds"`
	Crossed         int                  `json:"crossed"` // received majors past a threshold
	Coverage        FundingCoverage      `json:"coverage"`
	Liquidations    *FundingLiquidations `json:"liquidations"` // null when the feed is offline
}

type FundingThresholds struct {
	Long  float64 `json:"long"`
	Short float64 `json:"short"`
}

type FundingCoverage struct {
	Received      int      `json:"received"`
	Total         int      `json:"total"`
	Missing       []string `json:"missing"` // [] when none
	MinForVerdict int      `json:"min_for_verdict"`
}

// FundingLiquidations is the 1h window as the card counted it from the
// backend feed (newest feed_limit events; capped = the page was full, so the
// window may hold more).
type FundingLiquidations struct {
	WindowMinutes int                `json:"window_minutes"`
	Count         int                `json:"count"`
	USD           float64            `json:"usd"`
	LongUSD       float64            `json:"long_usd"`
	ShortUSD      float64            `json:"short_usd"`
	FeedLimit     int                `json:"feed_limit"`
	Capped        bool               `json:"capped"`
	Cluster       *FundingClusterOut `json:"cluster"` // null when the backend served no zone
}

// FundingClusterOut is the observed liquidation cluster the card shows: a
// past 0.5% price band of the backend's 1h window, not a forecast level.
type FundingClusterOut struct {
	Symbol      string  `json:"symbol"`
	Side        string  `json:"side"` // long_liq | short_liq: the larger USD side of the band
	PriceBand   string  `json:"price_band"`
	USD         float64 `json:"usd"`
	Count       int     `json:"count"`
	LastEventAt *string `json:"last_event_at"` // newest event of the band in the served feed, null when none is there
	// BandVsMark is where the band sits against the symbol's premiumIndex
	// mark price: "above" | "below" | "inside" (the mark within the band);
	// null without a mark price. A position, not a price or a distance: it
	// changes only when the mark crosses a band edge, so the body does not
	// move with every tick.
	BandVsMark *string `json:"band_vs_mark"`
}

// GoldReadout is the envelope's "gold" object (gold cards only, additive
// 2026-09-15; docs/demobot-http.md "Gold card"). Each part of the composite
// carries its own stamp: data_as_of stays the daily close.
type GoldReadout struct {
	Regime         string          `json:"regime"`          // trend state on 1d: flat | grey | up | down | conflict
	Confirmed      bool            `json:"confirmed"`       // a direction is stated: regime up/down AND a 1h price
	DailyAsOf      string          `json:"daily_as_of"`     // close of the last closed 1d bar (= data_as_of)
	PriceAsOf      *string         `json:"price_as_of"`     // close of the last closed 1h bar
	PriceFreshness *string         `json:"price_freshness"` // on_time | stale (older than 6h at answer time)
	Price          *float64        `json:"price"`           // that 1h close, raw
	PricePosition  *string         `json:"price_position"`  // above | inside | below the day range; null without a range
	DayRange       *GoldDayRange   `json:"day_range"`       // null when undefined (deeper than the inside-day cap)
	MacroAsOf      *string         `json:"macro_as_of"`     // oldest as_of of the voting macro lamps
	MacroBackdrop  *string         `json:"macro_backdrop"`  // support | pressure | neutral; null without a read
	MacroLamps     *GoldMacroLamps `json:"macro_lamps"`     // voting lamps by contribution for gold
}

// GoldDayRange is the range the scenarios classify against.
type GoldDayRange struct {
	High            float64 `json:"high"`
	Low             float64 `json:"low"`
	CandleDate      string  `json:"candle_date"`       // YYYY-MM-DD of the 1d candle it came from
	InsideDaysAfter int     `json:"inside_days_after"` // inside days walked back (0 = the last closed day)
}

// GoldMacroLamps counts the voting lamps of the gold macro model.
type GoldMacroLamps struct {
	For     int `json:"for"`
	Neutral int `json:"neutral"`
	Against int `json:"against"`
}

// Card is one agent's reply. RenderHTML produces the exact Telegram
// HTML-parse-mode body (golden-tested in card_test.go).
type Card struct {
	Emoji     string // one of emojiBull / emojiBear / emojiNeutral
	Agent     string // full agent name, e.g. "Momentum Agent"
	ShortName string // digest one-liner name, e.g. "Momentum"
	Asset     string // "" when the agent is not asset-specific — HUMAN label
	// AssetKey is the MACHINE value the HTTP envelope serves as "asset".
	// Empty means "same as Asset", which is the case for every agent whose
	// human label already is a plain ticker.
	//
	// It exists because the two jobs pulled apart on gold: the card must say
	// "GOLD · COMEX GC=F" (the reader has to know these are futures, not
	// spot), while integrations branch on the documented "XAUUSD" and must
	// not break over our honesty. One field per job.
	AssetKey   string
	Verdict    string // bold verdict line
	Short      string // short verdict for digest one-liners
	Facts      []string
	Confidence *int // nil → no confidence bar (API gave none)
	DataTime   time.Time
	Command    string // bot command key, used for Refresh/How-it-works callbacks
	HowItWorks string // ≤200 chars, shown via answerCallbackQuery alert
	Deviation  int    // 0..100 deviation-from-neutral used by the priority rule
	Offline    bool   // true when the data source was unreachable
	SourceNote string // extra footer note, e.g. "data: Yahoo Finance"
	// Status is the machine state behind the HTTP envelope's ok/reason pair.
	// Zero value = real reading; degraded builders set the matching enum.
	Status cardStatus
	// State is the builder's own state-machine output when it has one:
	// trend flat|grey|up|down|conflict, macro risk_on|mixed|risk_off|unknown,
	// vol expanding|normal|compressed. "" for agents without a state machine.
	// It travels into the AI payload as authoritative context (ai.go): a
	// state that WITHHELD confirmation must never be argued back into one by
	// the model — that contradiction is the defect stateConfirms() guards.
	State string
	// Levels carries one of TrendLevels / SRLevels / VolLevels (or nil) —
	// served verbatim as the envelope's "levels" object, ignored by Telegram.
	Levels any
	// Results is the per-asset machine outcome of a momentum card (every
	// multi-asset card and scan; one entry on the single-asset card; nil
	// elsewhere) — served as the envelope's "results", ignored by Telegram
	// (the facts carry the human form).
	Results []AssetResult
	// Blocks is the content-ready form of the card (trend, S/R, macro, the
	// single-asset momentum card) — served as the envelope's "blocks",
	// ignored by Telegram. nil elsewhere.
	Blocks *ContentBlocks
	// Macro is the machine-readable readout of a macro card (per-lamp rule,
	// weight, contribution, source, as_of; freshness; score bands) — served
	// as the envelope's "macro", ignored by Telegram. nil elsewhere.
	Macro *MacroReadout
	// Funding is the funding card's machine readout (state, coin, coverage,
	// liquidations) — served as the envelope's "funding". nil elsewhere.
	Funding *FundingReadout
	// Gold is the gold card's machine readout (per-part stamps, price, day
	// range, macro basis) — served as the envelope's "gold". nil elsewhere.
	Gold *GoldReadout
	// Whale is the whale card's machine readout (state, count, threshold,
	// window, the listed transactions) — served as the envelope's "whale"
	// (whale_text.go). nil elsewhere and on the offline card.
	Whale *WhaleReadout
	// noValidator marks a card whose body is NOT a function of one stamped
	// snapshot — composites of several sources or series, or text built from
	// the request clock (gold, fx, funding, the composite momentum card, a
	// multi-asset scan). No component time can serve as its Last-Modified: a
	// component may change while the chosen stamp stays put, and a
	// conditional GET would then answer 304 for a changed body. writeCard
	// serves such a card with no Last-Modified and ignores If-Modified-Since.
	// DataTime (data_as_of, the footer) is unaffected. Not rendered.
	noValidator bool
	// trendConclusion is the landing-page conclusion for an UNCONFIRMED trend
	// card ("" when confirmed and for every other agent). Not served; the
	// showcase uses it instead of calling a card with an EMA lean "neutral".
	trendConclusion string
	// confirmed marks a reading on which the agent's OWN rule committed to a
	// finding: funding crowded (either side), momentum bullish/bearish on a
	// ranked crypto asset, trend up/down. Digest ranking only (priority.go):
	// an unconfirmed reading never outranks a confirmed one. Not rendered,
	// not served.
	confirmed bool
	// noRankedRead marks a multi-asset momentum card with no BTC/ETH reading
	// (dead or short crypto, a live gold/FX read): nothing on it is a read the
	// digest ranks, so rankCandidate excludes it (no_ranked_read) instead of
	// judging a gold bar's freshness. Digest ranking only; not rendered.
	noRankedRead bool
	// rankAsOf is the data time of the reading Deviation comes from, when it
	// differs from DataTime (momentum: its Binance reads, while DataTime is
	// the oldest bar on the card, gold included). Zero → DataTime. Freshness
	// check of the digest ranking only; not rendered.
	rankAsOf time.Time
	// AIHTML is a pre-rendered AI block ("<b>AI idea:</b> <i>…</i>") appended
	// after the facts and confidence bar. Builders MUST esc() every dynamic
	// value when composing it — RenderHTML writes it verbatim.
	AIHTML string
}

// assetKey is the machine asset value: AssetKey when the card set one, else
// the human label (identical for every agent but gold).
func (c Card) assetKey() string {
	if c.AssetKey != "" {
		return c.AssetKey
	}
	return c.Asset
}

func esc(s string) string { return html.EscapeString(s) }

// RenderHTML renders the full card:
//
//	{emoji} <b>{Agent}</b> · {Asset}
//	<b>{Verdict}</b>
//	• fact …
//	Confidence: ■■■□□ 60%          (only when Confidence != nil)
//	<b>AI …:</b> <i>…</i>          (only when AIHTML != "")
//
//	<i>Analytics, not financial advice · AlphaVizor · {UTC time}</i>
func (c Card) RenderHTML() string {
	footer := "\n<i>Analytics, not financial advice · AlphaVizor · " +
		c.DataTime.UTC().Format("2006-01-02 15:04") + " UTC"
	if c.SourceNote != "" {
		footer += " · " + esc(c.SourceNote)
	}
	return c.renderBody() + footer + "</i>"
}

// renderBody is the card without the footer — /digest embeds the winner's
// body and closes the whole message with a single footer.
func (c Card) renderBody() string {
	var b strings.Builder
	b.WriteString(c.Emoji)
	b.WriteString(" <b>")
	b.WriteString(esc(c.Agent))
	b.WriteString("</b>")
	if c.Asset != "" {
		b.WriteString(" · ")
		b.WriteString(esc(c.Asset))
	}
	b.WriteString("\n<b>")
	b.WriteString(esc(c.Verdict))
	b.WriteString("</b>\n")
	for _, f := range c.Facts {
		b.WriteString("• ")
		b.WriteString(esc(f))
		b.WriteString("\n")
	}
	if c.Confidence != nil {
		v := clampInt(*c.Confidence, 0, 100)
		b.WriteString("Confidence: ")
		b.WriteString(confidenceBar(v))
		b.WriteString(" ")
		b.WriteString(strconv.Itoa(v))
		b.WriteString("%\n")
	}
	if c.AIHTML != "" {
		b.WriteString(c.AIHTML)
		b.WriteString("\n")
	}
	return b.String()
}

// OneLiner renders the compact digest form: "🟢 <b>Momentum</b> BTC: bullish".
func (c Card) OneLiner() string {
	var b strings.Builder
	b.WriteString(c.Emoji)
	b.WriteString(" <b>")
	b.WriteString(esc(c.ShortName))
	b.WriteString("</b>")
	if c.Asset != "" {
		b.WriteString(" ")
		b.WriteString(esc(c.Asset))
	}
	b.WriteString(": ")
	b.WriteString(esc(c.Short))
	return b.String()
}

// confidenceBar maps 0-100 to five blocks; filled = round(v/20),
// rounding half away from zero (Go math.Round), clamped to 0..5.
func confidenceBar(v int) string {
	filled := int(math.Round(float64(clampInt(v, 0, 100)) / 20.0))
	return strings.Repeat("■", filled) + strings.Repeat("□", 5-filled)
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// effectiveStatus is the machine status the HTTP layer serves: the builder's
// explicit Status, with one defensive collapse — a card flagged Offline that
// carries no specific status reads as source_offline, so a future degraded
// path that only sets the flag can never serve ok=true.
func (c Card) effectiveStatus() cardStatus {
	if c.Status == statusOK && c.Offline {
		return statusSourceOffline
	}
	return c.Status
}

// offlineCard is the honest degraded state — never fake data.
func offlineCard(agent, shortName, asset, command, how string) Card {
	return Card{
		Emoji:     emojiNeutral,
		Agent:     agent,
		ShortName: shortName,
		Asset:     asset,
		// AssetKey is set by the spec-aware wrappers (assetOffline /
		// insufficientCard); a bare string caller has no key to give.
		Verdict:    "Data source offline right now. The team is on it.",
		Short:      "offline",
		DataTime:   time.Now().UTC(),
		Command:    command,
		HowItWorks: how,
		Deviation:  0,
		Offline:    true,
		Status:     statusSourceOffline,
	}
}
