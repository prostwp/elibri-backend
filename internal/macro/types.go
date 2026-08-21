package macro

// types.go — wire + internal shapes for the Macro Sentiment scenario.
//
// Mirrors funding/types.go (JSON-tagged structs, nullable pointers for "no
// data" fields). The Response is the locked /api/v1/macro contract (spec §5):
// slices are always non-nil ([], not null); *int / *float64 fields serialise as
// JSON null when the underlying market is unavailable (N/D), NEVER a 0-sentinel.

import "time"

// Status — a lamp's direction relative to crypto.
//
// ⚠️ "support" is a BANNED word (DESIGN_SYSTEM §8, dnevnik 2026-05-30 15:05):
// it trips the safe-AI banned-grep and the isIdeaSafe() filter on the frontend.
// We use tailwind/neutral/headwind — consistent with the Hero copy
// "tailwind/headwind for crypto". NEVER "support" / "pressure" as an enum value.
const (
	StatusTailwind = "tailwind" // lamp favours crypto (🟢)
	StatusNeutral  = "neutral"  // lamp is neutral (🟡)
	StatusHeadwind = "headwind" // lamp weighs on crypto (🔴)
)

// Regime — the "big money" risk regime. 3 scored buckets (discovery §7
// collapsed the original 5 to these) plus the no-data state. Thresholds on
// composite: <35 risk_off, 35..65 mixed, >65 risk_on.
//
// "unknown" is NOT a scored bucket: it means zero lamps carry a value (all
// null — e.g. tradfin closed and every quote N/D), so there is no input to
// classify at all. "mixed" is reserved for REAL lamp values that don't add up
// to a single regime; it must never be emitted off an empty lamp set (the
// honesty defect this constant fixes: a confident "signals are split" with
// zero inputs).
const (
	RegimeRiskOn  = "risk_on"
	RegimeMixed   = "mixed"
	RegimeRiskOff = "risk_off"
	RegimeUnknown = "unknown"
)

// Lamp keys — the 5 traffic-light series, in render order.
const (
	KeyDXY   = "dxy"
	KeyRates = "rates"
	KeyVIX   = "vix"
	KeySPX   = "spx"
	KeyGold  = "gold"
)

// Correlation pairs — BTC vs the three big-money series we cross-check.
const (
	PairBTCSPX  = "btc_spx"
	PairBTCGold = "btc_gold"
	PairBTCDXY  = "btc_dxy"
)

// stooq symbol IDs (verbatim from dnevnik — do NOT "fix" them). The intuitive
// caret forms (^vix/^dxy/^tnx) return N/D; these are the ones that resolve.
const (
	SymSPX   = "^spx"     // S&P 500
	SymVIX   = "vi.f"     // VIX (front future ≈ spot)
	SymDXY   = "dx.f"     // Dollar index (ICE future ≈ spot)
	SymGold  = "xauusd"   // Gold spot
	SymRates = "10yusy.b" // US 10Y yield (%)
	SymBTC   = "btcusd"   // BTC (24/7, for correlations)
)

// Quote — the latest snapshot of one stooq symbol. On N/D we still store the
// quote with OK=false (we keep the last fact, not a stale "last valid"); the
// handler renders "—" for a !OK quote rather than a fabricated number.
//
// Open is the session open from the SAME CSV row; the lamp delta is the SESSION
// change (Close − Open), NOT a rolling 24h delta — stooq returns both on one
// row, so the delta is meaningful from the very first cycle (and on weekends,
// off the Friday session). Open is internal (json:"-") — it never ships.
type Quote struct {
	Symbol string    `json:"symbol"` // stooq id, e.g. "^spx"
	Price  float64   `json:"price"`  // Close
	Open   float64   `json:"-"`      // session Open (baseline for the session-change delta); 0 on N/D
	AsOf   time.Time `json:"as_of"`  // Date+Time parsed from the CSV (UTC)
	OK     bool      `json:"ok"`     // false on N/D → lamp shows "—"
}

// Lamp — one of the 5 traffic-light lamps (a compute output, not stored).
type Lamp struct {
	Key      string   `json:"key"`       // "dxy"|"rates"|"vix"|"spx"|"gold"
	Label    string   `json:"label"`     // "Dollar (DXY)" — human-readable, EN
	Value    *float64 `json:"value"`     // nil on N/D → UI "—"
	OK       bool     `json:"ok"`        // value present (Value != nil) — explicit data-presence flag (additive field)
	DeltaPct *float64 `json:"delta_pct"` // session change % (Close−Open); nil on N/D (delta==0 is a real "no move")
	Status   string   `json:"status"`    // tailwind|neutral|headwind; "" on N/D OR unknown direction
	AsOf     string   `json:"as_of"`     // ISO RFC3339 of the last KNOWN source date — filled even when Value is nil (stale N/D quote), "" only when the source never carried a date
}

// Correlation — BTC↔X over the daily-close window (a compute output).
//
// B2 rework: the coefficient is Pearson over the last 20-30 DAILY closes
// (stooq daily history, date-aligned), replacing the old ~3h intraday ring.
// Wire shape is backward compatible: pair/coef/label/window keep their names
// and types; OK and Points are ADDITIVE fields (older consumers ignore them).
type Correlation struct {
	Pair   string   `json:"pair"`   // "btc_spx"|"btc_gold"|"btc_dxy"
	Coef   *float64 `json:"coef"`   // nil when too few points / degenerate → UI "Building correlation window"
	Label  string   `json:"label"`  // "moving like stocks" etc.; "" when coef==nil
	Window string   `json:"window"` // "24 daily closes (20-30d window)" — human description
	// OK is true when Coef was computed (≥ MinDailyCorrPoints overlapping daily
	// closes, non-degenerate). ok:false + coef:null = window still building.
	OK bool `json:"ok"`
	// Points is the overlapping daily-close count behind the read (0..30).
	Points int `json:"points"`
}

// FnG — crypto Fear & Greed cross-check (alternative.me).
type FnG struct {
	Value int    `json:"value"` // 0..100
	Label string `json:"label"` // "Greed"/"Fear"/…
	OK    bool   `json:"ok"`    // false if alternative.me unreachable → UI hides the block
}

// CalEvent — a slim calendar event (reuse of macrocal.Event, narrowed for the
// frontend).
type CalEvent struct {
	Country string `json:"country"`
	Event   string `json:"event"`
	Impact  string `json:"impact"` // "high"|"medium"
	Time    string `json:"time"`   // ISO RFC3339 (UTC)
}

// Response — the body of GET /api/v1/macro. Slices are always non-nil ([], not
// null).
type Response struct {
	Regime            string        `json:"regime"`              // risk_on|mixed|risk_off|unknown (unknown = zero lamps carry a value)
	Composite         *int          `json:"composite"`           // 0..100; nil if ≥3 lamps N/D → UI "Not enough live markets"
	TradfinMarketOpen bool          `json:"tradfin_market_open"` // clock-based futures-week window (see TradfinWindowOpen) — NOT a data-presence signal
	TradfinOk         bool          `json:"tradfin_ok"`          // at least one lamp carries a value (additive field) — the data-presence signal that drives regime honesty
	TradfinAsOf       string        `json:"tradfin_as_of"`       // ISO of the freshest tradfin ts, or ""
	CapturedAt        string        `json:"captured_at"`         // ISO of when the response was assembled (now)
	Lamps             []Lamp        `json:"lamps"`               // exactly 5 (N/D → Value/Status empty)
	Correlations      []Correlation `json:"correlations"`        // exactly 3
	Fng               *FnG          `json:"fng"`                 // nil if alternative.me is down
	GeneratedIdea     string        `json:"generated_idea"`      // rule-based, safe; "" if it failed the filter
	Calendar          []CalEvent    `json:"calendar"`            // top-5 high+medium in 72h; [] if none
}
