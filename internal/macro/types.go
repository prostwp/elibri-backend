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

// Regime — the "big money" risk regime. 3 buckets (discovery §7 collapsed the
// original 5 to these). Thresholds on composite: <35 risk_off, 35..65 mixed,
// >65 risk_on.
const (
	RegimeRiskOn  = "risk_on"
	RegimeMixed   = "mixed"
	RegimeRiskOff = "risk_off"
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
	DeltaPct *float64 `json:"delta_pct"` // session change % (Close−Open); nil on N/D (delta==0 is a real "no move")
	Status   string   `json:"status"`    // tailwind|neutral|headwind; "" on N/D OR unknown direction
	AsOf     string   `json:"as_of"`     // ISO RFC3339, or "" on N/D
}

// Correlation — BTC↔X over the rolling window (a compute output).
type Correlation struct {
	Pair   string   `json:"pair"`   // "btc_spx"|"btc_gold"|"btc_dxy"
	Coef   *float64 `json:"coef"`   // nil when too few points / NaN → UI "Building correlation window"
	Label  string   `json:"label"`  // "moving like stocks" etc.; "" when coef==nil
	Window string   `json:"window"` // "≈3h (52 points)" — human description of the window
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
	Regime            string        `json:"regime"`              // risk_on|mixed|risk_off
	Composite         *int          `json:"composite"`           // 0..100; nil if ≥3 lamps N/D → UI "Not enough live markets"
	TradfinMarketOpen bool          `json:"tradfin_market_open"` // false on weekends / off-hours
	TradfinAsOf       string        `json:"tradfin_as_of"`       // ISO of the freshest tradfin ts, or ""
	CapturedAt        string        `json:"captured_at"`         // ISO of when the response was assembled (now)
	Lamps             []Lamp        `json:"lamps"`               // exactly 5 (N/D → Value/Status empty)
	Correlations      []Correlation `json:"correlations"`        // exactly 3
	Fng               *FnG          `json:"fng"`                 // nil if alternative.me is down
	GeneratedIdea     string        `json:"generated_idea"`      // rule-based, safe; "" if it failed the filter
	Calendar          []CalEvent    `json:"calendar"`            // top-5 high+medium in 72h; [] if none
}
