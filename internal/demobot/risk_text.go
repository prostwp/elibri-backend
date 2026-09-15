package demobot

// risk_text.go — everything the Risk Calculator card SAYS (stage 1,
// 2026-09-15).
//
// The formula is unchanged: (balance × risk%) ÷ |entry − stop| (calcRisk),
// the same numbers on the same inputs. What changed is only what the card
// claims:
//
//   - the first line is the status: calculated / not calculated (not enough
//     parameters, a parameter is not a number, numbers that do not work);
//   - the calculator knows four numbers and no instrument, so the result is
//     "abstract units" under one stated condition (a 1.0 price move changes
//     one unit's value by 1.0 in the account currency), never a "position
//     size" of a real instrument;
//   - "Max loss at stop: $100" → "Planned price risk: 100 (account
//     currency), before fees and slippage": no "$" (the account currency is
//     not a parameter) and no promise about the fill;
//   - no position notional: its meaning depends on the instrument;
//   - the limits sit on the card itself (FX lots, futures multipliers, CFDs,
//     account-currency conversion), not only behind How it works;
//   - the footer and the machine field say "calculation time": the card reads
//     no market data;
//   - no content blocks: the calculator observes no market (docs "Risk card").
//
// The print rule (riskPrint): float noise is dropped first — a value is
// rounded to 12 significant digits (riskClean: 0.0050000000000001155 → 0.005).
// Then the shown size is rounded DOWN to 4 significant digits and money is
// rounded UP (to cents from 1 to 1e15, to 4 significant digits outside).
// Hence a printed price risk is never below the computed one, except for a
// value with more than 12 significant digits, and then by at most half a unit
// in its 12th significant digit (≤ 5e-12 relative; from about 1e9 on that is
// a fraction of a cent or more). The Check value never exceeds the printed
// planned price risk: the shown size steps down when it would. A number that
// cannot be printed as a finite float64 is the range error, never NaN / Inf.

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
)

const (
	riskFactMaxRunes = 110

	riskStatusCalculated = "calculated"

	riskFormula      = "(balance × risk%) ÷ |entry − stop|"
	riskCondition    = "a 1.0 price move changes one unit's value by 1.0 in the account currency"
	riskFitsExample  = "BTC/USD spot on a USD account"
	riskExampleLabel = "BTC/USD spot, USD account"

	riskLineValid       = "Valid only if " + riskCondition
	riskLineFits        = "Fits e.g. " + riskFitsExample + "; the calculator does not know your instrument"
	riskLineUnsupported = "Not supported: FX lots, futures with a multiplier, CFDs, account-currency conversion"
	riskLineDisclaimer  = "Position sizing math only — not a trade suggestion."
	riskLineExample     = "Example inputs read as " + riskFitsExample + ": here 1 abstract unit = 1 BTC"
	riskLineExampleCmd  = "Example: /risk 10000 1 64000 62500 (" + riskExampleLabel + ")"
	riskSourceNote      = "calculation time, no market data read"

	riskErrNotFinite = "every number must be finite"
	riskErrRange     = "numbers too large or too small to calculate"
)

// riskArityError is a /risk command with the wrong count of numbers.
type riskArityError struct{ Got int }

func (e riskArityError) Error() string { return fmt.Sprintf("expected 4 numbers, got %d", e.Got) }

// RiskReadout is the envelope's "risk" object (additive, 2026-09-15;
// docs/demobot-http.md "Risk card"). Served only with a result: every
// not-calculated input is a 400 over HTTP.
type RiskReadout struct {
	Status                string            `json:"status"` // always "calculated"
	Formula               string            `json:"formula"`
	Inputs                RiskInputs        `json:"inputs"`
	PriceRiskAccountCcy   float64           `json:"price_risk_account_ccy"`    // balance × risk% ÷ 100, raw
	PriceDistance         float64           `json:"price_distance"`            // |entry − stop|, raw
	ResultAbstractUnits   float64           `json:"result_abstract_units"`     // the formula result, raw
	ShownAbstractUnits    float64           `json:"shown_abstract_units"`      // as printed: down to 4 significant digits after noise cleaning (can exceed the raw result by ≤ 5e-12 relative)
	PriceRiskAtShownUnits float64           `json:"price_risk_at_shown_units"` // shown size × distance, raw
	QuantityUnit          string            `json:"quantity_unit"`             // always "abstract_units"
	AccountCurrency       *string           `json:"account_currency"`          // always null: not a parameter
	Applicability         RiskApplicability `json:"applicability"`
	Excludes              []string          `json:"excludes"`
	DisplayRounding       RiskRounding      `json:"display_rounding"`
	ReadsMarketData       bool              `json:"reads_market_data"` // always false
	CalculatedAt          string            `json:"calculated_at"`     // = data_as_of
	ContentLine           string            `json:"content_line"`      // the "position sizing" service line
}

type RiskInputs struct {
	Balance float64 `json:"balance"`
	RiskPct float64 `json:"risk_pct"`
	Entry   float64 `json:"entry"`
	Stop    float64 `json:"stop"`
}

// RiskApplicability: the instrument model is never verified in stage 1 —
// the calculator has no instrument or account-currency parameter.
type RiskApplicability struct {
	Status       string   `json:"status"` // always "not_verified"
	Condition    string   `json:"condition"`
	FitsExample  string   `json:"fits_example"`
	NotSupported []string `json:"not_supported"`
}

type RiskRounding struct {
	Size  string `json:"size"`  // "down_4_significant"
	Money string `json:"money"` // "up_to_cents"
}

// riskCardFrom words the calculator for already-parsed input.
func riskCardFrom(c Card, args []float64, isExample bool, parseErr error) Card {
	c.SourceNote = riskSourceNote
	if parseErr != nil {
		var ae riskArityError
		switch {
		case errors.As(parseErr, &ae) && ae.Got < 4:
			c.Verdict = fmt.Sprintf("Not calculated: not enough parameters (%d of 4)", ae.Got)
			c.Facts = []string{riskUsage, riskLineExampleCmd}
		case errors.As(parseErr, &ae):
			c.Verdict = fmt.Sprintf("Not calculated: too many parameters (%d, need 4)", ae.Got)
			c.Facts = []string{riskUsage, riskLineExampleCmd}
		default:
			c.Verdict = "Not calculated: a parameter is not a number"
			c.Facts = []string{truncate(parseErr.Error(), riskFactMaxRunes), riskUsage, riskLineExampleCmd}
		}
		return c
	}
	r, err := calcRisk(args[0], args[1], args[2], args[3])
	var p riskPrinted
	if err == nil {
		p, err = riskPrint(r)
	}
	if err != nil {
		c.Verdict = "Not calculated: those numbers don't work"
		c.Facts = []string{err.Error(), riskUsage}
		return c
	}
	size, budget := p.Size, p.Budget

	c.Verdict = "Calculated, instrument model not confirmed: " + size + " abstract units"
	if isExample {
		c.Verdict = "Example (" + riskExampleLabel + "), calculated: " + size + " abstract units"
	}
	c.Facts = []string{
		riskLineValid,
		riskLineFits,
		"Planned price risk: " + budget + " (account currency), before fees and slippage",
		fmt.Sprintf("Basis: %s%% of balance %s; entry %s, stop %s",
			riskNum(args[1]), riskNum(args[0]), riskNum(args[2]), riskNum(args[3])),
		fmt.Sprintf("Check: %s (rounded down) × distance %s = %s ≤ %s",
			size, p.Distance, p.Check, budget),
		riskLineUnsupported,
		riskLineDisclaimer,
	}
	if isExample {
		c.Facts = append(c.Facts, riskLineExample, riskUsage)
	}
	c.Risk = &RiskReadout{
		Status:                riskStatusCalculated,
		Formula:               riskFormula,
		Inputs:                RiskInputs{Balance: args[0], RiskPct: args[1], Entry: args[2], Stop: args[3]},
		PriceRiskAccountCcy:   r.RiskAmount,
		PriceDistance:         r.PerUnit,
		ResultAbstractUnits:   r.Size,
		ShownAbstractUnits:    p.Shown,
		PriceRiskAtShownUnits: p.AtShown,
		QuantityUnit:          "abstract_units",
		Applicability: RiskApplicability{
			Status:       "not_verified",
			Condition:    riskCondition,
			FitsExample:  riskFitsExample,
			NotSupported: []string{"fx_lots", "futures_multiplier", "cfd", "account_currency_conversion"},
		},
		Excludes:        []string{"fees", "slippage", "price_gaps", "quantity_step_rounding"},
		DisplayRounding: RiskRounding{Size: "down_4_significant", Money: "up_to_cents"},
		CalculatedAt:    c.DataTime.UTC().Format(time.RFC3339),
		ContentLine:     "Position sizing only: " + size + " abstract units for a planned price risk of " + budget,
	}
	return c
}

// ── the print rule ───────────────────────────────────────────────────────────

// riskClean drops float noise below 12 significant digits
// (0.0050000000000001155 → 0.005, 99.99000000000001 → 99.99).
func riskClean(v float64) float64 {
	s, err := strconv.ParseFloat(strconv.FormatFloat(v, 'g', 12, 64), 64)
	if err != nil {
		return v
	}
	return s
}

// riskPrinted is what the card prints for one result, and the two raw
// numbers behind the printed size.
type riskPrinted struct {
	Shown, AtShown                float64 // shown size; shown size × distance
	Size, Budget, Check, Distance string
}

// riskPrint applies the print rule to a formula result (the formula itself is
// calcRisk's). The shown size steps down one 4-digit step while the Check
// value would print above the planned price risk (float noise only, at most a
// step or two). Anything not printable as a finite number is the range error.
func riskPrint(r riskResult) (riskPrinted, error) {
	fail := fmt.Errorf("%s", riskErrRange)
	budget, ok := riskMoneyOK(r.RiskAmount)
	if !ok {
		return riskPrinted{}, fail
	}
	limit, _ := strconv.ParseFloat(budget, 64)
	shown, ok := riskSig4(r.Size, false)
	for step := 0; ok && step <= 3; step++ {
		at := shown * r.PerUnit
		check, okc := riskMoneyOK(at)
		if !okc {
			return riskPrinted{}, fail
		}
		if v, _ := strconv.ParseFloat(check, 64); v <= limit {
			return riskPrinted{Shown: shown, AtShown: at, Size: riskNum(shown), Budget: budget,
				Check: check, Distance: riskNum(r.PerUnit)}, nil
		}
		shown, ok = riskStepDown4(shown)
	}
	return riskPrinted{}, fail
}

// riskSig4 rounds v > 0 to 4 significant digits, down (up=false) or up, after
// dropping the noise riskClean drops. The digits come from strconv, never
// from powers of ten: math.Pow(10, 3-e) overflowed below ~1e-305 and printed
// NaN. ok is false when the result is not a positive finite float64.
func riskSig4(v float64, up bool) (float64, bool) {
	c := riskClean(v)
	if !(c > 0) || math.IsInf(c, 0) {
		return math.NaN(), false
	}
	mant, exp, _ := strings.Cut(strconv.FormatFloat(c, 'e', -1, 64), "e")
	digits := strings.Replace(mant, ".", "", 1)
	e, err := strconv.Atoi(exp)
	if err != nil {
		return math.NaN(), false
	}
	rest := ""
	if len(digits) > 4 {
		digits, rest = digits[:4], digits[4:]
	}
	n, _ := strconv.Atoi((digits + "000")[:4])
	if up && strings.Trim(rest, "0") != "" {
		n++ // 9999 → 10000 parses as the next power of ten
	}
	return riskParse4(n, e)
}

// riskStepDown4 is the 4-significant-digit value one step below v (which
// already has at most 4 significant digits).
func riskStepDown4(v float64) (float64, bool) {
	mant, exp, _ := strings.Cut(strconv.FormatFloat(v, 'e', 3, 64), "e")
	n, _ := strconv.Atoi(strings.Replace(mant, ".", "", 1))
	e, err := strconv.Atoi(exp)
	if err != nil {
		return math.NaN(), false
	}
	if n--; n < 1000 {
		n, e = 9999, e-1
	}
	return riskParse4(n, e)
}

// riskParse4 is n × 10^(e−3) for a 4-digit n, when it is a positive finite float64.
func riskParse4(n, e int) (float64, bool) {
	f, err := strconv.ParseFloat(strconv.Itoa(n)+"e"+strconv.Itoa(e-3), 64)
	if err != nil || !(f > 0) || math.IsInf(f, 0) {
		return math.NaN(), false
	}
	return f, true
}

// riskFloor4 is the shown size before the Check step-down: down to 4
// significant digits (NaN when not printable).
func riskFloor4(v float64) float64 {
	f, _ := riskSig4(v, false)
	return f
}

// riskMoney prints an account-currency amount rounded UP: to cents from 1 to
// 1e15, to 4 significant digits outside that range, after the noise drop (see
// the print rule above for the one case it can print below v).
func riskMoney(v float64) string {
	s, _ := riskMoneyOK(v)
	return s
}

func riskMoneyOK(v float64) (string, bool) {
	c := riskClean(v)
	if !(c > 0) || math.IsInf(c, 0) {
		return "", false
	}
	if c >= 1 && c < 1e15 {
		return riskNum(math.Ceil(riskClean(c*100)) / 100), true
	}
	f, ok := riskSig4(c, true)
	if !ok {
		return "", false
	}
	return riskNum(f), true
}

// riskNum prints a number at 12 significant digits at most, plain from 1e-5
// to 1e15 and in exponent form outside (every number ≤ 18 characters, so each
// line fits riskFactMaxRunes).
func riskNum(v float64) string {
	v = riskClean(v)
	if a := math.Abs(v); v != 0 && (a >= 1e15 || a < 1e-5) {
		return strconv.FormatFloat(v, 'g', -1, 64)
	}
	return strconv.FormatFloat(v, 'f', -1, 64)
}
