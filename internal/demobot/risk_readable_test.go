package demobot

// risk_readable_test.go — Risk Calculator stage 1 (2026-09-15): the card's
// words, states and machine fields. The formula is untouched:
// (balance × risk%) ÷ |entry − stop|, the same numbers on the same inputs as
// on 36ee763 (TestRiskFormulaRegression pins them as literals captured there).

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"math/big"
	"math/rand"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

var riskTestNow = time.Date(2026, 9, 15, 20, 11, 0, 0, time.UTC)

func riskAgents() *Agents {
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	ag.now = func() time.Time { return riskTestNow }
	return ag
}

func riskBody(c Card) string { return c.renderBody() }

// ── formula regression: literals captured on 36ee763 ────────────────────────

type riskBase struct {
	in                 [4]float64
	amt, per, size, nt float64
}

// Captured on the base commit with fmt %v (shortest exact float repr).
var riskBaseCases = []riskBase{
	{[4]float64{10000, 1, 64000, 62500}, 100, 1500, 0.06666666666666667, 4266.666666666667},
	{[4]float64{10000, 1, 1.1, 1.095}, 100, 0.0050000000000001155, 19999.999999999538, 21999.999999999494},
	{[4]float64{5000, 2, 3400, 3500}, 100, 100, 1, 3400},
	{[4]float64{12345.678, 0.333, 1.23456, 1.23321}, 41.111107739999994, 0.0013500000000001844, 30452.672399995838, 37595.65123813887},
	{[4]float64{1e300, 100, 2, 1}, 1e+300, 1, 1e+300, 2e+300},
	{[4]float64{10000, 1, 1.10005, 1.1}, 100, 4.999999999988347e-05, 2.0000000000046613e+06, 2.2001000000051274e+06},
	{[4]float64{1e-3, 0.01, 100000, 99999.99}, 1.0000000000000001e-07, 0.00999999999476131, 1.000000000523869e-05, 1.000000000523869},
	{[4]float64{1e15, 100, 1e-9, 2e-9}, 1e+15, 1e-09, 1e+24, 1e+15},
}

func TestRiskFormulaRegression(t *testing.T) {
	ag := riskAgents()
	for _, bc := range riskBaseCases {
		r, err := calcRisk(bc.in[0], bc.in[1], bc.in[2], bc.in[3])
		if err != nil {
			t.Fatalf("%v: %v", bc.in, err)
		}
		if r.RiskAmount != bc.amt || r.PerUnit != bc.per || r.Size != bc.size || r.Notional != bc.nt {
			t.Errorf("%v: got amt=%v per=%v size=%v notional=%v, base amt=%v per=%v size=%v notional=%v",
				bc.in, r.RiskAmount, r.PerUnit, r.Size, r.Notional, bc.amt, bc.per, bc.size, bc.nt)
		}
		// The machine readout serves the same floats, bit for bit.
		c := ag.RiskCard(bc.in[:], false, nil)
		if c.Risk == nil {
			t.Fatalf("%v: no risk readout", bc.in)
		}
		rr := c.Risk
		if rr.PriceRiskAccountCcy != bc.amt || rr.PriceDistance != bc.per || rr.ResultAbstractUnits != bc.size {
			t.Errorf("%v: readout %v/%v/%v differs from the formula", bc.in, rr.PriceRiskAccountCcy, rr.PriceDistance, rr.ResultAbstractUnits)
		}
		if rr.Inputs.Balance != bc.in[0] || rr.Inputs.RiskPct != bc.in[1] || rr.Inputs.Entry != bc.in[2] || rr.Inputs.Stop != bc.in[3] {
			t.Errorf("%v: inputs echoed as %+v", bc.in, rr.Inputs)
		}
	}
}

// ── golden texts ─────────────────────────────────────────────────────────────

func TestRiskGoldenCalculated(t *testing.T) {
	c := riskAgents().RiskCard([]float64{10000, 1, 64000, 62500}, false, nil)
	want := "⚪ <b>Risk Calculator</b>\n" +
		"<b>Calculated, instrument model not confirmed: 0.06666 abstract units</b>\n" +
		"• Valid only if a 1.0 price move changes one unit&#39;s value by 1.0 in the account currency\n" +
		"• Fits e.g. BTC/USD spot on a USD account; the calculator does not know your instrument\n" +
		"• Planned price risk: 100 (account currency), before fees and slippage\n" +
		"• Basis: 1% of balance 10000; entry 64000, stop 62500\n" +
		"• Check: 0.06666 (rounded down) × distance 1500 = 99.99 ≤ 100\n" +
		"• Not supported: FX lots, futures with a multiplier, CFDs, account-currency conversion\n" +
		"• Position sizing math only — not a trade suggestion.\n"
	if got := riskBody(c); got != want {
		t.Errorf("calculated card:\ngot:\n%s\nwant:\n%s", got, want)
	}
	footer := "<i>Analytics, not financial advice · AlphaVizor · 2026-09-15 20:11 UTC · calculation time, no market data read</i>"
	if html := c.RenderHTML(); !strings.HasSuffix(html, "\n"+footer) {
		t.Errorf("footer must say calculation time:\n%s", html)
	}
	if c.Status != statusOK || c.Emoji != emojiNeutral || c.Blocks != nil || c.Levels != nil || c.Confidence != nil {
		t.Errorf("status/emoji/blocks/levels/confidence: %v %q %v %v %v", c.Status, c.Emoji, c.Blocks, c.Levels, c.Confidence)
	}
}

func TestRiskGoldenExample(t *testing.T) {
	c := riskAgents().RiskCard(riskExampleValues, true, nil)
	want := "⚪ <b>Risk Calculator</b>\n" +
		"<b>Example (BTC/USD spot, USD account), calculated: 0.06666 abstract units</b>\n" +
		"• Valid only if a 1.0 price move changes one unit&#39;s value by 1.0 in the account currency\n" +
		"• Fits e.g. BTC/USD spot on a USD account; the calculator does not know your instrument\n" +
		"• Planned price risk: 100 (account currency), before fees and slippage\n" +
		"• Basis: 1% of balance 10000; entry 64000, stop 62500\n" +
		"• Check: 0.06666 (rounded down) × distance 1500 = 99.99 ≤ 100\n" +
		"• Not supported: FX lots, futures with a multiplier, CFDs, account-currency conversion\n" +
		"• Position sizing math only — not a trade suggestion.\n" +
		"• Example inputs read as BTC/USD spot on a USD account: here 1 abstract unit = 1 BTC\n" +
		"• Usage: /risk &lt;balance&gt; &lt;risk%&gt; &lt;entry&gt; &lt;stop&gt;\n"
	if got := riskBody(c); got != want {
		t.Errorf("example card:\ngot:\n%s\nwant:\n%s", got, want)
	}
}

// The EURUSD-shaped input the plan was written about: no "$", no notional,
// no bare "units"; float noise of the distance does not reach the text.
func TestRiskGoldenEURUSDShaped(t *testing.T) {
	c := riskAgents().RiskCard([]float64{10000, 1, 1.1, 1.095}, false, nil)
	if c.Verdict != "Calculated, instrument model not confirmed: 20000 abstract units" {
		t.Errorf("verdict: %q", c.Verdict)
	}
	wantFacts := map[int]string{
		2: "Planned price risk: 100 (account currency), before fees and slippage",
		3: "Basis: 1% of balance 10000; entry 1.1, stop 1.095",
		4: "Check: 20000 (rounded down) × distance 0.005 = 100 ≤ 100",
	}
	for i, w := range wantFacts {
		if c.Facts[i] != w {
			t.Errorf("fact %d: got %q, want %q", i, c.Facts[i], w)
		}
	}
}

func TestRiskGoldenNotCalculated(t *testing.T) {
	ag := riskAgents()
	usage := riskUsage
	example := "Example: /risk 10000 1 64000 62500 (BTC/USD spot, USD account)"
	cases := []struct {
		name    string
		args    []float64
		err     error
		verdict string
		facts   []string
	}{
		{"two args", nil, riskArityError{Got: 2}, "Not calculated: not enough parameters (2 of 4)", []string{usage, example}},
		{"zero args", nil, riskArityError{Got: 0}, "Not calculated: not enough parameters (0 of 4)", []string{usage, example}},
		{"five args", nil, riskArityError{Got: 5}, "Not calculated: too many parameters (5, need 4)", []string{usage, example}},
		{"not a number", nil, errors.New(`"abc" is not a number`), "Not calculated: a parameter is not a number",
			[]string{`"abc" is not a number`, usage, example}},
		{"entry == stop", []float64{10000, 1, 64000, 64000}, nil, "Not calculated: those numbers don't work",
			[]string{"stop must differ from entry", usage}},
		{"negative entry", []float64{10000, 1, -5, 90}, nil, "Not calculated: those numbers don't work",
			[]string{"entry and stop must be positive", usage}},
		{"zero balance", []float64{0, 1, 100, 90}, nil, "Not calculated: those numbers don't work",
			[]string{"balance must be positive", usage}},
		{"negative balance", []float64{-100, 1, 100, 90}, nil, "Not calculated: those numbers don't work",
			[]string{"balance must be positive", usage}},
		{"zero risk", []float64{1000, 0, 100, 90}, nil, "Not calculated: those numbers don't work",
			[]string{"risk% must be in (0, 100]", usage}},
		{"risk over 100", []float64{1000, 250, 100, 90}, nil, "Not calculated: those numbers don't work",
			[]string{"risk% must be in (0, 100]", usage}},
		{"NaN", []float64{math.NaN(), 1, 64000, 62500}, nil, "Not calculated: those numbers don't work",
			[]string{"every number must be finite", usage}},
		{"Inf", []float64{10000, 1, math.Inf(1), 62500}, nil, "Not calculated: those numbers don't work",
			[]string{"every number must be finite", usage}},
		{"overflow", []float64{1.7e308, 100, 2, 1}, nil, "Not calculated: those numbers don't work",
			[]string{"numbers too large or too small to calculate", usage}},
		{"subnormal distance", []float64{10000, 1, 1e-320, 2e-320}, nil, "Not calculated: those numbers don't work",
			[]string{"numbers too large or too small to calculate", usage}},
	}
	for _, tc := range cases {
		c := ag.RiskCard(tc.args, false, tc.err)
		if c.Verdict != tc.verdict {
			t.Errorf("%s: verdict %q, want %q", tc.name, c.Verdict, tc.verdict)
		}
		if strings.Join(c.Facts, "|") != strings.Join(tc.facts, "|") {
			t.Errorf("%s: facts %q, want %q", tc.name, c.Facts, tc.facts)
		}
		if c.Risk != nil {
			t.Errorf("%s: a card without a result must carry no readout", tc.name)
		}
	}
}

// A Telegram token can be thousands of characters: the fact stays ≤110.
func TestRiskLongTokenTruncated(t *testing.T) {
	_, err := parseMoney(strings.Repeat("x", 4000))
	c := riskAgents().RiskCard(nil, false, err)
	for _, f := range append([]string{c.Verdict}, c.Facts...) {
		if n := utf8.RuneCountInString(f); n > riskFactMaxRunes {
			t.Errorf("%d runes: %q", n, f)
		}
	}
}

// ── the rule for printed numbers ─────────────────────────────────────────────

func TestRiskPrintRule(t *testing.T) {
	cases := []struct {
		v              float64
		floor4, money4 string
	}{
		{0.06666666666666667, "0.06666", "0.06667"},
		{19999.999999999538, "20000", "20000"},
		{30452.672399995838, "30450", "30452.68"},
		{1e+300, "1e+300", "1e+300"},
		{2.0000000000046613e+06, "2000000", "2000000"},
		{1.000000000523869e-05, "0.00001", "0.00001001"},
		{1e+24, "1e+24", "1e+24"},
		{41.111107739999994, "41.11", "41.12"},
		{99.99000000000001, "99.99", "99.99"},
		{110.00000000000001, "110", "110"},
		{1.0000000000000001e-07, "1e-07", "1e-07"},
		{1234567.891, "1234000", "1234567.9"},
	}
	for _, tc := range cases {
		if got := riskNum(riskFloor4(tc.v)); got != tc.floor4 {
			t.Errorf("size %v: got %q, want %q", tc.v, got, tc.floor4)
		}
		if got := riskMoney(tc.v); got != tc.money4 {
			t.Errorf("money %v: got %q, want %q", tc.v, got, tc.money4)
		}
	}
}

// The print rule, checked exactly (math/big) with no other tolerance:
//   - the shown size is at most the formula result, plus at most the noise the
//     rule drops (half a unit in the result's 12th significant digit, and only
//     when it has more than 12);
//   - a printed price risk is never below the computed one, except for a value
//     with more than 12 significant digits, and then by at most half a unit in
//     its 12th significant digit (≤ 5e-12 relative);
//   - the Check value never exceeds the printed planned price risk.
//
// Sums from 1e11 up exercise the boundary (a case must actually undershoot).
func TestRiskShownNumbersSafe(t *testing.T) {
	ag := riskAgents()
	inputs := append([][4]float64{}, riskBaseInputs()...)
	for _, bal := range []float64{1, 777.77, 10000, 123456.789, 3e9, 123456789012.345,
		9.87654321098765e13, 3.33333333333333e12, 1.23456789012345e15} {
		for _, pct := range []float64{0.1, 0.333, 1, 2.5, 33.3333, 100} {
			for _, es := range [][2]float64{{64000, 62500}, {1.1, 1.095}, {1.23456, 1.23321}, {3400, 3500},
				{0.00001234, 0.00001233}, {150.25, 149.9}} {
				inputs = append(inputs, [4]float64{bal, pct, es[0], es[1]})
			}
		}
	}
	for _, in := range inputs {
		c := ag.RiskCard(in[:], false, nil)
		rr := c.Risk
		if rr == nil {
			t.Fatalf("%v: no readout (%s)", in, c.Verdict)
		}
		riskCheckOver(t, in, "shown size", riskBigF(rr.ShownAbstractUnits), rr.ResultAbstractUnits)
		if rr.ShownAbstractUnits < rr.ResultAbstractUnits*(1-1e-3) {
			t.Errorf("%v: shown size %v loses more than 0.1%% of %v", in, rr.ShownAbstractUnits, rr.ResultAbstractUnits)
		}
		budget := strings.TrimPrefix(c.Facts[2], "Planned price risk: ")
		budget, _, _ = strings.Cut(budget, " (")
		riskCheckUnder(t, in, "planned price risk", rr.PriceRiskAccountCcy, budget)
		_, tail, _ := strings.Cut(c.Facts[4], " = ")
		check, shownBudget, _ := strings.Cut(tail, " ≤ ")
		if shownBudget != budget {
			t.Errorf("%v: Check budget %q != printed budget %q", in, shownBudget, budget)
		}
		if riskExact(t, check).Cmp(riskExact(t, budget)) > 0 {
			t.Errorf("%v: Check %s above the planned price risk %s", in, check, budget)
		}
		riskCheckUnder(t, in, "check", rr.PriceRiskAtShownUnits, check)
	}
	// Sums from 1e11 up with more than 12 significant digits: the printed
	// risk is below the computed one, within the bound and never more.
	underSeen := 0
	for _, v := range []float64{123456789012.345, 987654321012.3449, 1234567890.12345, 4.44444444444444e14} {
		p := riskMoney(v)
		if riskPrintedF(t, p).Cmp(riskBigF(v)) < 0 {
			underSeen++
		}
		riskCheckUnder(t, [4]float64{v}, "money", v, p)
	}
	if underSeen == 0 {
		t.Error("no case exercised the documented undershoot boundary")
	}
}

// riskBigF is v exactly; riskExact parses a printed decimal exactly.
func riskBigF(v float64) *big.Float { return new(big.Float).SetPrec(4096).SetFloat64(v) }

func riskExact(t *testing.T, s string) *big.Float {
	t.Helper()
	f, _, err := big.ParseFloat(s, 10, 4096, big.ToNearestEven)
	if err != nil {
		t.Fatalf("printed %q: %v", s, err)
	}
	return f
}

// riskNoise is the noise the print rule drops for v: half a unit in its 12th
// significant digit, and whether v has more than 12 significant digits.
func riskNoise(t *testing.T, v float64) (*big.Float, bool) {
	t.Helper()
	mant, exp, _ := strings.Cut(strconv.FormatFloat(v, 'e', -1, 64), "e")
	e, err := strconv.Atoi(exp)
	if err != nil {
		t.Fatal(err)
	}
	return riskExact(t, "5e"+strconv.Itoa(e-12)), len(strings.Replace(mant, ".", "", 1)) > 12
}

// riskPrintedF is the float64 a printed number stands for: the card prints
// the shortest decimal of a float ("0.01" is the float 0.01, whose binary
// expansion is 0.01000000000000000020…), so the comparison is float to float,
// the difference taken exactly.
func riskPrintedF(t *testing.T, s string) *big.Float {
	t.Helper()
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsInf(f, 0) || math.IsNaN(f) {
		t.Fatalf("printed %q: %v %v", s, f, err)
	}
	return riskBigF(f)
}

// riskCheckUnder: printed must not be below raw, except within the noise.
func riskCheckUnder(t *testing.T, in [4]float64, what string, raw float64, printed string) {
	t.Helper()
	under := new(big.Float).SetPrec(4096).Sub(riskBigF(raw), riskPrintedF(t, printed))
	if under.Sign() <= 0 {
		return
	}
	bound, long := riskNoise(t, raw)
	if !long || under.Cmp(bound) > 0 {
		t.Errorf("%v: %s printed %s is below %v by %s (allowed %s, >12 digits %v)",
			in, what, printed, raw, under.Text('g', 6), bound.Text('g', 3), long)
	}
}

// riskCheckOver: shown must not exceed raw, except within the noise.
func riskCheckOver(t *testing.T, in [4]float64, what string, shown *big.Float, raw float64) {
	t.Helper()
	over := new(big.Float).SetPrec(4096).Sub(shown, riskBigF(raw))
	if over.Sign() <= 0 {
		return
	}
	bound, long := riskNoise(t, raw)
	if !long || over.Cmp(bound) > 0 {
		t.Errorf("%v: %s %s above %v by %s (allowed %s)", in, what, shown.Text('g', 17), raw, over.Text('g', 6), bound.Text('g', 3))
	}
}

// The review's cases: tiny finite results used to print NaN and answer 500.
func TestRiskTinyResultsFinite(t *testing.T) {
	ag := riskAgents()
	c := ag.RiskCard([]float64{1e-304, 1, 2, 1}, false, nil)
	if c.Verdict != "Calculated, instrument model not confirmed: 1e-306 abstract units" ||
		c.Facts[2] != "Planned price risk: 1e-306 (account currency), before fees and slippage" ||
		c.Facts[4] != "Check: 1e-306 (rounded down) × distance 1 = 1e-306 ≤ 1e-306" {
		t.Errorf("1e-304 balance: %q %q", c.Verdict, c.Facts)
	}
	c = ag.RiskCard([]float64{1, 1, 1e306, 1}, false, nil)
	if c.Verdict != "Calculated, instrument model not confirmed: 1e-308 abstract units" ||
		c.Facts[4] != "Check: 1e-308 (rounded down) × distance 1e+306 = 0.01 ≤ 0.01" {
		t.Errorf("1e306 entry: %q %q", c.Verdict, c.Facts)
	}
	_, srv := newTestAPI(t, deadAgents(t), true)
	for _, q := range []string{"balance=1e-304&risk=1&entry=2&stop=1", "balance=1&risk=1&entry=1e306&stop=1"} {
		code, _, body := httpGet(t, srv.URL+"/agents/risk?"+q)
		if code != 200 || strings.Contains(string(body), "NaN") {
			t.Errorf("%s: %d %s", q, code, body)
		}
	}
}

// A shown number that cannot be represented is an honest range error, never
// NaN or Inf (defensive: the formula's own checks keep real inputs away).
func TestRiskUnprintableIsRangeError(t *testing.T) {
	_, err := riskPrint(riskResult{RiskAmount: math.MaxFloat64, PerUnit: 1, Size: math.MaxFloat64})
	if err == nil || err.Error() != riskErrRange {
		t.Errorf("max float budget: err %v, want %q", err, riskErrRange)
	}
}

// Fuzz over the whole float64 range: no NaN / Inf in any text or JSON, no 500.
func TestRiskFuzzFiniteEverywhere(t *testing.T) {
	rng := rand.New(rand.NewSource(20260916))
	logU := func(lo, hi float64) float64 { return math.Pow(10, lo+rng.Float64()*(hi-lo)) }
	draw := func() [4]float64 {
		b, r, e := logU(-320, 308), math.Min(logU(-8, 2.5), 100), logU(-320, 308)
		s := e * (1 + (rng.Float64()-0.5)*math.Pow(10, -rng.Float64()*15))
		if rng.Intn(3) == 0 {
			s = logU(-320, 308)
		}
		return [4]float64{b, r, e, s}
	}
	ag := riskAgents()
	n, calculated := 20000, 0
	if testing.Short() {
		n = 2000
	}
	for i := 0; i < n; i++ {
		in := draw()
		c := ag.RiskCard(in[:], false, nil)
		for _, l := range riskLines(c) {
			if strings.Contains(l, "NaN") || strings.Contains(l, "Inf") || utf8.RuneCountInString(l) > riskFactMaxRunes {
				t.Fatalf("%v: line %q", in, l)
			}
		}
		if c.Risk == nil {
			if !strings.HasPrefix(c.Verdict, "Not calculated: ") {
				t.Fatalf("%v: no readout on %q", in, c.Verdict)
			}
			continue
		}
		calculated++
		b, err := json.Marshal(cardEnvelope(c))
		if err != nil || strings.Contains(string(b), "NaN") || strings.Contains(string(b), "Inf") {
			t.Fatalf("%v: envelope %v %s", in, err, b)
		}
	}
	if calculated < n/4 {
		t.Errorf("only %d of %d draws calculated: the fuzz does not reach the printer", calculated, n)
	}
	_, srv := newTestAPI(t, deadAgents(t), true)
	for i := 0; i < 400; i++ {
		in := draw()
		q := url.Values{}
		for j, p := range []string{"balance", "risk", "entry", "stop"} {
			q.Set(p, strconv.FormatFloat(in[j], 'g', -1, 64))
		}
		code, _, body := httpGet(t, srv.URL+"/agents/risk?"+q.Encode())
		if (code != 200 && code != 400) || strings.Contains(string(body), "NaN") || strings.Contains(string(body), "Inf") {
			t.Fatalf("%v: %d %s", in, code, body)
		}
	}
}

func riskBaseInputs() [][4]float64 {
	out := make([][4]float64, 0, len(riskBaseCases))
	for _, bc := range riskBaseCases {
		out = append(out, bc.in)
	}
	return out
}

func mustFloat(t *testing.T, s string) float64 {
	t.Helper()
	v, err := parseMoney(s)
	if err != nil {
		t.Fatalf("%q: %v", s, err)
	}
	return v
}

// ── lines, words ─────────────────────────────────────────────────────────────

// Adversarial widths: 12 significant digits everywhere, both number forms.
func riskWideInputs() [][4]float64 {
	return [][4]float64{
		{123456789012.345, 12.3456789012, 0.0000123456789012, 0.0000123456789011},
		{0.0000123456789012, 0.0000123456789012, 98765432101.2345, 98765432101.2344},
		{9.87654321098e+299, 99.9999999999, 1.23456789012e+299, 1.23456789011e+299},
		{1.23456789012e-290, 0.0000123456789012, 1.23456789012e-280, 9.87654321098e-281},
		{999999999999999, 100, 999999999999998, 1},
		{123456789012345, 55.5555555555, 0.00012345678901, 0.00012345678902},
	}
}

func riskAllCards() []Card {
	ag := riskAgents()
	var out []Card
	for _, in := range append(riskBaseInputs(), riskWideInputs()...) {
		out = append(out, ag.RiskCard(in[:], false, nil), ag.RiskCard(in[:], true, nil))
	}
	out = append(out, ag.RiskCard(riskExampleValues, true, nil),
		ag.RiskCard(nil, false, riskArityError{Got: 3}),
		ag.RiskCard(nil, false, errors.New(`"abc" is not a number`)),
		ag.RiskCard([]float64{10000, 1, 64000, 64000}, false, nil),
		ag.RiskCard([]float64{math.NaN(), 1, 1, 2}, false, nil))
	return out
}

func riskLines(c Card) []string {
	lines := append([]string{c.Verdict}, c.Facts...)
	if c.Risk != nil {
		lines = append(lines, c.Risk.ContentLine, c.Risk.Applicability.Condition)
	}
	body := c.RenderHTML()
	lines = append(lines, htmlToPlain(body[strings.LastIndex(body, "\n")+1:]))
	return lines
}

func TestRiskLinesFit(t *testing.T) {
	for _, c := range riskAllCards() {
		for _, l := range riskLines(c) {
			if n := utf8.RuneCountInString(l); n > riskFactMaxRunes {
				t.Errorf("%d runes > %d: %q", n, riskFactMaxRunes, l)
			}
		}
	}
}

var (
	riskBareUnits = regexp.MustCompile(`\bunits\b`)
	riskBannedRes = []*regexp.Regexp{
		regexp.MustCompile(`(?i)max loss`),
		regexp.MustCompile(`(?i)notional`),
		regexp.MustCompile(`(?i)position size:`),
		regexp.MustCompile(`\$`),
		regexp.MustCompile(`\bUSD [0-9]|[0-9] USD\b`),
		regexp.MustCompile(`\b(LONG|SHORT|long|short|buy|sell)\b`),
	}
)

func riskCheckText(t *testing.T, where, s string) {
	t.Helper()
	for _, re := range riskBannedRes {
		if re.MatchString(s) {
			t.Errorf("%s: banned %q in %q", where, re.String(), s)
		}
	}
	for _, loc := range riskBareUnits.FindAllStringIndex(s, -1) {
		if !strings.HasSuffix(s[:loc[0]], "abstract ") {
			t.Errorf("%s: \"units\" without \"abstract\" in %q", where, s)
		}
	}
}

func TestRiskBannedWords(t *testing.T) {
	for _, c := range riskAllCards() {
		for _, l := range riskLines(c) {
			riskCheckText(t, "card", l)
		}
		if c.Risk != nil {
			b, _ := json.Marshal(c.Risk)
			riskCheckText(t, "readout", string(b))
		}
	}
	riskCheckText(t, "how-text", howTexts[keyRisk])
	if !strings.Contains(howTexts[keyRisk], "abstract units") || !strings.Contains(howTexts[keyRisk], "account currency") {
		t.Errorf("how-text must name abstract units and the account currency: %q", howTexts[keyRisk])
	}
}

// ── machine fields ───────────────────────────────────────────────────────────

func TestRiskReadoutGolden(t *testing.T) {
	c := riskAgents().RiskCard([]float64{10000, 1, 64000, 62500}, false, nil)
	env := cardEnvelope(c)
	b, err := json.Marshal(env)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	if _, has := m["blocks"]; has {
		t.Errorf("risk serves no content blocks (the calculator observes no market): %s", m["blocks"])
	}
	want := `{"status":"calculated","formula":"(balance × risk%) ÷ |entry − stop|",` +
		`"inputs":{"balance":10000,"risk_pct":1,"entry":64000,"stop":62500},` +
		`"price_risk_account_ccy":100,"price_distance":1500,"result_abstract_units":0.06666666666666667,` +
		`"shown_abstract_units":0.06666,"price_risk_at_shown_units":99.99,` +
		`"quantity_unit":"abstract_units","account_currency":null,` +
		`"applicability":{"status":"not_verified","condition":"a 1.0 price move changes one unit's value by 1.0 in the account currency",` +
		`"fits_example":"BTC/USD spot on a USD account","not_supported":["fx_lots","futures_multiplier","cfd","account_currency_conversion"]},` +
		`"excludes":["fees","slippage","price_gaps","quantity_step_rounding"],` +
		`"display_rounding":{"size":"down_4_significant","money":"up_to_cents"},` +
		`"reads_market_data":false,"calculated_at":"2026-09-15T20:11:00Z",` +
		`"content_line":"Position sizing only: 0.06666 abstract units for a planned price risk of 100"}`
	if got := string(m["risk"]); got != want {
		t.Errorf("risk readout:\ngot:  %s\nwant: %s", got, want)
	}
	if env.DataAsOf != "2026-09-15T20:11:00Z" {
		t.Errorf("data_as_of stays in the contract, the calculation time: %q", env.DataAsOf)
	}
}

func TestRiskHTTP(t *testing.T) {
	_, srv := newTestAPI(t, deadAgents(t), true)
	status, hdr, body := httpGet(t, srv.URL+"/agents/risk?balance=10000&risk=1&entry=64000&stop=62500")
	if status != 200 {
		t.Fatalf("status %d: %s", status, body)
	}
	if hdr.Get("Last-Modified") != "" {
		t.Errorf("risk carries no validator")
	}
	var env struct {
		Verdict  string          `json:"verdict"`
		DataAsOf string          `json:"data_as_of"`
		Blocks   json.RawMessage `json:"blocks"`
		Risk     *RiskReadout    `json:"risk"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if env.Verdict != "Calculated, instrument model not confirmed: 0.06666 abstract units" {
		t.Errorf("verdict: %q", env.Verdict)
	}
	if env.Risk == nil || env.Risk.CalculatedAt != env.DataAsOf || env.Risk.Status != riskStatusCalculated {
		t.Fatalf("risk readout: %+v (data_as_of %q)", env.Risk, env.DataAsOf)
	}
	if env.Blocks != nil {
		t.Errorf("blocks: %s", env.Blocks)
	}
	for q, msg := range map[string]string{
		"balance=NaN&risk=1&entry=64000&stop=62500":      "every number must be finite",
		"balance=Inf&risk=1&entry=64000&stop=62500":      "every number must be finite",
		"balance=1.7e308&risk=100&entry=2&stop=1":        "numbers too large or too small to calculate",
		"balance=10000&risk=1&entry=1e-320&stop=2e-320":  "numbers too large or too small to calculate",
		"balance=10000&risk=1&entry=64000&stop=abc":      "not a number",
		"balance=10000&risk=1&entry=64000&stop=64000":    "stop must differ from entry",
		"balance=10000&risk=1&entry=64000":               "missing ?stop=",
		"balance=-10000&risk=1&entry=64000&stop=62500":   "balance must be positive",
		"balance=10000&risk=1&entry=64000&stop=-62500.5": "entry and stop must be positive",
	} {
		code, _, b := httpGet(t, srv.URL+"/agents/risk?"+q)
		if code != 400 || !strings.Contains(string(b), msg) {
			t.Errorf("%s: %d %s, want 400 with %q", q, code, b, msg)
		}
	}
}

// Telegram /risk renders the same card as the HTTP body.
func TestRiskTelegramSameText(t *testing.T) {
	bot := testBot("http://127.0.0.1:1")
	ctx := context.Background()
	text, _ := bot.buildReply(ctx, keyRisk, []string{"10,000", "1%", "$64000", "62500"})
	want := stripFooter(riskAgents().RiskCard([]float64{10000, 1, 64000, 62500}, false, nil).RenderHTML())
	if stripFooter(text) != want {
		t.Errorf("telegram /risk:\ngot:\n%s\nwant:\n%s", text, want)
	}
	short, _ := bot.buildReply(ctx, keyRisk, []string{"10000", "1"})
	if !strings.Contains(short, "Not calculated: not enough parameters (2 of 4)") {
		t.Errorf("/risk with 2 numbers: %s", short)
	}
	long, _ := bot.buildReply(ctx, keyRisk, []string{"1", "2", "3", "4", "5"})
	if !strings.Contains(long, "Not calculated: too many parameters (5, need 4)") {
		t.Errorf("/risk with 5 numbers: %s", long)
	}
	bad, _ := bot.buildReply(ctx, keyRisk, []string{"10000", "1", "abc", "62500"})
	if !strings.Contains(bad, "Not calculated: a parameter is not a number") {
		t.Errorf("/risk with a word: %s", bad)
	}
}

// The landing row shows the labeled example.
func TestRiskShowcaseRow(t *testing.T) {
	b := &showcaseBuild{cards: map[string]Card{keyRisk: riskAgents().RiskCard(riskExampleValues, true, nil)}}
	row := b.row(keyRisk)
	if row.Headline != "Example (BTC/USD spot, USD account), calculated: 0.06666 abstract units" {
		t.Errorf("headline: %q", row.Headline)
	}
	riskCheckText(t, "showcase one-liner", row.OneLiner)
	if !row.OK {
		t.Errorf("risk stays live on the landing: %+v", row)
	}
}

// Risk stays out of the push hook: a function of user input.
func TestRiskNotInHook(t *testing.T) {
	for _, tg := range hookTargets() {
		if tg.Agent == keyRisk || strings.Contains(tg.Path, "/agents/risk") {
			t.Errorf("risk in the hook address set: %+v", tg)
		}
	}
}
