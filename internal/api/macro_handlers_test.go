package api

// macro_handlers_test.go — offline tests for handleMacro via the
// macro.SnapshotReader interface (the stooq worker is stubbed entirely).
// Production wiring uses api.SetMacroStore(macro.NewStore()); tests set the
// package global to a fakeMacroReader and reset it after.
//
// Coverage:
//   - nil store → 200, regime:"unknown" (zero lamps carry a value), composite:null,
//     tradfin_ok:false, 5 empty lamps (ok:false), 3 null-coef corr, fng:null,
//     calendar non-nil.
//   - populated fake → regime/composite/lamps (ok:true)/correlations assembled;
//     tradfin_ok:true; idea safe.
//   - stale quotes → tradfin_as_of echoes the old ts; tradfin_market_open is the
//     CLOCK-based futures week (macro.TradfinWindowOpen), not staleness.
//   - all-null lamps (the team-testing honesty defect) → regime:"unknown",
//     composite:null, tradfin_ok:false, per-lamp ok:false, as_of propagated from
//     stale N/D quotes, generated_idea "" (never "signals are split").
//   - calendar is always a non-nil slice.
//
// tradfin_market_open depends on the wall clock (no injectable now in the
// handler), so tests assert it via expectedOpenNow(), which self-disarms in the
// astronomically unlikely case the test run straddles a window boundary second.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/internal/macro"
)

// fakeMacroReader implements macro.SnapshotReader with canned data.
type fakeMacroReader struct {
	latest map[string]macro.Quote
	corr   map[string]*float64 // keyed by symX
	points int
	fng    macro.FnG
	hasFnG bool
}

func (f *fakeMacroReader) Latest() map[string]macro.Quote {
	out := make(map[string]macro.Quote, len(f.latest))
	for k, v := range f.latest {
		out[k] = v
	}
	return out
}

func (f *fakeMacroReader) Correlation(_, symB string) *float64 { return f.corr[symB] }
func (f *fakeMacroReader) PointCount() int                     { return f.points }
func (f *fakeMacroReader) FnG() (macro.FnG, bool)              { return f.fng, f.hasFnG }

var _ macro.SnapshotReader = (*fakeMacroReader)(nil)

func withMacroStore(t *testing.T, r macro.SnapshotReader) {
	t.Helper()
	prev := macroStore
	macroStore = r
	t.Cleanup(func() { macroStore = prev })
}

func runMacroRequest(t *testing.T) (int, macro.Response) {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/macro", nil)
	rec := httptest.NewRecorder()
	handleMacro(rec, req)

	res := rec.Result()
	defer res.Body.Close()

	var body macro.Response
	if res.StatusCode == http.StatusOK {
		if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
			t.Fatalf("decode body: %v", err)
		}
	}
	return res.StatusCode, body
}

// checkTradfinOpen asserts tradfin_market_open matches the clock-based window,
// bracketing the request (before/after) so a run that straddles the exact
// Fri-21:00 / Sun-22:00 boundary second skips instead of flaking. The window's
// own boundaries are pinned deterministically in macro/compute_test.go.
func checkTradfinOpen(t *testing.T, before bool, got bool) {
	t.Helper()
	after := macro.TradfinWindowOpen(time.Now().UTC())
	if before != after {
		t.Logf("window boundary crossed mid-test; skipping the open assertion")
		return
	}
	if got != before {
		t.Errorf("tradfin_market_open = %v, want %v (clock-based futures week)", got, before)
	}
}

func TestMacro_NilStore(t *testing.T) {
	withMacroStore(t, nil)
	openBefore := macro.TradfinWindowOpen(time.Now().UTC())
	code, body := runMacroRequest(t)
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	// Zero lamps carry a value → the regime is honestly UNKNOWN, never a
	// fabricated "mixed" (the team-testing defect).
	if body.Regime != macro.RegimeUnknown {
		t.Errorf("regime = %q, want unknown (no data at all)", body.Regime)
	}
	if body.Composite != nil {
		t.Errorf("composite = %v, want null", *body.Composite)
	}
	// Market-open is the futures-week clock — a statement about the market,
	// valid (and independent of data) even with a nil store.
	checkTradfinOpen(t, openBefore, body.TradfinMarketOpen)
	if body.TradfinOk {
		t.Errorf("tradfin_ok = true, want false (no data)")
	}
	if body.TradfinAsOf != "" {
		t.Errorf("tradfin_as_of = %q, want empty", body.TradfinAsOf)
	}
	if len(body.Lamps) != 5 {
		t.Errorf("lamps len = %d, want 5", len(body.Lamps))
	}
	for _, l := range body.Lamps {
		if l.Value != nil || l.Status != "" || l.OK {
			t.Errorf("nil-store lamp %s = %+v, want empty value/status and ok:false", l.Key, l)
		}
	}
	if len(body.Correlations) != 3 {
		t.Errorf("correlations len = %d, want 3", len(body.Correlations))
	}
	for _, c := range body.Correlations {
		if c.Coef != nil {
			t.Errorf("nil-store corr %s coef = %v, want null", c.Pair, *c.Coef)
		}
	}
	if body.Fng != nil {
		t.Errorf("fng = %+v, want null", body.Fng)
	}
	if body.GeneratedIdea != "" {
		t.Errorf("generated_idea = %q, want empty", body.GeneratedIdea)
	}
	// Calendar must serialise as [] not null.
	if body.Calendar == nil {
		t.Errorf("calendar = null, want [] (non-nil slice)")
	}
	// captured_at present + valid.
	if _, err := time.Parse(time.RFC3339, body.CapturedAt); err != nil {
		t.Errorf("captured_at = %q, not valid RFC3339: %v", body.CapturedAt, err)
	}
}

func TestMacro_Populated(t *testing.T) {
	// A fresh weekday timestamp so the market reads OPEN. Use now so the staleness
	// check passes regardless of when the test runs... but weekday matters, so
	// pin to a known Tuesday and override via a recent AsOf relative to now is
	// fragile. Instead assert the FIELDS that don't depend on the wall clock here,
	// and cover the open/closed logic explicitly in the weekend test below.
	asOf := time.Now().UTC().Add(-2 * time.Minute) // fresh (<15min)
	reader := &fakeMacroReader{
		latest: map[string]macro.Quote{
			macro.SymDXY:   {Symbol: macro.SymDXY, Price: 98.85, Open: 99.05, AsOf: asOf, OK: true}, // session down → tailwind
			macro.SymRates: {Symbol: macro.SymRates, Price: 4.40, Open: 4.45, AsOf: asOf, OK: true}, // session down → tailwind
			macro.SymVIX:   {Symbol: macro.SymVIX, Price: 17.5, Open: 17.5, AsOf: asOf, OK: true},   // <18 → tailwind
			macro.SymSPX:   {Symbol: macro.SymSPX, Price: 7580, Open: 7530, AsOf: asOf, OK: true},   // session up → tailwind
			macro.SymGold:  {Symbol: macro.SymGold, Price: 4546, Open: 4540, AsOf: asOf, OK: true},  // session up small → neutral
		},
		corr: map[string]*float64{
			macro.SymSPX:  ptrF(0.78),
			macro.SymGold: ptrF(0.12),
			macro.SymDXY:  ptrF(-0.64),
		},
		points: 52,
		fng:    macro.FnG{Value: 23, Label: "Extreme Fear", OK: true},
		hasFnG: true,
	}
	withMacroStore(t, reader)

	code, body := runMacroRequest(t)
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}

	// 4 tailwinds + 1 neutral, all weights active → composite should be high
	// (risk_on). DXY25 + Rates15 + VIX25 + SPX25 tailwind = 90; Gold10 neutral =
	// 5 → 95/100 = 95.
	if body.Composite == nil {
		t.Fatalf("composite = null, want a number")
	}
	if *body.Composite != 95 {
		t.Errorf("composite = %d, want 95", *body.Composite)
	}
	if body.Regime != macro.RegimeRiskOn {
		t.Errorf("regime = %q, want risk_on", body.Regime)
	}

	// Lamps: 5, with values + statuses set.
	if len(body.Lamps) != 5 {
		t.Fatalf("lamps len = %d, want 5", len(body.Lamps))
	}
	byKey := map[string]macro.Lamp{}
	for _, l := range body.Lamps {
		byKey[l.Key] = l
	}
	if byKey[macro.KeyDXY].Status != macro.StatusTailwind {
		t.Errorf("dxy status = %q, want tailwind", byKey[macro.KeyDXY].Status)
	}
	if byKey[macro.KeyGold].Status != macro.StatusNeutral {
		t.Errorf("gold status = %q, want neutral", byKey[macro.KeyGold].Status)
	}
	if byKey[macro.KeyDXY].Value == nil || *byKey[macro.KeyDXY].Value != 98.85 {
		t.Errorf("dxy value = %v, want 98.85", byKey[macro.KeyDXY].Value)
	}
	if byKey[macro.KeyDXY].DeltaPct == nil {
		t.Errorf("dxy delta_pct = nil, want a value (Open set)")
	}
	// Data-presence flags: every lamp carries a value here → ok:true each and
	// tradfin_ok:true at the top level.
	for key, l := range byKey {
		if !l.OK {
			t.Errorf("lamp %s ok = false, want true (value present)", key)
		}
	}
	if !body.TradfinOk {
		t.Errorf("tradfin_ok = false, want true (5 valued lamps)")
	}

	// Correlations forwarded with labels.
	if len(body.Correlations) != 3 {
		t.Fatalf("correlations len = %d, want 3", len(body.Correlations))
	}
	corrByPair := map[string]macro.Correlation{}
	for _, c := range body.Correlations {
		corrByPair[c.Pair] = c
	}
	if corrByPair[macro.PairBTCSPX].Coef == nil || *corrByPair[macro.PairBTCSPX].Coef != 0.78 {
		t.Errorf("btc_spx coef = %v, want 0.78", corrByPair[macro.PairBTCSPX].Coef)
	}
	if corrByPair[macro.PairBTCSPX].Label != "moving like stocks" {
		t.Errorf("btc_spx label = %q, want 'moving like stocks'", corrByPair[macro.PairBTCSPX].Label)
	}
	if corrByPair[macro.PairBTCDXY].Label != "inverse to the dollar" {
		t.Errorf("btc_dxy label = %q, want 'inverse to the dollar'", corrByPair[macro.PairBTCDXY].Label)
	}
	if corrByPair[macro.PairBTCSPX].Window == "" {
		t.Errorf("window string empty, want a description")
	}

	// F&G forwarded.
	if body.Fng == nil || body.Fng.Value != 23 {
		t.Errorf("fng = %+v, want value 23", body.Fng)
	}

	// generated_idea present + SAFE (no banned words).
	if body.GeneratedIdea == "" {
		t.Errorf("generated_idea empty, want a sentence for risk_on")
	}
	if !macro.IsDiagnosisSafe(body.GeneratedIdea) {
		t.Errorf("generated_idea contains a banned word: %q", body.GeneratedIdea)
	}
}

func TestMacro_StaleQuotes(t *testing.T) {
	// A months-old Friday close: tradfin_as_of echoes that old ts (staleness is
	// visible, never hidden), lamps still carry their values + dates, and
	// tradfin_market_open follows the CLOCK (futures week), not staleness —
	// data honesty is carried by tradfin_ok / per-lamp ok instead.
	friday := time.Date(2026, 5, 29, 22, 15, 10, 0, time.UTC) // a Friday close
	reader := &fakeMacroReader{
		latest: map[string]macro.Quote{
			macro.SymSPX: {Symbol: macro.SymSPX, Price: 7580.1, AsOf: friday, OK: true},
			macro.SymDXY: {Symbol: macro.SymDXY, Price: 98.85, AsOf: friday, OK: true},
		},
		corr:   map[string]*float64{},
		points: 0,
		fng:    macro.FnG{Value: 54, Label: "Greed", OK: true},
		hasFnG: true,
	}
	withMacroStore(t, reader)

	openBefore := macro.TradfinWindowOpen(time.Now().UTC())
	code, body := runMacroRequest(t)
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	checkTradfinOpen(t, openBefore, body.TradfinMarketOpen)
	want := friday.Format(time.RFC3339)
	if body.TradfinAsOf != want {
		t.Errorf("tradfin_as_of = %q, want %q", body.TradfinAsOf, want)
	}
	// Two lamps DO carry (stale) values → tradfin_ok true, regime NOT unknown
	// (partial data stays "mixed" — composite is null under 3 active lamps).
	if !body.TradfinOk {
		t.Errorf("tradfin_ok = false, want true (2 valued lamps)")
	}
	if body.Regime != macro.RegimeMixed {
		t.Errorf("regime = %q, want mixed (real but insufficient lamps)", body.Regime)
	}
	// Crypto F&G is still live even when tradfin is closed.
	if body.Fng == nil || !body.Fng.OK {
		t.Errorf("fng = %+v, want live (crypto is 24/7)", body.Fng)
	}
	// Cold ring → all coefs null.
	for _, c := range body.Correlations {
		if c.Coef != nil {
			t.Errorf("corr %s coef = %v, want null (cold ring)", c.Pair, *c.Coef)
		}
	}
}

// TestMacro_AllNullUnknown is the team-testing honesty defect, reproduced:
// tradfin closed, every lamp quote N/D → the response must ADMIT it (regime
// "unknown", no idea line) instead of a confident "mixed / signals are split"
// built on zero inputs. Stale dates still propagate into per-lamp as_of and
// tradfin_as_of ("last time we had data" is a real fact).
func TestMacro_AllNullUnknown(t *testing.T) {
	friday := time.Date(2026, 8, 14, 20, 55, 0, 0, time.UTC)
	older := time.Date(2026, 8, 14, 17, 0, 0, 0, time.UTC)
	reader := &fakeMacroReader{
		latest: map[string]macro.Quote{
			// All five lamp symbols N/D. Three still carry their last known
			// date (from the CSV row / the store's carry-forward), two are
			// fully dark.
			macro.SymSPX:   {Symbol: macro.SymSPX, OK: false, AsOf: friday},
			macro.SymDXY:   {Symbol: macro.SymDXY, OK: false, AsOf: older},
			macro.SymVIX:   {Symbol: macro.SymVIX, OK: false, AsOf: older},
			macro.SymRates: {Symbol: macro.SymRates, OK: false},
			macro.SymGold:  {Symbol: macro.SymGold, OK: false},
		},
		corr:   map[string]*float64{},
		fng:    macro.FnG{Value: 61, Label: "Greed", OK: true},
		hasFnG: true,
	}
	withMacroStore(t, reader)

	code, body := runMacroRequest(t)
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if body.Regime != macro.RegimeUnknown {
		t.Errorf("regime = %q, want unknown (ALL lamps null)", body.Regime)
	}
	if body.Composite != nil {
		t.Errorf("composite = %v, want null", *body.Composite)
	}
	if body.TradfinOk {
		t.Errorf("tradfin_ok = true, want false (zero valued lamps)")
	}
	// The knowledge claim must be gone: no "signals are split" off no inputs.
	if body.GeneratedIdea != "" {
		t.Errorf("generated_idea = %q, want empty (no data, no claim)", body.GeneratedIdea)
	}

	byKey := map[string]macro.Lamp{}
	for _, l := range body.Lamps {
		byKey[l.Key] = l
	}
	for key, l := range byKey {
		if l.Value != nil || l.OK || l.Status != "" {
			t.Errorf("lamp %s = %+v, want valueless ok:false", key, l)
		}
	}
	// as_of propagation: dated N/D quotes ship their last known date; fully
	// dark ones stay "".
	if got, want := byKey[macro.KeySPX].AsOf, friday.Format(time.RFC3339); got != want {
		t.Errorf("spx as_of = %q, want %q (stale date propagated)", got, want)
	}
	if got, want := byKey[macro.KeyDXY].AsOf, older.Format(time.RFC3339); got != want {
		t.Errorf("dxy as_of = %q, want %q", got, want)
	}
	if got := byKey[macro.KeyRates].AsOf; got != "" {
		t.Errorf("rates as_of = %q, want empty (never saw a date)", got)
	}
	// tradfin_as_of = freshest KNOWN date across the (all-N/D) lamps.
	if want := friday.Format(time.RFC3339); body.TradfinAsOf != want {
		t.Errorf("tradfin_as_of = %q, want %q (freshest known date)", body.TradfinAsOf, want)
	}
	// F&G (crypto side) still flows — the card can honestly state what IS known.
	if body.Fng == nil || body.Fng.Value != 61 {
		t.Errorf("fng = %+v, want value 61", body.Fng)
	}
}

func TestMacro_ThreeNDLampsCompositeNull(t *testing.T) {
	asOf := time.Now().UTC().Add(-1 * time.Minute)
	reader := &fakeMacroReader{
		latest: map[string]macro.Quote{
			macro.SymDXY: {Symbol: macro.SymDXY, Price: 98.85, Open: 99.0, AsOf: asOf, OK: true},
			macro.SymVIX: {Symbol: macro.SymVIX, Price: 17.5, AsOf: asOf, OK: true},
			// rates, spx, gold all missing → N/D → 3 lamps out → composite null.
		},
		corr: map[string]*float64{},
	}
	withMacroStore(t, reader)

	code, body := runMacroRequest(t)
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if body.Composite != nil {
		t.Errorf("composite = %v, want null (3 lamps N/D)", *body.Composite)
	}
	// Two lamps DO carry values → "mixed" (real but insufficient), NOT
	// "unknown" (which is reserved for zero real lamps).
	if body.Regime != macro.RegimeMixed {
		t.Errorf("regime = %q, want mixed (null composite, real lamps)", body.Regime)
	}
	if !body.TradfinOk {
		t.Errorf("tradfin_ok = false, want true (2 valued lamps)")
	}
	// generated_idea: mixed regime still produces the safe "split" line — with
	// real lamps behind it, that's a legitimate claim.
	if body.GeneratedIdea != "" && !macro.IsDiagnosisSafe(body.GeneratedIdea) {
		t.Errorf("generated_idea banned: %q", body.GeneratedIdea)
	}
}

// TestMacro_CalendarAlwaysSlice: even with a populated store, calendar is a
// non-nil slice (empty when macrocal isn't initialised in tests).
func TestMacro_CalendarAlwaysSlice(t *testing.T) {
	withMacroStore(t, &fakeMacroReader{latest: map[string]macro.Quote{}, corr: map[string]*float64{}})
	code, body := runMacroRequest(t)
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if body.Calendar == nil {
		t.Errorf("calendar = null, want [] (non-nil)")
	}
}

func ptrF(f float64) *float64 { return &f }
