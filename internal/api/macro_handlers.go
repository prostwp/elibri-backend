package api

// macro_handlers.go — Macro Sentiment REST surface (GET /api/v1/macro).
//
// AUTH: registered on the same mux that auth.Middleware wraps in NewRouter, and
// allowlisted by EXACT match "/api/v1/macro" in internal/auth/jwt.go publicPaths
// (a leaf path with no slug — like /api/v1/whale-flow). NOT the same route as
// the existing /api/v1/macrocal (the blackout calendar) — distinct exact paths.
//
// WIRING: like funding (and unlike whale's pool-backed inline store), the macro
// store is a PROCESS SINGLETON — the stooq worker writes to it, this handler
// reads from it, and it has no DB. It's threaded in via SetMacroStore() from
// cmd/server/main.go, mirroring SetFundingStore. The handler reads the
// macro.SnapshotReader interface (not *Store) so macro_handlers_test.go can pass
// a fake and exercise the nil-store / populated / weekend paths offline.
//
// The calendar (feature #5) is EMBEDDED in this response (not a separate route):
// the frontend wants one fetch, and macrocal.Snapshot() is already in memory.

import (
	"fmt"
	"net/http"
	"time"

	"github.com/prostwp/elibri-backend/internal/macro"
	"github.com/prostwp/elibri-backend/internal/macrocal"
)

const (
	// tradfinStaleThreshold — if the freshest tradfin quote is older than this,
	// the market is treated as closed/stale (tradfin_market_open=false). stooq
	// live-updates during trading hours; a >15min-old freshest ts means we're on
	// a weekend or after the close. This IS the whole weekend mechanism: the
	// freshness of the timestamp honestly reflects whether the market is open.
	tradfinStaleThreshold = 15 * time.Minute

	// macroCalendarWindowHours / macroCalendarMax bound the embedded calendar:
	// top-5 medium+high-impact events in the next 72h.
	macroCalendarWindowHours = 72
	macroCalendarMax         = 5
)

// lampSpec defines a lamp's stooq symbol + human label, in render order.
type lampSpec struct {
	key    string
	symbol string
	label  string
}

// lampSpecs is the canonical 5-lamp set (order: dxy, rates, vix, spx, gold).
var lampSpecs = []lampSpec{
	{macro.KeyDXY, macro.SymDXY, "Dollar (DXY)"},
	{macro.KeyRates, macro.SymRates, "US 10Y"},
	{macro.KeyVIX, macro.SymVIX, "VIX"},
	{macro.KeySPX, macro.SymSPX, "S&P 500"},
	{macro.KeyGold, macro.SymGold, "Gold"},
}

// tradfinLampSymbols are the lamp symbols that close on weekends (BTC is 24/7
// and excluded from the market-open / freshest-ts computation).
var tradfinLampSymbols = []string{macro.SymDXY, macro.SymRates, macro.SymVIX, macro.SymSPX, macro.SymGold}

// correlationPairs maps the 3 correlation outputs to their (BTC, X) symbols.
var correlationPairs = []struct {
	pair string
	symX string
}{
	{macro.PairBTCSPX, macro.SymSPX},
	{macro.PairBTCGold, macro.SymGold},
	{macro.PairBTCDXY, macro.SymDXY},
}

// macroStore is the package-level snapshot reader wired in from
// cmd/server/main.go via SetMacroStore. nil → the handler serves an honest
// empty/degraded (but well-formed) response, so unit tests and a DB-less /
// worker-less boot don't have to plumb anything. Mirrors fundingStore.
var macroStore macro.SnapshotReader

// SetMacroStore wires the snapshot store the macro handler reads from. Called
// once from cmd/server/main.go with the same *macro.Store the stooq worker
// writes to. nil leaves the handler serving a degraded payload (regime "mixed",
// composite null, lamps/correlations with empty values) — the correct cold
// behaviour; the frontend renders its skeleton/empty states.
func SetMacroStore(r macro.SnapshotReader) { macroStore = r }

// handleMacro serves GET /api/v1/macro. Always 200; slices always non-nil ([]).
//
// Response shape (spec §5): regime/composite/tradfin_market_open/tradfin_as_of/
// captured_at + lamps[5] + correlations[3] + fng + generated_idea + calendar.
func handleMacro(w http.ResponseWriter, r *http.Request) {
	now := time.Now().UTC()

	// Calendar is independent of the macro store (it reads macrocal's cache), so
	// it's always present — empty [] when macrocal isn't initialised.
	cal := topCalendar()

	// nil store (not wired / DB-less boot) → degraded-but-valid payload.
	if macroStore == nil {
		writeJSON(w, macro.Response{
			Regime:            macro.RegimeMixed,
			Composite:         nil,
			TradfinMarketOpen: false,
			TradfinAsOf:       "",
			CapturedAt:        now.Format(time.RFC3339),
			Lamps:             emptyLamps(),
			Correlations:      emptyCorrelations(),
			Fng:               nil,
			GeneratedIdea:     "",
			Calendar:          cal,
		})
		return
	}

	quotes := macroStore.Latest()

	lamps := buildLamps(quotes)
	composite := macro.Composite(lamps)
	regime := macro.ClassifyRegime(composite)
	corrs := buildCorrelations(macroStore)
	open, asOf := tradfinOpenAsOf(quotes, now)
	idea := macro.BuildDiagnosis(regime, lamps)

	var fng *macro.FnG
	if f, ok := macroStore.FnG(); ok && f.OK {
		fc := f
		fng = &fc
	}

	writeJSON(w, macro.Response{
		Regime:            regime,
		Composite:         composite,
		TradfinMarketOpen: open,
		TradfinAsOf:       asOf,
		CapturedAt:        now.Format(time.RFC3339),
		Lamps:             lamps,
		Correlations:      corrs,
		Fng:               fng,
		GeneratedIdea:     idea,
		Calendar:          cal,
	})
}

// buildLamps turns the latest quotes into the 5 lamps (render order). An N/D /
// missing symbol → Value:nil, DeltaPct:nil, Status:"", AsOf:"" (frontend "—").
//
// Delta is the SESSION change (Close−Open) the quote already carries. delta==0
// (Open==Close) is a real "no move" and is reported; only a missing Open (0 →
// N/D / unparseable) leaves DeltaPct nil.
//
// Status (P1-4 — never fabricate a direction we don't have):
//   - VIX is level-based → always set from the value (delta ignored).
//   - directional lamps (dxy/rates/spx/gold) get a status ONLY when DeltaPct is
//     known; without a delta the lamp shows its value with Status:"" and is
//     excluded from Composite (which skips Status=="").
func buildLamps(quotes map[string]macro.Quote) []macro.Lamp {
	lamps := make([]macro.Lamp, 0, len(lampSpecs))
	for _, spec := range lampSpecs {
		lamp := macro.Lamp{Key: spec.key, Label: spec.label}
		q, ok := quotes[spec.symbol]
		if ok && q.OK {
			price := q.Price
			lamp.Value = &price
			if !q.AsOf.IsZero() {
				lamp.AsOf = q.AsOf.UTC().Format(time.RFC3339)
			}
			// Session delta from Open. Open>0 guards the N/D / unparseable case.
			if q.Open > 0 {
				d := pctChange(q.Open, q.Price)
				lamp.DeltaPct = &d
			}

			if spec.key == macro.KeyVIX {
				// Level-based: a status even without a delta.
				lamp.Status = macro.LampStatus(macro.KeyVIX, q.Price, 0)
			} else if lamp.DeltaPct != nil {
				// Directional: only with a known delta, else leave Status="".
				lamp.Status = macro.LampStatus(spec.key, q.Price, *lamp.DeltaPct)
			}
		}
		lamps = append(lamps, lamp)
	}
	return lamps
}

// buildCorrelations computes the 3 BTC↔X correlations + their labels + a human
// window string from the ring size. coef nil → label "" (frontend "Building
// correlation window").
func buildCorrelations(reader macro.SnapshotReader) []macro.Correlation {
	window := windowDescription(reader.PointCount())
	corrs := make([]macro.Correlation, 0, len(correlationPairs))
	for _, cp := range correlationPairs {
		coef := reader.Correlation(macro.SymBTC, cp.symX)
		corrs = append(corrs, macro.Correlation{
			Pair:   cp.pair,
			Coef:   coef,
			Label:  macro.CorrelationLabel(cp.pair, coef),
			Window: window,
		})
	}
	return corrs
}

// tradfinOpenAsOf computes (market_open, freshest_iso) from the freshest OK
// tradfin quote (BTC excluded — it's 24/7). "Open" is purely a STALENESS signal:
// the freshest ts being within tradfinStaleThreshold. No weekday gate — these
// are futures (vi.f/dx.f) that trade ~24×5, so a Mon..Fri test would falsely
// report "closed" at the Sunday-night futures reopen, and it's redundant with
// staleness on the weekend (the Friday close is long stale → open=false anyway).
// asOf is "" when every tradfin symbol is N/D.
func tradfinOpenAsOf(quotes map[string]macro.Quote, now time.Time) (bool, string) {
	var freshest time.Time
	for _, sym := range tradfinLampSymbols {
		q, ok := quotes[sym]
		if !ok || !q.OK || q.AsOf.IsZero() {
			continue
		}
		if q.AsOf.After(freshest) {
			freshest = q.AsOf
		}
	}
	if freshest.IsZero() {
		return false, ""
	}
	asOf := freshest.UTC().Format(time.RFC3339)
	open := now.Sub(freshest) < tradfinStaleThreshold
	return open, asOf
}

// topCalendar returns up to 5 medium+high-impact macro events in the next 72h,
// sorted by time asc, narrowed to the slim CalEvent shape. Uses an INLINE
// Config{MinImpact:"medium"} (NOT the package macroCfg, which is the blackout
// config with MinImpact:"high" — we want medium+high here). macrocal not
// initialised (no FINNHUB_API_KEY) → Snapshot() empty → UpcomingEvents [] →
// Calendar [] → frontend honest empty-state. Never fabricated.
func topCalendar() []macro.CalEvent {
	evts := macrocal.UpcomingEvents(macroCalendarWindowHours, macrocal.Config{MinImpact: "medium"})
	out := make([]macro.CalEvent, 0, macroCalendarMax)
	for _, e := range evts {
		if len(out) >= macroCalendarMax {
			break
		}
		out = append(out, macro.CalEvent{
			Country: e.Country,
			Event:   e.Event,
			Impact:  e.Impact,
			Time:    e.Time.UTC().Format(time.RFC3339),
		})
	}
	return out
}

// emptyLamps / emptyCorrelations build the degraded (nil-store) slices with the
// right keys/labels but no values.
func emptyLamps() []macro.Lamp {
	lamps := make([]macro.Lamp, 0, len(lampSpecs))
	for _, spec := range lampSpecs {
		lamps = append(lamps, macro.Lamp{Key: spec.key, Label: spec.label})
	}
	return lamps
}

func emptyCorrelations() []macro.Correlation {
	window := windowDescription(0)
	corrs := make([]macro.Correlation, 0, len(correlationPairs))
	for _, cp := range correlationPairs {
		corrs = append(corrs, macro.Correlation{Pair: cp.pair, Window: window})
	}
	return corrs
}

// pctChange returns the percent change from prev to cur (e.g. 100→101 = +1.0).
// prev==0 guarded by the caller.
func pctChange(prev, cur float64) float64 {
	return (cur - prev) / prev * 100
}

// windowDescription renders the rolling-window size as a human string, e.g.
// "≈3h (52 points)". One point per ~3-min cycle → minutes ≈ points×3.
func windowDescription(points int) string {
	if points <= 0 {
		return "building"
	}
	plural := "points"
	if points == 1 {
		plural = "point"
	}
	// At ring capacity the window stops growing — say so honestly.
	suffix := ""
	if points >= macro.RingCap {
		suffix = " (rolling)"
	}
	mins := points * 3
	if mins < 60 {
		return fmt.Sprintf("≈%dm (%d %s)%s", mins, points, plural, suffix)
	}
	hours := (mins + 30) / 60 // rounded
	return fmt.Sprintf("≈%dh (%d %s)%s", hours, points, plural, suffix)
}
