package demobot

// gold_test.go — the gold agent's own logic. Everything here is pure and
// offline: no network, no clock beyond what the case passes in.

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// bar is a terse constructor for the table cases: day N with the given range.
func bar(day int, low, high float64) types.OHLCVCandle {
	return types.OHLCVCandle{
		// 04:00 UTC stamps, the real GC=F daily grid (midnight New York).
		Time: time.Date(2026, 1, day, 4, 0, 0, 0, time.UTC).Unix(),
		Open: low, High: high, Low: low, Close: high,
	}
}

func TestGoldDayLevelsUnwindsInsideDays(t *testing.T) {
	tests := []struct {
		name        string
		bars        []types.OHLCVCandle
		wantDefined bool
		wantHigh    float64
		wantLow     float64
		wantUnwound int
	}{
		{
			name: "plain day — the last closed bar bounds the day",
			bars: []types.OHLCVCandle{
				bar(1, 100, 110),
				bar(2, 105, 120),
			},
			wantDefined: true, wantHigh: 120, wantLow: 105, wantUnwound: 0,
		},
		{
			name: "one inside day — the enclosing day bounds it",
			bars: []types.OHLCVCandle{
				bar(1, 100, 130),
				bar(2, 110, 120), // inside day 1
			},
			wantDefined: true, wantHigh: 130, wantLow: 100, wantUnwound: 1,
		},
		{
			name: "two nested inside days — unwind twice",
			bars: []types.OHLCVCandle{
				bar(1, 100, 140),
				bar(2, 110, 130), // inside day 1
				bar(3, 115, 125), // inside day 2
			},
			wantDefined: true, wantHigh: 140, wantLow: 100, wantUnwound: 2,
		},
		{
			name: "three nested — still resolves, cap is three steps",
			bars: []types.OHLCVCandle{
				bar(1, 100, 150),
				bar(2, 110, 140),
				bar(3, 115, 135),
				bar(4, 120, 130),
			},
			wantDefined: true, wantHigh: 150, wantLow: 100, wantUnwound: 3,
		},
		{
			name: "four nested — deeper than the cap, no levels rather than wrong ones",
			bars: []types.OHLCVCandle{
				bar(1, 100, 160),
				bar(2, 105, 155),
				bar(3, 110, 150),
				bar(4, 115, 145),
				bar(5, 120, 140),
			},
			wantDefined: false,
		},
		{
			name: "equal high and low counts as inside — an exact repeat bounds nothing new",
			bars: []types.OHLCVCandle{
				bar(1, 100, 120),
				bar(2, 100, 120), // identical range
			},
			wantDefined: true, wantHigh: 120, wantLow: 100, wantUnwound: 1,
		},
		{
			name: "outside day — wider than yesterday, no unwind",
			bars: []types.OHLCVCandle{
				bar(1, 110, 120),
				bar(2, 100, 130),
			},
			wantDefined: true, wantHigh: 130, wantLow: 100, wantUnwound: 0,
		},
		{
			name:        "single bar — nothing to compare against, still usable",
			bars:        []types.OHLCVCandle{bar(1, 100, 110)},
			wantDefined: true, wantHigh: 110, wantLow: 100, wantUnwound: 0,
		},
		{
			name:        "no bars — undefined, never a zero-valued level",
			bars:        nil,
			wantDefined: false,
		},
		{
			name: "inside chain reaches the start of the series — that first bar bounds it",
			bars: []types.OHLCVCandle{
				bar(1, 110, 130),
				bar(2, 115, 125), // inside, but there is nothing before day 1 to fall back to
			},
			wantDefined: true, wantHigh: 130, wantLow: 110, wantUnwound: 1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := goldDayLevelsOf(tc.bars)
			if got.Defined != tc.wantDefined {
				t.Fatalf("Defined = %v, want %v", got.Defined, tc.wantDefined)
			}
			if !tc.wantDefined {
				return
			}
			if got.High != tc.wantHigh || got.Low != tc.wantLow {
				t.Errorf("range = %.0f-%.0f, want %.0f-%.0f",
					got.Low, got.High, tc.wantLow, tc.wantHigh)
			}
			if got.Unwound != tc.wantUnwound {
				t.Errorf("Unwound = %d, want %d", got.Unwound, tc.wantUnwound)
			}
		})
	}
}

func TestGoldDayLevelsPositionWords(t *testing.T) {
	dr := goldDayLevels{High: 120, Low: 100, Defined: true}
	tests := []struct {
		price float64
		want  string
	}{
		{130, dayAbove},
		{120, dayInside}, // exactly at the edge is not yet beyond it
		{110, dayInside},
		{100, dayInside},
		{90, dayBelow},
	}
	for _, tc := range tests {
		if got := dr.positionOf(tc.price); got != tc.want {
			t.Errorf("positionOf(%.0f) = %q, want %q", tc.price, got, tc.want)
		}
	}
}

func TestGoldDayLevelsPositionUndefinedHasNoWord(t *testing.T) {
	dr := goldDayLevels{Defined: false}
	if got := dr.positionOf(105); got != "" {
		t.Errorf("an undefined range must not place price, got %q", got)
	}
}

// ── the card ─────────────────────────────────────────────────────────────────

// stubGoldYahoo serves both intervals the gold agent asks for from one server:
// `daily` closed daily bars built by shape(i), and a short hourly series whose
// last closed bar carries lastHour as its close.
func stubGoldYahoo(t *testing.T, daily int, shape func(i int) (o, h, l, c float64), lastHour float64) {
	t.Helper()
	// End two days back so every daily bar is unambiguously closed.
	dayEnd := time.Now().UTC().Truncate(24 * time.Hour).Add(-48 * time.Hour)

	chart := func(ts []int64, o, h, l, c []float64) []byte {
		payload := map[string]any{
			"chart": map[string]any{
				"result": []any{map[string]any{
					"timestamp": ts,
					"indicators": map[string]any{
						"quote": []any{map[string]any{
							"open": o, "high": h, "low": l, "close": c,
							"volume": make([]any, len(ts)),
						}},
					},
				}},
			},
		}
		b, err := json.Marshal(payload)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}

	dts := make([]int64, daily)
	do, dh, dl, dc := make([]float64, daily), make([]float64, daily), make([]float64, daily), make([]float64, daily)
	for i := 0; i < daily; i++ {
		dts[i] = dayEnd.Unix() - int64(daily-1-i)*86400
		do[i], dh[i], dl[i], dc[i] = shape(i)
	}
	dailyBody := chart(dts, do, dh, dl, dc)

	const hours = 8
	hEnd := time.Now().UTC().Truncate(time.Hour).Add(-2 * time.Hour)
	hts := make([]int64, hours)
	ho, hh, hl, hc := make([]float64, hours), make([]float64, hours), make([]float64, hours), make([]float64, hours)
	for i := 0; i < hours; i++ {
		hts[i] = hEnd.Unix() - int64(hours-1-i)*3600
		ho[i], hh[i], hl[i], hc[i] = lastHour, lastHour+1, lastHour-1, lastHour
	}
	hourBody := chart(hts, ho, hh, hl, hc)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Query().Get("interval") == "1d" {
			_, _ = w.Write(dailyBody)
			return
		}
		_, _ = w.Write(hourBody)
	}))
	t.Cleanup(srv.Close)
	orig := yahooChartBase
	yahooChartBase = srv.URL + "/v8/finance/chart/"
	t.Cleanup(func() { yahooChartBase = orig })
}

// risingDay is a clean uptrend: EMA/ADX confirm, monotone so the swing
// structure is unreadable and (correctly) does not veto.
func risingDay(i int) (o, h, l, c float64) {
	p := 2000 + 12*float64(i)
	return p - 2, p + 5, p - 5, p
}

// flatDay is a dead-flat tape: ADX collapses, no regime.
func flatDay(i int) (o, h, l, c float64) {
	p := 2000.0
	if i%2 == 0 {
		p += 0.5
	}
	return p, p + 1, p - 1, p
}

func TestGoldCardConfirmedRegimeDescribesTheRegime(t *testing.T) {
	// Price INSIDE the day range (last closed daily bar is 5103–5113), so the
	// trigger reads in its plain conditional form.
	stubGoldYahoo(t, 260, risingDay, 5108)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.GoldCard(context.Background())

	if c.Offline {
		t.Fatalf("card went offline: %q", c.Verdict)
	}
	if !strings.Contains(c.Verdict, "Daily regime: confirmed UPTREND") {
		t.Errorf("verdict = %q, want the regime described on a confirmed uptrend", c.Verdict)
	}
	// The Этап 5 run measured the old forecast wording and did not find the
	// edge it promised. Forecast words must not come back by accident.
	for _, banned := range []string{"bias", "Bias", "will ", "expect"} {
		if strings.Contains(c.Verdict, banned) {
			t.Errorf("verdict %q claims a forecast via %q — the run found none", c.Verdict, banned)
		}
	}
	joined := strings.Join(c.Facts, "|")
	for _, want := range []string{"Day range", "Last closed 1h price", "classifies the day as an upside break"} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing fact %q in %v", want, c.Facts)
		}
	}
	// A confirmed regime is fully stated by the verdict — the old "Regime:"
	// fact would only repeat it.
	if strings.Contains(joined, "Regime:") {
		t.Errorf("confirmed regime must not repeat itself as a fact: %v", c.Facts)
	}
	// The macro backend is unreachable in this test — the card must SAY so,
	// not quietly drop the line.
	if !strings.Contains(joined, "Macro backdrop: no gold read available") {
		t.Errorf("an absent macro read must be stated: %v", c.Facts)
	}
}

// A scenario that ignores where price already is contradicts the line above
// it. Found on the first live render: price above the range followed by a
// plain "close above" condition reads as though nothing happened. The level
// still needs a daily CLOSE, so the card must say both halves.
func TestGoldCardTriggerAcknowledgesPriceAlreadyBeyondTheRange(t *testing.T) {
	stubGoldYahoo(t, 260, risingDay, 5200) // well above the 5103–5113 range
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.GoldCard(context.Background())

	joined := strings.Join(c.Facts, "|")
	if !strings.Contains(joined, "the last 1h close is already above it") {
		t.Errorf("scenario must acknowledge price is already beyond the level: %v", c.Facts)
	}
	if !strings.Contains(joined, "A daily close above") {
		t.Errorf("scenario must still require a daily close, not an intraday touch: %v", c.Facts)
	}
	if strings.Contains(joined, "already below") {
		t.Errorf("the other side must stay plain: %v", c.Facts)
	}
}

func TestGoldCardUnconfirmedRegimeGivesNoDirection(t *testing.T) {
	stubGoldYahoo(t, 260, flatDay, 2000)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.GoldCard(context.Background())

	if !strings.Contains(c.Verdict, "Daily regime: not confirmed") {
		t.Errorf("verdict = %q, want no direction on an unconfirmed regime", c.Verdict)
	}
	// Unconfirmed states DO keep the regime fact: it carries the reason
	// ("flat, trading not advised (ADX …)"), which the verdict does not.
	if !strings.Contains(strings.Join(c.Facts, "|"), "Regime:") {
		t.Errorf("an unconfirmed regime must state WHY: %v", c.Facts)
	}
	if c.Emoji != emojiNeutral {
		t.Errorf("emoji = %q, want neutral when no direction is claimed", c.Emoji)
	}
	// Silence is about DIRECTION only — the levels still ship.
	if !strings.Contains(strings.Join(c.Facts, "|"), "classifies the day as an upside break") {
		t.Errorf("levels must survive an unconfirmed regime: %v", c.Facts)
	}
}

// The instrument is named on every card: spot XAUUSD does not exist on this
// feed, and a levels agent that silently prints futures prices under a spot
// ticker is off by the basis on every number it shows.
func TestGoldCardNamesTheInstrument(t *testing.T) {
	stubGoldYahoo(t, 260, risingDay, 5200)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.GoldCard(context.Background())

	if !strings.Contains(c.Asset, "COMEX") {
		t.Errorf("asset label = %q, want the contract named", c.Asset)
	}
	if !strings.Contains(c.SourceNote, "not spot XAUUSD") {
		t.Errorf("source note must disown spot: %q", c.SourceNote)
	}
	if !strings.Contains(c.RenderHTML(), "COMEX") {
		t.Error("the rendered card must carry the instrument")
	}
}

func TestGoldCardOfflineWhenSourceIsDown(t *testing.T) {
	orig := yahooChartBase
	yahooChartBase = "http://127.0.0.1:1/v8/finance/chart/"
	t.Cleanup(func() { yahooChartBase = orig })

	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.GoldCard(context.Background())

	if !c.Offline || c.Status != statusSourceOffline {
		t.Errorf("dead source must degrade honestly: offline=%v status=%v", c.Offline, c.Status)
	}
	if len(c.Facts) != 0 {
		t.Errorf("an offline card must carry no facts, got %v", c.Facts)
	}
}

// ── Этап 6 blockers: regression guards ───────────────────────────────────────

// stubGoldDailyOnly serves daily bars and FAILS every intraday request.
func stubGoldDailyOnly(t *testing.T, daily int, shape func(i int) (o, h, l, c float64)) {
	t.Helper()
	dayEnd := time.Now().UTC().Truncate(24 * time.Hour).Add(-48 * time.Hour)
	ts := make([]int64, daily)
	o, h, l, c := make([]float64, daily), make([]float64, daily), make([]float64, daily), make([]float64, daily)
	for i := 0; i < daily; i++ {
		ts[i] = dayEnd.Unix() - int64(daily-1-i)*86400
		o[i], h[i], l[i], c[i] = shape(i)
	}
	body, err := json.Marshal(map[string]any{"chart": map[string]any{"result": []any{map[string]any{
		"timestamp": ts,
		"indicators": map[string]any{"quote": []any{map[string]any{
			"open": o, "high": h, "low": l, "close": c, "volume": make([]any, daily),
		}}},
	}}}})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("interval") == "1d" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(body)
			return
		}
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)
	orig := yahooChartBase
	yahooChartBase = srv.URL + "/v8/finance/chart/"
	t.Cleanup(func() { yahooChartBase = orig })
}

// Б1: a dead intraday feed used to leave a green "confirmed UPTREND" card
// serving ok=true, with nothing anywhere saying a source had failed — while
// the agent's own conflict-priority comment said "no price → silent".
func TestGoldCardNoPriceClaimsNoDirection(t *testing.T) {
	stubGoldDailyOnly(t, 260, risingDay) // daily alone would confirm an uptrend
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.GoldCard(context.Background())

	if c.Emoji != emojiNeutral {
		t.Errorf("emoji = %q, want neutral — no direction may be coloured without a price", c.Emoji)
	}
	if strings.Contains(c.Verdict, "confirmed UPTREND") {
		t.Errorf("verdict = %q, must not state a direction without a live price", c.Verdict)
	}
	if c.effectiveStatus() == statusOK {
		t.Error("a partial source failure must not serve ok=true")
	}
	joined := strings.Join(c.Facts, "|")
	if !strings.Contains(joined, "Intraday price feed is down") {
		t.Errorf("the failure must be stated, not silently dropped: %v", c.Facts)
	}
	// The regime is known and may be reported — but never as a bare claim that
	// argues with the header.
	if strings.Contains(joined, "Regime: confirmed uptrend") {
		t.Errorf("a bare regime claim under a no-direction header reads as self-contradiction: %v", c.Facts)
	}
	// The levels still ship: they come from the daily series, which is alive.
	if !strings.Contains(joined, "classifies the day as an upside break") {
		t.Errorf("day levels must survive a dead intraday feed: %v", c.Facts)
	}
}

// Б3: display precision must be sufficient wherever the card asserts an
// inequality. trimFloat rounds ≥1000 to a whole number, which made the card
// print "4615 — above the range" against a level also shown as 4615.
func TestGoldComparedPricesArePrintedPreciselyEnough(t *testing.T) {
	d := goldDayLevels{High: 4614.6, Low: 4600.2, Defined: true}
	const px = 4614.8

	if got := d.positionOf(px); got != dayAbove {
		t.Fatalf("precondition: positionOf(%v) = %q, want above", px, got)
	}
	line := strings.Join(d.scenarios(px, true), " | ")
	if strings.Contains(line, "4615") {
		t.Errorf("trigger rounds the level into the price it is compared against: %q", line)
	}
	if !strings.Contains(line, "4614.60") {
		t.Errorf("trigger must print the level at tick resolution: %q", line)
	}
	if goldPx(px) == goldPx(d.High) {
		t.Errorf("a price the code calls ABOVE a level must not print equal to it: %s vs %s",
			goldPx(px), goldPx(d.High))
	}
}

// Б3, second instance — found by reading the live card, not by a test: an ADX
// of 19.6 printed at zero decimals rendered "Flat — no trend to read
// (ADX 20 < 20)". The same rule applies wherever text asserts a comparison.
func TestFlatVerdictNeverPrintsAContradictoryInequality(t *testing.T) {
	for _, adx := range []float64{19.5, 19.6, 19.94, 19.99, 19.999, 18.0, 0.04} {
		v := trendVerdict(trendRead{State: trendFlat, Raw: trendFlat, ADX: adx}, "")
		if strings.Contains(v, "20 < 20") || strings.Contains(v, "20.0 < 20") {
			t.Errorf("adx %.2f → %q: the printed numbers contradict the claim", adx, v)
		}
	}
}

// Б4: the honest human label must not leak into the machine contract.
func TestGoldCardKeepsDocumentedAssetKey(t *testing.T) {
	stubGoldYahoo(t, 260, risingDay, 5108)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	c := ag.GoldCard(context.Background())

	if c.Asset != "GOLD · COMEX GC=F" {
		t.Errorf("human label = %q, want the contract named", c.Asset)
	}
	if got := cardEnvelope(c).Asset; got != "XAUUSD" {
		t.Errorf("envelope asset = %q, want XAUUSD — integrations branch on it", got)
	}
}

// Б1/Д2/Д3: every degraded gold card keeps the instrument disclosure and must
// not describe a COMEX failure as an FX one.
func TestGoldDegradedCardsKeepInstrumentDisclosure(t *testing.T) {
	t.Run("source dead", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "boom", http.StatusInternalServerError)
		}))
		defer srv.Close()
		orig := yahooChartBase
		yahooChartBase = srv.URL + "/v8/finance/chart/"
		defer func() { yahooChartBase = orig }()

		c := NewAgents(NewBackendClient("http://127.0.0.1:1")).GoldCard(context.Background())
		if !strings.Contains(c.SourceNote, "not spot XAUUSD") {
			t.Errorf("disclosure lost on the offline card: %q", c.SourceNote)
		}
		if strings.Contains(c.Verdict, "FX data source") {
			t.Errorf("a COMEX futures failure must not be called an FX one: %q", c.Verdict)
		}
	})
	t.Run("short history", func(t *testing.T) {
		stubGoldYahoo(t, 40, risingDay, 2400)
		c := NewAgents(NewBackendClient("http://127.0.0.1:1")).GoldCard(context.Background())
		if !strings.Contains(c.SourceNote, "not spot XAUUSD") {
			t.Errorf("disclosure lost on the insufficient-history card: %q", c.SourceNote)
		}
	})
}

// Б2: analytic statements only — a recommendation to abstain is still a
// recommendation, and the footer disclaimer does not change the body.
func TestNoAdviceLanguageInVerdicts(t *testing.T) {
	banned := []string{"advised", "stand aside", "no trade", "should", "avoid", "recommend"}
	for _, st := range []string{trendFlat, trendGrey, trendUp, trendDown, trendConflict} {
		v := trendVerdict(trendRead{State: st, Raw: st, ADX: 18, EMA50: 1.1, EMA200: 1.2, Last: 1.15}, "1h")
		for _, b := range banned {
			if strings.Contains(strings.ToLower(v), b) {
				t.Errorf("trendVerdict(%q) = %q contains advice word %q", st, v, b)
			}
		}
	}
	for k, how := range howTexts {
		for _, b := range banned {
			if strings.Contains(strings.ToLower(how), b) {
				t.Errorf("howTexts[%q] contains advice word %q: %q", k, b, how)
			}
		}
	}
}
