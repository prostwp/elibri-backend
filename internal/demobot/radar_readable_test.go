package demobot

// radar_readable_test.go — Risk Radar: a golden card off the exact file the
// external Python job writes, honest degrades (stale forecast, no file,
// broken JSON, Volatility unavailable), the 110-rune line budget, the banned
// forecast words, and the HTTP wiring (routing, ?asset=, /agents, the push
// hook staying silent about it — all without touching the network).

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// radarAt is the same UTC day the golden fixture's forecast is for.
var radarAt = time.Date(2026, 9, 29, 12, 0, 0, 0, time.UTC)

const radarFixture = "testdata/radar_2026-09-29.json"

// radarTestAgents wires an Agents whose network is dead (stubExternalBases —
// which also points DEMOBOT_RADAR_FILE at a guaranteed-missing path) and
// then, if path != "", overrides DEMOBOT_RADAR_FILE with it.
func radarTestAgents(t *testing.T, path string) *Agents {
	t.Helper()
	stubExternalBases(t)
	if path != "" {
		t.Setenv("DEMOBOT_RADAR_FILE", path)
	}
	return NewAgents(NewBackendClient("http://127.0.0.1:1"))
}

// flatRangeCandles is n BTC 4h bars at a constant price and a constant
// high-low range: true range is identical on every bar (no gaps, since the
// close never moves), so ATR(14) is exactly flat and the ratio against its
// own 30-bar baseline is exactly 1.0 — deterministically "normal", with no
// dependence on any wave shape.
func flatRangeCandles(n int, lastClose time.Time) []types.OHLCVCandle {
	cs := make([]types.OHLCVCandle, n)
	for i := range cs {
		cs[i] = types.OHLCVCandle{
			Time: lastClose.Add(-time.Duration(n-i) * 4 * time.Hour).Unix(),
			Open: 60000, High: 60050, Low: 59950, Close: 60000, Volume: 100,
		}
	}
	return cs
}

// seedNormalBTCVol installs a deterministic "normal" Volatility read into
// ag's shared kline cache — the exact cache a.VolCard(ctx, btcSpec) (and so
// RadarCard) reads, no Binance stub involved.
func seedNormalBTCVol(ag *Agents) {
	now := time.Now().UTC().Truncate(time.Second)
	lastClose := now.Add(-time.Hour)
	series := flatRangeCandles(binanceFetchLimit, lastClose)
	seedBTC4h(ag, withForming(series[1:], lastClose), now)
}

// ── golden: the live wiring end to end ───────────────────────────────────────

func TestRadarCardGoldenNormalVolatility(t *testing.T) {
	ag := radarTestAgents(t, radarFixture)
	ag.now = func() time.Time { return radarAt }
	seedNormalBTCVol(ag)

	c := ag.RadarCard(context.Background())
	if c.Status != statusOK {
		t.Fatalf("status %v, want ok (facts: %v)", c.Status, c.Facts)
	}
	if c.Agent != "Risk Radar" || c.ShortName != "Risk Radar" || c.Asset != "BTC" {
		t.Errorf("identity: agent %q short %q asset %q", c.Agent, c.ShortName, c.Asset)
	}
	if c.Emoji != emojiNeutral {
		t.Errorf("emoji %q, want neutral", c.Emoji)
	}

	wantVerdict := "Next 24h · BTC: expected move ±2.1% — more active than usual"
	if c.Verdict != wantVerdict {
		t.Errorf("verdict:\n got  %q\n want %q", c.Verdict, wantVerdict)
	}

	wantFacts := []string{
		"Model forecast for 2026-09-29 UTC: 2.06% daily volatility (1σ of the day's move); yesterday 1.83%",
		"Higher than 69% of the last 180 days · issued 00:00 UTC from closed hourly candles",
		"Volatility agent: last 4h candles vs prior 30 (now: normal); this forecast covers the whole UTC day",
		"Tested on 154 unseen days: 29% less error than HAR-RV, 37% less than ATR(14)",
		"Against HAR-RV + weekday: 7.5% less error, not yet statistically confirmed (p=0.053)",
		"Size of the move only: no direction and no guaranteed range",
	}
	if strings.Join(c.Facts, "\n") != strings.Join(wantFacts, "\n") {
		t.Errorf("facts:\n got:\n%s\nwant:\n%s", strings.Join(c.Facts, "\n"), strings.Join(wantFacts, "\n"))
	}
	// forward.days is 0 in the fixture — no 7th fact.
	if len(c.Facts) != 6 {
		t.Errorf("fact count %d, want 6 (forward.days=0 must print nothing)", len(c.Facts))
	}
	if c.Blocks == nil || c.Blocks.Limitations != radarLimitations {
		t.Errorf("blocks limitations: %+v", c.Blocks)
	}
}

// ── Volatility unavailable ───────────────────────────────────────────────────

func TestRadarVolatilityUnavailable(t *testing.T) {
	ag := radarTestAgents(t, radarFixture) // network stays dead: no vol stub
	ag.now = func() time.Time { return radarAt }

	c := ag.RadarCard(context.Background())
	if c.Status != statusOK {
		t.Fatalf("status %v, want ok even without Volatility", c.Status)
	}
	want := "Volatility agent reading unavailable; this forecast covers the whole UTC day"
	if len(c.Facts) < 3 || c.Facts[2] != want {
		t.Errorf("fact 3: %q, want %q", safeFact(c.Facts, 2), want)
	}
}

func safeFact(facts []string, i int) string {
	if i < len(facts) {
		return facts[i]
	}
	return "<missing>"
}

// ── stale forecast: honest, no numbers ───────────────────────────────────────

func TestRadarStaleForecastCard(t *testing.T) {
	ag := radarTestAgents(t, radarFixture)
	ag.now = func() time.Time { return radarAt.AddDate(0, 0, 1) } // 2026-09-30

	c := ag.RadarCard(context.Background())
	if c.Status != statusNoData {
		t.Fatalf("status %v, want no_data", c.Status)
	}
	if env := cardEnvelope(c); env.OK {
		t.Error("ok must be false on a stale forecast")
	} else if env.Reason == nil || *env.Reason != "no_data" {
		t.Errorf("reason %v, want no_data", env.Reason)
	}
	want := "No model forecast for today yet (last: 2026-09-29)"
	if c.Verdict != want {
		t.Errorf("verdict: %q, want %q", c.Verdict, want)
	}
	for _, s := range append([]string{c.Verdict}, c.Facts...) {
		if strings.ContainsAny(s, "0123456789") && !strings.Contains(s, "2026-09-29") && !strings.Contains(s, "00:00") {
			t.Errorf("stale card must carry no forecast numbers: %q", s)
		}
	}
	if c.Blocks != nil {
		t.Errorf("a degraded card carries no blocks, got %+v", c.Blocks)
	}
}

// ── no file / broken JSON: honest, no panic ──────────────────────────────────

func TestRadarNoFileDegradesHonestly(t *testing.T) {
	ag := radarTestAgents(t, "") // stubExternalBases already points at a missing path
	c := ag.RadarCard(context.Background())
	if !c.Offline || c.effectiveStatus() != statusSourceOffline {
		t.Fatalf("no file: offline=%v status=%v, want an honest offline card", c.Offline, c.effectiveStatus())
	}
}

func TestRadarBrokenJSONDegradesHonestly(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "radar.json")
	if err := os.WriteFile(path, []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	ag := radarTestAgents(t, path)
	c := ag.RadarCard(context.Background()) // must not panic
	if !c.Offline || c.effectiveStatus() != statusSourceOffline {
		t.Fatalf("broken json: offline=%v status=%v, want an honest offline card", c.Offline, c.effectiveStatus())
	}
}

// A file that vanishes between two GETs (mtime cache) must not wedge the
// agent on the old content or panic on the stat failure.
func TestRadarFileRemovedBetweenReads(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "radar.json")
	raw, err := os.ReadFile(radarFixture)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	ag := radarTestAgents(t, path)
	ag.now = func() time.Time { return radarAt }
	if c := ag.RadarCard(context.Background()); c.Status != statusOK {
		t.Fatalf("first read: status %v", c.Status)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	c := ag.RadarCard(context.Background())
	if !c.Offline {
		t.Errorf("removed file must degrade honestly, got %+v", c)
	}
}

// ── pure builder: pre_registered_pass and the forward-test line ─────────────

func radarFixtureFile(t *testing.T) radarFile {
	t.Helper()
	raw, err := os.ReadFile(radarFixture)
	if err != nil {
		t.Fatal(err)
	}
	var f radarFile
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestRadarFact5ConfirmedWhenPreRegisteredPass(t *testing.T) {
	f := radarFixtureFile(t)
	if got := radarFact5(f); got != "Against HAR-RV + weekday: 7.5% less error, not yet statistically confirmed (p=0.053)" {
		t.Errorf("fact5 (fail): %q", got)
	}
	f.Validation.PreRegisteredPass = true
	want := "Against HAR-RV + weekday: 7.5% less error, confirmed (p=0.053)"
	if got := radarFact5(f); got != want {
		t.Errorf("fact5 (pass): got %q, want %q", got, want)
	}
}

func TestRadarFact7OnlyWhenForwardHasDays(t *testing.T) {
	f := radarFixtureFile(t)
	if _, ok := radarFact7(f); ok {
		t.Error("forward.days=0 must print nothing")
	}
	gain := 0.084
	f.Forward = radarForward{Days: 12, GainVsHARDow: &gain}
	got, ok := radarFact7(f)
	if !ok {
		t.Fatal("forward.days>0 must print a fact")
	}
	if want := "Live forward test: 12 days, 8.4% vs HAR-RV + weekday"; got != want {
		t.Errorf("fact7: got %q, want %q", got, want)
	}
	full := radarFacts(f, "x")
	if len(full) != 7 || full[6] != got {
		t.Errorf("radarFacts must append fact7 last, got %v", full)
	}
}

// ── readability budget + banned words ────────────────────────────────────────

func radarAllLines(c Card) []string {
	out := append([]string{c.Verdict}, c.Facts...)
	if c.Blocks != nil {
		out = append(out, c.Blocks.WhatHappened, c.Blocks.WhyLevel, c.Blocks.Limitations)
		out = append(out, c.Blocks.Scenarios...)
	}
	return out
}

func TestRadarLinesFitBudget(t *testing.T) {
	ag := radarTestAgents(t, radarFixture)
	ag.now = func() time.Time { return radarAt }
	seedNormalBTCVol(ag)
	c := ag.RadarCard(context.Background())

	// Stress case: large numbers at every field, both pass states, a live
	// forward test — the widest realistic card.
	stress := radarFixtureFile(t)
	stress.VolPct, stress.YesterdayPct, stress.Percentile180d = 99.99, 99.99, 0.999
	stress.Validation.Days = 99999
	stress.Validation.GainVsHAR, stress.Validation.GainVsATR14 = 0.999, 0.999
	stress.Validation.GainVsHARDow, stress.Validation.PVsHARDow = 0.999, 0.00001
	gain := 0.999
	stress.Forward = radarForward{Days: 9999, GainVsHARDow: &gain}
	stressCard := radarCardFrom(stress, "Volatility agent: last 4h candles vs prior 30 (now: elevated); this forecast covers the whole UTC day", radarAt)

	for _, c := range []Card{c, stressCard} {
		for _, s := range radarAllLines(c) {
			if n := utf8.RuneCountInString(s); n > radarFactMaxRunes {
				t.Errorf("%q: %d runes > %d", s, n, radarFactMaxRunes)
			}
		}
	}
}

func TestRadarNoForecastWords(t *testing.T) {
	ag := radarTestAgents(t, radarFixture)
	ag.now = func() time.Time { return radarAt }
	seedNormalBTCVol(ag)
	c := ag.RadarCard(context.Background())

	banned := []string{"will ", "likely", "expect price", "target"}
	for _, s := range radarAllLines(c) {
		for _, b := range banned {
			if strings.Contains(strings.ToLower(s), b) {
				t.Errorf("%q contains banned forecast word %q", s, b)
			}
		}
	}
	// "expected move" is the one allowed forecast phrase, and it must appear.
	if !strings.Contains(c.Verdict, "expected move") {
		t.Errorf("verdict must use the one allowed phrase: %q", c.Verdict)
	}
}

// ── HTTP wiring ───────────────────────────────────────────────────────────────

// freshRadarFixture copies the golden fixture with forecast_day/generated_at
// rewritten to "today" (real wall clock — the HTTP server under test uses no
// injected clock), so an HTTP-level 200 does not depend on the fixture's
// date staying in the future forever.
func freshRadarFixture(t *testing.T) string {
	t.Helper()
	f := radarFixtureFile(t)
	now := time.Now().UTC()
	f.ForecastDay = now.Format("2006-01-02")
	f.GeneratedAt = now.Format(time.RFC3339)
	raw, err := json.Marshal(f)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "radar.json")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestHTTPRadarRouting(t *testing.T) {
	ag := radarTestAgents(t, freshRadarFixture(t))
	_, srv := newTestAPI(t, ag, true)

	for _, path := range []string{"/agents/radar", "/agents/radar?asset=btc"} {
		status, _, body := httpGet(t, srv.URL+path)
		if status != 200 {
			t.Errorf("%s: status %d (%s)", path, status, body)
		}
	}

	status, _, body := httpGet(t, srv.URL+"/agents/radar?asset=eth")
	if status != 400 {
		t.Errorf("asset=eth: status %d, want 400 (%s)", status, body)
	}
	var e struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(body, &e); err != nil || !strings.Contains(e.Error, "unknown asset") {
		t.Errorf("asset=eth error %q must name the unknown asset", e.Error)
	}
}

func TestAgentsListIncludesRadar(t *testing.T) {
	_, srv := newTestAPI(t, deadAgents(t), true)
	status, _, body := httpGet(t, srv.URL+"/agents")
	if status != 200 {
		t.Fatalf("/agents: status %d (%s)", status, body)
	}
	var list struct {
		Agents []struct {
			Name string `json:"name"`
		} `json:"agents"`
	}
	if err := json.Unmarshal(body, &list); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, a := range list.Agents {
		if a.Name == keyRadar {
			found = true
		}
	}
	if !found {
		t.Errorf("/agents must list %q, got %+v", keyRadar, list.Agents)
	}
}

func TestHookTargetsExcludeRadar(t *testing.T) {
	for _, tg := range hookTargets() {
		if tg.Agent == keyRadar {
			t.Errorf("hookTargets must not sweep radar yet, got %+v", tg)
		}
	}
}
