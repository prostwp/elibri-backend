package demobot

// macro_arith_test.go — Codex review of ac80a18, two findings:
//   1. printed arithmetic: "=" only when 50 + the PRINTED contributions adds up
//      exactly to the PRINTED score (the unrounded check printed
//      "100 = 50 + 16.7 + 16.7 + 16.7");
//   2. no false 304 on a changed card, data_as_of stays the oldest live lamp.
//      Since 2026-09-15 macro carries no validator at all: the backend's
//      captured_at is its request time at one-second resolution, so two
//      different bodies can share any stamp we could send.

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/internal/macro"
)

var (
	macroBreakdownRe = regexp.MustCompile(`^(?:Rule|Gold) score (\d+) ([=≈]) 50(.*)$`)
	macroScoreLineRe = regexp.MustCompile(`^(?:Rule|Gold) score (\d+)/100 ([=≈]) 50 \+ the contributions above`)
	macroLampTailRe  = regexp.MustCompile(` → .*, ([+-]\d+\.\d|0)$`)
)

func tenthsOf(t *testing.T, s string) int {
	t.Helper()
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		t.Fatalf("unparseable contribution %q", s)
	}
	return int(math.Round(f * 10))
}

// checkSum asserts the sign against the printed numbers: "=" adds up exactly,
// "≈" differs, by less than 1.
func checkSum(t *testing.T, label, line string, score int, sign string, sumTenths int) {
	t.Helper()
	diff := 500 + sumTenths - score*10
	switch {
	case sign == "=" && diff != 0:
		t.Errorf("%s: %q — printed terms add up to %.1f, not %d", label, line, float64(500+sumTenths)/10, score)
	case sign == "≈" && diff == 0:
		t.Errorf("%s: %q — printed terms add up exactly, must be \"=\"", label, line)
	case sign == "≈" && (diff >= 10 || diff <= -10):
		t.Errorf("%s: %q — printed terms off by %.1f (≥ 1)", label, line, float64(diff)/10)
	}
}

// arithLamp builds one lamp variant; statuses come from the real rule.
func arithLamp(key, variant string) MacroLamp {
	l := MacroLamp{Key: key, Label: key, AsOf: "2026-09-14T13:30:00Z", Source: "yahoo"}
	if variant == "missing" {
		return l
	}
	v := 100.0
	var d *float64
	if key == "vix" {
		v = map[string]float64{"tail": 15, "neutral": 20, "head": 30, "nodelta": 15}[variant]
	} else if variant != "nodelta" {
		sign := 1.0
		if key == "spx" {
			sign = -1
		}
		x := sign * map[string]float64{"tail": -0.3, "neutral": 0.2, "head": 0.8}[variant]
		d = &x
	}
	l.Value, l.OK, l.DeltaPct = &v, true, d
	switch {
	case key == "vix":
		l.Status = macro.LampStatus(key, v, 0)
	case d != nil:
		l.Status = macro.LampStatus(key, v, *d)
	}
	return l
}

func TestMacroPrintedArithmeticAddsUp(t *testing.T) {
	keys := []string{"dxy", "rates", "vix", "spx", "gold"}
	variants := []string{"tail", "neutral", "head", "missing", "nodelta"}
	var eq, approx, lines int
	// Per view: how many payloads must produce a score line (a scored model)
	// and how many score lines were actually parsed — a regex that silently
	// stops matching must fail the test, not pass it vacuously.
	want, matched := map[string]int{}, map[string]int{}
	idx := make([]int, len(keys))
	for {
		lamps := make([]MacroLamp, len(keys))
		for i, k := range keys {
			lamps[i] = arithLamp(k, variants[idx[i]])
		}
		comp := macro.Composite(toMacroLamps(lamps))
		regime := macro.ClassifyRegime(comp, toMacroLamps(lamps))
		m := &MacroResp{Regime: regime, Composite: comp, TradfinOpen: true,
			CapturedAt: "2026-09-15T04:44:32Z", Lamps: lamps}
		g, _ := macroCardFrom(m)
		btc, gold := macroAssetCardFrom(m, macroAssetBTC), macroAssetCardFrom(m, macroAssetGold)
		if comp != nil {
			want["global"]++
			want["btc"]++
		}
		if goldViewScore(lamps) != nil {
			want["gold"]++
		}

		// Global: the breakdown line carries its terms.
		found := false
		for _, f := range g.Facts {
			mm := macroBreakdownRe.FindStringSubmatch(f)
			if mm == nil {
				continue
			}
			found = true
			matched["global"]++
			score, _ := strconv.Atoi(mm[1])
			sum := 0
			for _, term := range strings.Split(strings.SplitN(mm[3], " · ", 2)[0], " + ")[1:] {
				fs := strings.Fields(term)
				sum += tenthsOf(t, fs[len(fs)-1])
			}
			checkSum(t, "global", f, score, mm[2], sum)
			lines++
			if mm[2] == "=" {
				eq++
			} else {
				approx++
			}
		}
		if comp != nil && !found {
			t.Errorf("global %v: scored card without a breakdown line: %v", lamps, g.Facts)
		}
		// Asset views: the score line sums the lamp lines above it.
		for label, c := range map[string]Card{"btc": btc, "gold": gold} {
			sum := 0
			for _, f := range c.Facts {
				if mm := macroLampTailRe.FindStringSubmatch(f); mm != nil {
					sum += tenthsOf(t, mm[1])
				}
			}
			for _, f := range c.Facts {
				if mm := macroScoreLineRe.FindStringSubmatch(f); mm != nil {
					score, _ := strconv.Atoi(mm[1])
					checkSum(t, label, f, score, mm[2], sum)
					matched[label]++
					lines++
				}
			}
		}
		i := 0
		for ; i < len(idx); i++ {
			idx[i]++
			if idx[i] < len(variants) {
				break
			}
			idx[i] = 0
		}
		if i == len(idx) {
			break
		}
	}
	for _, v := range []string{"global", "btc", "gold"} {
		if matched[v] == 0 || matched[v] != want[v] {
			t.Errorf("%s: %d score lines parsed, want %d (one per scored payload)", v, matched[v], want[v])
		}
	}
	if eq == 0 || approx == 0 {
		t.Errorf("sweep never produced both signs: %d lines, %d \"=\", %d \"≈\"", lines, eq, approx)
	}

	// Codex's example: three live 25-weight lamps, all positive — exactly 100
	// unrounded, but printed terms add up to 100.1.
	lamps := []MacroLamp{arithLamp("dxy", "tail"), arithLamp("rates", "missing"), arithLamp("vix", "tail"),
		arithLamp("spx", "tail"), arithLamp("gold", "missing")}
	comp := macro.Composite(toMacroLamps(lamps))
	g, _ := macroCardFrom(&MacroResp{Regime: "risk_on", Composite: comp, TradfinOpen: true, Lamps: lamps})
	if !strings.Contains(strings.Join(g.Facts, "|"), "Rule score 100 ≈ 50 + DXY +16.7 + VIX +16.7 + S&P 500 +16.7") {
		t.Errorf("Codex example must read ≈: %v", g.Facts)
	}
}

// ── No validator ─────────────────────────────────────────────────────────────

// All three macro variants: no Last-Modified, and If-Modified-Since — even a
// future one — never turns a changed card (same captured_at) into a 304.
func TestHTTPMacroSendsNoValidator(t *testing.T) {
	var mu sync.Mutex
	payload := macroLiveFixture
	set := func(s string) { mu.Lock(); payload = s; mu.Unlock() }
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/macro" {
			http.NotFound(w, r)
			return
		}
		mu.Lock()
		b := payload
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(b))
	}))
	t.Cleanup(upstream.Close)
	_, api := newTestAPI(t, NewAgents(NewBackendClient(upstream.URL)), true)

	get := func(path, ims string) (status int, lastMod, dataAsOf, body string) {
		t.Helper()
		st, lm, raw := cget(t, api.URL+path, ims)
		if st == http.StatusOK {
			var env struct {
				DataAsOf string `json:"data_as_of"`
			}
			if err := json.Unmarshal(raw, &env); err != nil {
				t.Fatal(err)
			}
			dataAsOf = env.DataAsOf
		}
		return st, lm, dataAsOf, string(raw)
	}
	future := time.Date(2026, 9, 16, 0, 0, 0, 0, time.UTC).Format(http.TimeFormat)

	for _, path := range []string{"/agents/macro", "/agents/macro?asset=btc", "/agents/macro?asset=gold"} {
		set(macroLiveFixture)
		st1, lm1, asOf1, body1 := get(path, "")
		if st1 != 200 || lm1 != "" || asOf1 != "2026-09-14T07:00:00Z" {
			t.Fatalf("%s: first GET %d, Last-Modified %q, data_as_of %q; want 200, none, the oldest lamp", path, st1, lm1, asOf1)
		}
		if st, lm, _, b := get(path, future); st != 200 || lm != "" || b != body1 {
			t.Errorf("%s: unchanged payload + future If-Modified-Since → %d, Last-Modified %q; want 200, none, the same body", path, st, lm)
		}

		// A newer lamp changes (DXY: new value, new session stamp, now positive
		// → score 95) while the OLDEST lamp (VIX, Sep 14 07:00) and even
		// captured_at stay the same.
		changed := strings.Replace(macroLiveFixture,
			`"value":99.61299896240234,"ok":true,"delta_pct":0.13369079310498477,"status":"neutral","as_of":"2026-09-15T04:00:00Z"`,
			`"value":99.2,"ok":true,"delta_pct":-0.28,"status":"tailwind","as_of":"2026-09-15T05:00:00Z"`, 1)
		changed = strings.Replace(changed, `"composite":83`, `"composite":95`, 1)
		if changed == macroLiveFixture {
			t.Fatal("fixture replace did not apply")
		}
		set(changed)
		st2, lm2, asOf2, body2 := get(path, future)
		if st2 != http.StatusOK || lm2 != "" || body2 == body1 {
			t.Errorf("%s: changed card, same captured_at, future If-Modified-Since → %d, Last-Modified %q, changed=%v; want 200, none, the new body",
				path, st2, lm2, body2 != body1)
		}
		if asOf2 != asOf1 {
			t.Errorf("%s: data_as_of %q → %q must stay the oldest lamp", path, asOf1, asOf2)
		}
	}
}
