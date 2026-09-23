package demobot

// hook_macro_noise_test.go — a macro lamp's live quote under a session stamp
// (hook.go "macro: a live quote under a session stamp"). Prod 2026-09-21 sent
// the macro address 480 events in a day, one every three minutes, all under
// the same data_as_of: two bodies 200 s apart differed only in the second
// decimal of a lamp percent ("Gold -0.48% (fell)" → "-0.47%") while the rule
// score, every contribution and every as_of stood still.
//
// Every case here is driven from the BACKEND PAYLOAD and recomputes the whole
// card: editing a finished body instead is what let the gold view's own lamp
// (lampSelfLine — no rule condition, no contribution) go unnoticed.
//
// What the tests pin:
//   - a quote that moves inside its bucket is not a new reading: one hash on
//     all three macro addresses, although the printed card moved;
//   - a quote that leaves its bucket IS one, on all three addresses — gold
//     walking 3 pp or 12% included, which is the only number the gold view
//     exists for;
//   - everything the rule reads off a lamp still fires an event;
//   - the buckets are the macro addresses' alone, and only where the card
//     renders a lamp: prose keeps its numbers;
//   - the body sent to the site is the GET body, quotes and all.

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"strings"
	"testing"
	"time"
)

var macroHookPaths = []string{"/agents/macro", "/agents/macro?asset=btc", "/agents/macro?asset=gold"}

// ── payload edits ────────────────────────────────────────────────────────────

// macroPayload is macroLiveFixture with edits applied to the BACKEND payload,
// so every card is recomputed from it exactly as the handler would.
func macroPayload(t *testing.T, edits ...func(m map[string]any)) string {
	t.Helper()
	dec := json.NewDecoder(strings.NewReader(macroLiveFixture))
	dec.UseNumber()
	var m map[string]any
	if err := dec.Decode(&m); err != nil {
		t.Fatalf("fixture: %v", err)
	}
	for _, e := range edits {
		e(m)
	}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("payload: %v", err)
	}
	return string(b)
}

// payloadLamp is one lamp of the backend payload, by key.
func payloadLamp(t *testing.T, m map[string]any, key string) map[string]any {
	t.Helper()
	for _, l := range asAnySlice(m["lamps"]) {
		if lm, ok := l.(map[string]any); ok && lm["key"] == key {
			return lm
		}
	}
	t.Fatalf("no lamp %q in the payload", key)
	return nil
}

// setLamp edits one lamp's fields; a nil value writes JSON null.
func setLamp(t *testing.T, key string, fields map[string]any) func(map[string]any) {
	return func(m map[string]any) {
		l := payloadLamp(t, m, key)
		for k, v := range fields {
			l[k] = v
		}
	}
}

// macroRead is one macro address read once.
type macroRead struct{ hash, body string }

// macroReads serves payload and reads all three macro addresses.
func macroReads(t *testing.T, th *testHook, setMacro func(string), payload string) map[string]macroRead {
	t.Helper()
	setMacro(payload)
	out := make(map[string]macroRead, len(macroHookPaths))
	for _, p := range macroHookPaths {
		_, b := th.fetch(context.Background(), mustHookTarget(t, p))
		out[p] = macroRead{hash: hashOf(t, keyMacro, b, time.Time{}, time.Time{}), body: string(b)}
	}
	return out
}

func newMacroHook(t *testing.T) (*testHook, func(string)) {
	t.Helper()
	ag, setMacro := mutableMacroAgents(t)
	return newTestHook(t, ag, "http://127.0.0.1:1", nil), setMacro
}

// ── the buckets ──────────────────────────────────────────────────────────────

// macroInsideBucket is, per lamp, a quote that MOVED and stayed in its
// bucket: the printed level and percent both change, the level keeps its
// three significant digits and the session change its tenth of a point. No
// status, score or stamp is touched, so the reading is the same reading.
var macroInsideBucket = map[string]map[string]any{
	// 99.61 → 99.62 (both 99.6) · +0.13% → +0.14% (both 0.1)
	"dxy": {"value": 99.6188, "delta_pct": 0.13977},
	// 4.9610 → 4.9633 (both 4.96) · -0.36% (both -0.4)
	"rates": {"value": 4.9633, "delta_pct": -0.3599},
	// 17.10 → 17.14 (both 17.1) · +7.95% → +7.96% (both 8.0)
	"vix": {"value": 17.14, "delta_pct": 7.96},
	// 7620 → 7624 (both 7.62e+03) · +0.11% (both 0.1)
	"spx": {"value": 7624, "delta_pct": 0.1149},
	// 4344 → 4340 (both 4.34e+03) · +0.08% → +0.09% (both 0.1)
	"gold": {"value": 4340.0, "delta_pct": 0.0896},
}

// TestHookMacroQuoteInsideItsBucket: the card text moves with the quote and
// the hash does not — every lamp, on the global card, the BTC view and the
// gold view.
func TestHookMacroQuoteInsideItsBucket(t *testing.T) {
	th, setMacro := newMacroHook(t)
	base := macroReads(t, th, setMacro, macroPayload(t))
	for _, key := range macroLampKeys {
		got := macroReads(t, th, setMacro, macroPayload(t, setLamp(t, key, macroInsideBucket[key])))
		for _, p := range macroHookPaths {
			if got[p].body == base[p].body {
				t.Fatalf("%s %s: the two bodies are identical, the tick moved nothing", key, p)
			}
			if got[p].hash != base[p].hash {
				t.Errorf("%s %s: a quote moving inside its bucket is not a new reading", key, p)
			}
		}
	}
}

// macroOutsideBucket is the same quote moved OUT of its bucket, each lamp
// twice: a session change three points away, and a level 12% away. Status,
// score, stamps and counts are untouched — the number alone must be enough.
var macroOutsideBucket = map[string][]map[string]any{
	"dxy":   {{"delta_pct": 3.13369}, {"value": 111.566}},
	"rates": {{"delta_pct": 2.63848}, {"value": 5.5563}},
	"vix":   {{"delta_pct": 10.9545}, {"value": 19.152}},
	"spx":   {{"delta_pct": 3.1122}, {"value": 8534.38}},
	"gold":  {{"delta_pct": 3.1}, {"delta_pct": -12.0}, {"value": 4864.94}},
}

// TestHookMacroQuoteOutsideItsBucket: a quote large enough to be a reading
// always reaches the site. The gold rows are the regression that the first
// attempt at this mask lost — on the gold view the gold lamp is the SUBJECT
// of the card (lampSelfLine: no rule condition, no contribution), so with its
// number dropped instead of bucketed, gold walking 12% in a session changed
// nothing at all.
func TestHookMacroQuoteOutsideItsBucket(t *testing.T) {
	th, setMacro := newMacroHook(t)
	base := macroReads(t, th, setMacro, macroPayload(t))
	for _, key := range macroLampKeys {
		for _, move := range macroOutsideBucket[key] {
			got := macroReads(t, th, setMacro, macroPayload(t, setLamp(t, key, move)))
			for _, p := range macroHookPaths {
				if got[p].hash == base[p].hash {
					t.Errorf("%s %s %v: a quote leaving its bucket must change the hash", key, p, move)
				}
			}
		}
	}
}

// ── everything that is a reading ─────────────────────────────────────────────

// TestHookMacroRealChangeAlwaysSends: every edit below is a changed READING,
// each applied to the untouched payload so they never hide one another.
func TestHookMacroRealChangeAlwaysSends(t *testing.T) {
	th, setMacro := newMacroHook(t)
	base := macroReads(t, th, setMacro, macroPayload(t))

	// The risk model's own fields (composite, regime) do not reach the gold
	// view: it reads the separate gold model, so they are checked where the
	// card actually carries them.
	riskPaths := macroHookPaths[:2]
	cases := []struct {
		name  string
		paths []string
		edit  func(m map[string]any)
	}{
		{"a lamp crosses its threshold", macroHookPaths, setLamp(t, "gold", map[string]any{
			"delta_pct": -0.42, "status": "tailwind"})},
		{"a lamp turns negative", macroHookPaths, setLamp(t, "spx", map[string]any{
			"delta_pct": -1.2, "status": "headwind"})},
		{"a lamp's as_of moves to the next session", macroHookPaths, setLamp(t, "dxy", map[string]any{
			"as_of": "2026-09-15T05:00:00Z"})},
		{"a lamp loses its value", macroHookPaths, setLamp(t, "dxy", map[string]any{
			"value": nil, "ok": false, "status": ""})},
		{"a lamp loses its session change", macroHookPaths, setLamp(t, "dxy", map[string]any{
			"delta_pct": nil})},
		{"a lamp changes provider", macroHookPaths, setLamp(t, "gold", map[string]any{"source": "stooq"})},
		{"a lamp leaves the set", macroHookPaths, func(m map[string]any) {
			m["lamps"] = asAnySlice(m["lamps"])[1:]
		}},
		{"the composite moves", riskPaths, func(m map[string]any) { m["composite"] = 71 }},
		{"the composite goes away", riskPaths, func(m map[string]any) { m["composite"] = nil }},
		{"the regime changes", riskPaths, func(m map[string]any) { m["regime"] = "mixed" }},
		{"the tradfin week closes", macroHookPaths, func(m map[string]any) { m["tradfin_market_open"] = false }},
		{"Fear & Greed moves", macroHookPaths, func(m map[string]any) {
			m["fng"].(map[string]any)["value"] = 31
			m["fng"].(map[string]any)["label"] = "Fear"
		}},
		{"the freshest tradfin stamp moves", macroHookPaths, func(m map[string]any) {
			m["tradfin_as_of"] = "2026-09-15T05:00:00Z"
		}},
	}
	for _, tc := range cases {
		got := macroReads(t, th, setMacro, macroPayload(t, tc.edit))
		for _, p := range tc.paths {
			if got[p].hash == base[p].hash {
				t.Errorf("%s: %s must change the hash", p, tc.name)
			}
		}
	}
}

// TestHookMacroScoreIsNeverBucketed: the score the card prints is the
// reading itself — a rounded one and an unrounded one both count, and the
// rule thresholds beside a lamp are not numbers of a quote.
func TestHookMacroScoreIsNeverBucketed(t *testing.T) {
	th, _ := newMacroHook(t)
	zero := time.Time{}
	_, body := th.fetch(context.Background(), mustHookTarget(t, "/agents/macro"))
	base := hashOf(t, keyMacro, body, zero, zero)

	for _, tc := range []struct {
		name string
		edit func(env map[string]any)
	}{
		{"rule_score", func(env map[string]any) {
			env["macro"].(map[string]any)["rule_score"] = json.Number("82")
		}},
		{"rule_score_unrounded", func(env map[string]any) {
			env["macro"].(map[string]any)["rule_score_unrounded"] = json.Number("82.4")
		}},
		{"a printed contribution", func(env map[string]any) {
			macroEditText(t, env, " → +12.5", " → +7.5")
		}},
		{"the side a lamp is filed under", func(env map[string]any) {
			macroEditText(t, env, upperFirst(contribPositive)+" for ", upperFirst(contribNegative)+" for ")
		}},
		{"the rule condition beside a lamp", func(env map[string]any) {
			macroEditText(t, env, "(<18)", "(18-25)")
		}},
	} {
		env := macroDecode(t, body)
		tc.edit(env)
		if hashOf(t, keyMacro, macroEncode(t, env), zero, zero) == base {
			t.Errorf("%s must change the hash", tc.name)
		}
	}
}

// ── where the buckets apply ──────────────────────────────────────────────────

// TestHookMacroBucketsAreMacroOnly: the same body under any other agent keeps
// every lamp number in its hash.
func TestHookMacroBucketsAreMacroOnly(t *testing.T) {
	th, _ := newMacroHook(t)
	zero := time.Time{}
	_, body := th.fetch(context.Background(), mustHookTarget(t, "/agents/macro"))

	env := macroDecode(t, body)
	macroLampOf(t, env, "dxy")["value"] = json.Number("99.6188")
	macroEditText(t, env, "VIX 17.10", "VIX 17.14")
	moved := macroEncode(t, env)

	for _, agent := range []string{keyDigest, keyTop, keyWhale, keyFunding, keyNews, keyMomentum, keyTrend, keyFX} {
		if hashOf(t, agent, body, zero, zero) == hashOf(t, agent, moved, zero, zero) {
			t.Errorf("%s: the macro buckets must not reach this agent", agent)
		}
	}
	if hashOf(t, keyMacro, body, zero, zero) != hashOf(t, keyMacro, moved, zero, zero) {
		t.Error("macro: the same edit is inside the bucket on the macro address")
	}
}

// TestHookMacroWinnerElsewhereKeepsItsNumbers: a macro object inside another
// agent's body (digest/top with a macro winner) is untouched.
func TestHookMacroWinnerElsewhereKeepsItsNumbers(t *testing.T) {
	th := newTestHook(t, liveHookAgents(t), "http://127.0.0.1:1", nil)
	ctx, zero := context.Background(), time.Time{}
	for _, tg := range hookTargets() {
		if tg.Agent == keyMacro {
			continue
		}
		_, body := th.fetch(ctx, tg)
		norm, err := hookNormalize(tg.Agent, body, zero, zero)
		if err != nil {
			t.Fatalf("%s: %v", tg.Path, err)
		}
		var got map[string]any
		if err := json.Unmarshal(norm, &got); err != nil {
			t.Fatalf("%s: %v", tg.Path, err)
		}
		m, ok := got["macro"].(map[string]any)
		if !ok {
			continue
		}
		for _, l := range asAnySlice(m["lamps"]) {
			lm, ok := l.(map[string]any)
			if !ok {
				continue
			}
			for _, k := range []string{"value", "delta_pct"} {
				if _, isString := lm[k].(string); isString {
					t.Errorf("%s: a macro winner inside another agent must keep its %s", tg.Path, k)
				}
			}
		}
	}
}

// TestHookMacroProseKeepsItsNumbers: the buckets apply to the fields that
// RENDER a lamp (facts[], card_html), never to free prose — a sentence
// naming a lamp beside a number is not a lamp reading, and rewriting it
// would hide a regenerated text.
func TestHookMacroProseKeepsItsNumbers(t *testing.T) {
	th, _ := newMacroHook(t)
	zero := time.Time{}
	_, body := th.fetch(context.Background(), mustHookTarget(t, "/agents/macro"))

	const prose = "DXY 99.61 broke out and VIX 17.10 held"
	env := macroDecode(t, body)
	env["ai_text"] = prose
	env["reason"] = prose
	env["blocks"].(map[string]any)["what_happened"] = prose
	withProse := macroEncode(t, env)

	norm, err := hookNormalize(keyMacro, withProse, zero, zero)
	if err != nil {
		t.Fatal(err)
	}
	out := macroDecode(t, norm)
	for _, k := range []string{"ai_text", "reason"} {
		if out[k] != prose {
			t.Errorf("%s: prose must pass unchanged, got %q", k, out[k])
		}
	}
	if got := out["blocks"].(map[string]any)["what_happened"]; got != prose {
		t.Errorf("blocks: prose must pass unchanged, got %q", got)
	}
	// Same body, same numbers: the facts ARE bucketed, so the test above
	// shows selectivity and not a mask that stopped working.
	if f := asAnySlice(out["facts"]); len(f) == 0 || !strings.Contains(f[0].(string), "VIX 17.1 ") {
		t.Errorf("facts must be bucketed, got %v", f)
	}
	// And a moved prose is still a change.
	env["ai_text"] = "DXY 99.62 broke out and VIX 17.14 held"
	if hashOf(t, keyMacro, withProse, zero, zero) == hashOf(t, keyMacro, macroEncode(t, env), zero, zero) {
		t.Error("a regenerated text must change the hash")
	}
}

// ── the wire ─────────────────────────────────────────────────────────────────

// TestHookMacroDataCarriesTheQuote: the buckets live inside change detection.
// What the site receives is the GET body, quotes and all.
func TestHookMacroDataCarriesTheQuote(t *testing.T) {
	ag, _ := mutableMacroAgents(t)
	sink := newHookSink(t, http.StatusCreated)
	th := newTestHook(t, ag, sink.srv.URL, []hookTarget{mustHookTarget(t, "/agents/macro")})
	th.sweep(context.Background())

	sink.mu.Lock()
	evs := append([]hookEvent(nil), sink.events...)
	sink.mu.Unlock()
	if len(evs) != 1 {
		t.Fatalf("want one event, got %d", len(evs))
	}
	_, body := th.fetch(context.Background(), mustHookTarget(t, "/agents/macro"))
	if string(evs[0].Data) != string(body) {
		t.Errorf("data is not the GET body:\n%s\n%s", evs[0].Data, body)
	}
	env := macroDecode(t, evs[0].Data)
	v, ok := macroLampOf(t, env, "dxy")["value"].(json.Number)
	if !ok || v.String() != "99.61299896240234" {
		t.Errorf("the event must carry the raw quote, got %#v", macroLampOf(t, env, "dxy")["value"])
	}
	for _, f := range asAnySlice(env["facts"]) {
		if s, _ := f.(string); strings.Contains(s, "DXY 99.6,") || strings.Contains(s, "VIX 17.1 ") {
			t.Errorf("a bucket leaked into the body sent to the site: %q", s)
		}
	}
}

// ── the renderings, one by one ───────────────────────────────────────────────

// TestHookMacroLampGroupsMatch: every capture group of the pattern has a
// bucket. A new alternative without one would silently hash raw.
func TestHookMacroLampGroupsMatch(t *testing.T) {
	if got, want := len(hookMacroLampBuckets), hookMacroLampRe.NumSubexp(); got != want {
		t.Fatalf("%d buckets for %d capture groups", got, want)
	}
	// The count alone let a swapped pair through: each group is checked
	// against the bucket it is paired with, by feeding one rendering per
	// alternative and reading which group caught it.
	probes := []struct {
		name  string
		in    string
		group int
		kind  string // "level" or "pct"
	}{
		{"lampFull level", "DXY 100.29, session +0.13% (0 to +0.5%)", 1, "level"},
		{"lampFull session", "DXY 100.29, session +0.13% (0 to +0.5%)", 2, "pct"},
		{"no session change", "DXY 100.29 (no session change)", 3, "level"},
		{"the asset itself", "Gold (GC=F futures) 4344 — the asset itself", 4, "level"},
		{"bare session", "Gold +0.08% (fell)", 5, "pct"},
		{"bare VIX level", "VIX 14.21 (<18)", 6, "level"},
	}
	// One probe per group: a new group added without its own probe would
	// otherwise pass here whatever bucket it was paired with.
	// Every group 1..N is probed exactly once: counting probes alone let a new
	// group through when another group's probe was duplicated.
	covered := map[int]int{}
	for _, tc := range probes {
		covered[tc.group]++
	}
	for g := 1; g <= hookMacroLampRe.NumSubexp(); g++ {
		if covered[g] != 1 {
			t.Errorf("capture group %d has %d probes, want exactly 1", g, covered[g])
		}
	}
	if len(covered) != hookMacroLampRe.NumSubexp() {
		t.Errorf("probes name %d groups, the pattern has %d", len(covered), hookMacroLampRe.NumSubexp())
	}
	for _, tc := range probes {
		m := hookMacroLampRe.FindStringSubmatchIndex(tc.in)
		if m == nil {
			t.Errorf("%s: no match for %q", tc.name, tc.in)
			continue
		}
		g := tc.group
		if 2*g+1 >= len(m) || m[2*g] < 0 {
			t.Errorf("%s: group %d did not catch %q", tc.name, g, tc.in)
			continue
		}
		want := hookMacroLevelBucket
		if tc.kind == "pct" {
			want = hookMacroPctBucket
		}
		gotBucket, wantBucket := hookMacroLampBuckets[g-1](1.25), want(1.25)
		if gotBucket != wantBucket {
			t.Errorf("%s: group %d is paired with the %s bucket (%q, want %q)",
				tc.name, g, map[bool]string{true: "pct", false: "level"}[tc.kind != "pct"], gotBucket, wantBucket)
		}
	}
}

// Idempotence over every renderer output, not a handful of picked strings.
// A picked set is what let a wrong fix through: the self-line test that stood
// here passed on the code it claimed to fix. This sweeps lampFull, lampCause
// and lampSelfLine over every lamp (plus one the rule does not know), source,
// level, session change and status, inside the wrappings the card uses, and
// requires hookMacroLampText(hookMacroLampText(s)) == hookMacroLampText(s)
// for all of them — the ones real renderers print and the synthetic rest.
//
// What broke it before: a first pass buckets a session change to "0.0%",
// which carries no sign, so the second pass no longer sees the lampFull form
// and the VIX bare-level pattern bit into a long level ("VIX 0.0001" became
// "VIX 001"). That pattern now ends on a word boundary.
func TestHookMacroLampTextIsIdempotent(t *testing.T) {
	f := func(v float64) *float64 { return &v }
	vals := []*float64{nil, f(0), f(0.0001), f(-0.0001), f(0.0050), f(0.0123), f(1.2345), f(9.9999),
		f(4343.7001953125), f(4343.70), f(4344), f(4865), f(100.2939987182617), f(99.61), f(99.6188),
		f(4.9610), f(17.14), f(14.21), f(7620), f(7619.98), f(-12.34), f(-0.48), f(18), f(25),
		f(999.99), f(1000), f(1e7), f(-1e7), f(1e9), f(1e-9), f(0.005), f(123456.789), f(0.449), f(0.451)}
	deltas := []*float64{nil, f(0), f(-0.0001), f(0.0001), f(0.04), f(-0.04), f(0.07834460240114281),
		f(0.13), f(-0.36), f(-0.3599), f(-0.48), f(-0.47), f(0.5), f(0.49), f(0.51), f(0.46), f(0.54),
		f(3.1), f(-12), f(999.99), f(1000), f(-1000), f(5000), f(-5000), f(7.9545), f(0.1149)}
	statuses := []string{"", "tailwind", "neutral", "headwind"}
	sources := []string{"", "yahoo", "stooq", "other"}
	// Each wrapping alone catches the same base forms (review, round 5), so a
	// -race build — which checks nothing in this single-goroutine sweep and
	// multiplies its cost ~25x — keeps three: bare, HTML-escaped, two lines.
	wrap := func(s string) []string {
		if raceBuild {
			return []string{s, esc(s), s + "\n" + s}
		}
		return []string{s,
			"Positive for rule score: " + s + " → +5.0",
			"Negative for rule score: " + s + " → neutral, 0",
			"• " + s + " · next",
			esc(s),
			"• " + esc("Positive for rule score: "+s+" → +12.5"),
			s + "\n" + s,
		}
	}
	seen, bad := 0, 0
	done := map[string]bool{} // the grid repeats renderings; each is checked once
	check := func(s string) {
		for _, in := range wrap(s) {
			if done[in] {
				continue
			}
			done[in] = true
			seen++
			once := hookMacroLampText(in)
			if twice := hookMacroLampText(once); once != twice {
				bad++
				if bad <= 5 {
					t.Errorf("not idempotent\n in %q\n 1x %q\n 2x %q", in, once, twice)
				}
			}
		}
	}
	for _, k := range append(append([]string{}, macroLampKeys...), "oil") {
		for _, src := range sources {
			for _, v := range vals {
				for _, d := range deltas {
					for _, st := range statuses {
						l := MacroLamp{Key: k, Label: k, Value: v, OK: v != nil, DeltaPct: d, Status: st, Source: src, AsOf: "2026-09-21T10:00:00Z"}
						check(lampFull(l))
						check(lampCause(l))
						if v != nil {
							check(lampSelfLine(l))
						}
					}
				}
			}
		}
	}
	if bad > 0 {
		t.Errorf("%d of %d renderings are not idempotent", bad, seen)
	}
	// The floor sits just under today's count, so a shrinking grid is noticed.
	floor := 160000
	if raceBuild {
		floor = hookMacroIdemRaceFloor
	}
	if seen < floor {
		t.Errorf("sweep shrank to %d distinct renderings (floor %d): the grid no longer covers what it did", seen, floor)
	}
	if testing.Verbose() {
		t.Logf("idempotence sweep: %d distinct renderings (race=%v)", seen, raceBuild)
	}
}

// TestHookMacroLampTextBuckets pins the renderer-by-renderer contract: the
// numbers are coarsened, the name, the rule condition, the contribution and
// anything that is not a lamp reading stay byte for byte.
func TestHookMacroLampTextBuckets(t *testing.T) {
	cases := []struct{ in, want string }{
		// lampCause in a factor line — the prod pair of 2026-09-21, one bucket
		{"Positive for rule score: VIX 14.94 (<18) → +12.5 · Gold -0.48% (fell) → +5.0",
			"Positive for rule score: VIX 14.9 (<18) → +12.5 · Gold -0.5% (fell) → +5.0"},
		{"Positive for rule score: VIX 14.94 (<18) → +12.5 · Gold -0.47% (fell) → +5.0",
			"Positive for rule score: VIX 14.9 (<18) → +12.5 · Gold -0.5% (fell) → +5.0"},
		// the same line as card_html escapes it
		{"• Positive for rule score: VIX 14.94 (&lt;18) → +12.5 · S&amp;P 500 +0.11% (rose) → +12.5",
			"• Positive for rule score: VIX 14.9 (&lt;18) → +12.5 · S&amp;P 500 0.1% (rose) → +12.5"},
		// lampFull, asset view: level and session change, outcome kept
		{"DXY 99.61, session +0.13% (0 to +0.5%) → neutral, 0", "DXY 99.6, session 0.1% (0 to +0.5%) → neutral, 0"},
		{"US 10Y 4.9610, session -0.36% (fell) → positive, +7.5", "US 10Y 4.96, session -0.4% (fell) → positive, +7.5"},
		{"S&P 500 7620, session +0.11% (rose) → negative for gold, -5.0",
			"S&P 500 7.62e+03, session 0.1% (rose) → negative for gold, -5.0"},
		// lampFull without a session change, and the gold instrument
		{"Gold (GC=F futures) 4344 (no session change) → not voting",
			"Gold (GC=F futures) 4.34e+03 (no session change) → not voting"},
		{"Gold (XAUUSD spot) 4343.70, session +0.08% (0 to +0.5%) → neutral, 0",
			"Gold (XAUUSD spot) 4.34e+03, session 0.1% (0 to +0.5%) → neutral, 0"},
		// lampSelfLine, both shapes — the gold view's own subject
		{"Gold (GC=F futures) 4344, session +0.08% — the asset itself, not an input",
			"Gold (GC=F futures) 4.34e+03, session 0.1% — the asset itself, not an input"},
		{"Gold (GC=F futures) 4344 — the asset itself, not an input",
			"Gold (GC=F futures) 4.34e+03 — the asset itself, not an input"},
		{"Gold (GC=F futures) 4865, session -12.00% — the asset itself, not an input",
			"Gold (GC=F futures) 4.86e+03, session -12.0% — the asset itself, not an input"},
		// the clamps are not numbers the card computed: left as printed
		{"Gold >+999% (rose >0.5%) → 0", "Gold >+999% (rose >0.5%) → 0"},
		{"VIX n/a (<18) → positive, +12.5", "VIX n/a (<18) → positive, +12.5"},
		{"DXY >9999999, session <-999% (fell) → positive, +12.5", "DXY >9999999, session <-999% (fell) → positive, +12.5"},
		// a lamp with no reading at all: nothing to bucket, and it is a change
		{"US 10Y — no data (last seen Sep 11) → not voting", "US 10Y — no data (last seen Sep 11) → not voting"},
		{"VIX (no session change) → not voting", "VIX (no session change) → not voting"},
		// everything else on a macro card keeps its numbers
		{"Rule score 83 ≈ 50 + US 10Y +7.5 + VIX +12.5 + S&P 500 +12.5 · neutral: DXY, Gold",
			"Rule score 83 ≈ 50 + US 10Y +7.5 + VIX +12.5 + S&P 500 +12.5 · neutral: DXY, Gold"},
		{"Weights: DXY 40 · US 10Y 25 · VIX 25 · S&P 500 10 (thresholds are the risk model's)",
			"Weights: DXY 40 · US 10Y 25 · VIX 25 · S&P 500 10 (thresholds are the risk model's)"},
		{"Data: 5 of 5 lamps live · Sep 14: US 10Y, VIX, S&P 500 · Sep 15: DXY, Gold",
			"Data: 5 of 5 lamps live · Sep 14: US 10Y, VIX, S&P 500 · Sep 15: DXY, Gold"},
		{"Risk-on ends at a rule score of 65 or below, or with fewer than 3 voting lamps (now 83, 5 voting)",
			"Risk-on ends at a rule score of 65 or below, or with fewer than 3 voting lamps (now 83, 5 voting)"},
		{"Crypto Fear & Greed 69 (Greed), Sep 15 · separate index, not in the rule score",
			"Crypto Fear & Greed 69 (Greed), Sep 15 · separate index, not in the rule score"},
		{"Liquidations 1h: $90K longs vs $45K shorts", "Liquidations 1h: $90K longs vs $45K shorts"},
		// a lamp key the rule does not know keeps its number (never bucketed wrongly)
		{"Oil 71.40, session +0.13% → not voting", "Oil 71.40, session +0.13% → not voting"},
	}
	for _, tc := range cases {
		got := hookMacroLampText(tc.in)
		if got != tc.want {
			t.Errorf("bucket %q\n got: %q\nwant: %q", tc.in, got, tc.want)
		}
		if again := hookMacroLampText(got); again != got {
			t.Errorf("not idempotent: %q → %q", got, again)
		}
	}
}

// TestHookMacroBucketEdges pins the two buckets, the zero spelling included:
// a move of nothing must have one spelling, or "-0.00%" and "+0.00%" would
// be two readings.
func TestHookMacroBucketEdges(t *testing.T) {
	for _, tc := range []struct {
		v    float64
		want string
	}{
		{0.13369079310498477, "0.1"}, {0.13977, "0.1"}, {-0.48, "-0.5"}, {-0.47, "-0.5"},
		{3.1, "3.1"}, {-12, "-12.0"}, {0, "0.0"}, {math.Copysign(0, -1), "0.0"}, {-0.04, "0.0"},
		// the documented edge: two reads either side of 0.05 are two buckets
		{0.049, "0.0"}, {0.051, "0.1"},
	} {
		if got := hookMacroPctBucket(tc.v); got != tc.want {
			t.Errorf("pct %v = %q, want %q", tc.v, got, tc.want)
		}
	}
	for _, tc := range []struct {
		v    float64
		want string
	}{
		{99.61299896240234, "99.6"}, {99.62, "99.6"}, {100.294, "100"}, {100.27, "100"},
		{4391.9, "4.39e+03"}, {4392.4, "4.39e+03"}, {4864.94, "4.86e+03"},
		{17.100000381469727, "17.1"}, {4.960999965667725, "4.96"}, {0, "0"},
		{math.Copysign(0, -1), "0"}, {-17.16, "-17.2"},
	} {
		if got := hookMacroLevelBucket(tc.v); got != tc.want {
			t.Errorf("level %v = %q, want %q", tc.v, got, tc.want)
		}
	}
}

// ── helpers ──────────────────────────────────────────────────────────────────

func macroDecode(t *testing.T, b []byte) map[string]any {
	t.Helper()
	dec := json.NewDecoder(strings.NewReader(string(b)))
	dec.UseNumber()
	var env map[string]any
	if err := dec.Decode(&env); err != nil {
		t.Fatalf("decode: %v", err)
	}
	return env
}

func macroEncode(t *testing.T, env map[string]any) []byte {
	t.Helper()
	b, err := encodeJSON(env)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	return b
}

// macroLampOf is one lamp of the machine readout, by key.
func macroLampOf(t *testing.T, env map[string]any, key string) map[string]any {
	t.Helper()
	m, ok := env["macro"].(map[string]any)
	if !ok {
		t.Fatal("body carries no macro readout")
	}
	for _, l := range asAnySlice(m["lamps"]) {
		if lm, ok := l.(map[string]any); ok && lm["key"] == key {
			return lm
		}
	}
	t.Fatalf("no lamp %q", key)
	return nil
}

// macroEditText rewrites old → new wherever the card prints it, and fails
// when the body does not print it at all.
func macroEditText(t *testing.T, env map[string]any, old, new string) {
	t.Helper()
	hits := 0
	for _, k := range []string{"verdict", "card_html"} {
		if s, ok := env[k].(string); ok && strings.Contains(s, old) {
			env[k] = strings.ReplaceAll(s, old, new)
			hits++
		}
	}
	for i, f := range asAnySlice(env["facts"]) {
		if s, ok := f.(string); ok && strings.Contains(s, old) {
			env["facts"].([]any)[i] = strings.ReplaceAll(s, old, new)
			hits++
		}
	}
	if hits == 0 {
		t.Fatalf("the card does not print %q", old)
	}
}

// The bare-number alternative belongs to VIX only. As a free alternative its
// `-?\d+\.\d\d` matched the first two decimals of ANY lamp's level and left
// the rest of the digits behind, which is wrong and not idempotent. Today's
// renderers cannot reach it (sessionPctShown always prints two decimals or a
// clamp, so the first alternative wins), but a stray tail could collapse two
// different levels into one hash.
func TestHookMacroBareNumberIsVIXOnly(t *testing.T) {
	for _, tc := range []struct{ name, in string }{
		{"long gold level", "Gold 4343.7001953125, session +0.07834460240114281% (fell)"},
		{"tiny gold level", "Gold (XAUUSD spot) 0.0001, session -0.01% (fell) \u2192 +5.0"},
		{"long dxy level", "DXY 100.2939987182617, session +0.134% (rose)"},
	} {
		once := hookMacroLampText(tc.in)
		if twice := hookMacroLampText(once); once != twice {
			t.Errorf("%s: not idempotent\n 1x %q\n 2x %q", tc.name, once, twice)
		}
		if strings.Contains(once, "e+0301953125") || strings.Contains(once, "e+03019") {
			t.Errorf("%s: a chewed tail survived: %q", tc.name, once)
		}
	}
	// VIX still buckets its own bare level, condition and contribution intact.
	const vix = "VIX 14.21 (<18) \u2192 +14.7"
	got := hookMacroLampText(vix)
	if got == vix {
		t.Errorf("VIX bare level not bucketed: %q", got)
	}
	if !strings.Contains(got, "(<18)") || !strings.Contains(got, "+14.7") {
		t.Errorf("VIX condition or contribution eaten: %q", got)
	}
}

// hookMacroIdemRaceFloor is the distinct-rendering floor of the -race sweep
// (three wrappings instead of seven), just under its count today.
const hookMacroIdemRaceFloor = 60000
