package demobot

import (
	"context"
	"testing"
	"time"
)

// readingOf is hookMacroReading over a raw GET body.
func readingOf(t *testing.T, body string) string {
	t.Helper()
	r, err := hookMacroReading([]byte(body), time.Time{}, time.Time{})
	if err != nil {
		t.Fatalf("reading: %v", err)
	}
	return r
}

// macroQuoteOnly are quote moves that leave every lamp's status and
// direction as they are: the hash moves, the reading does not.
var macroQuoteOnly = []struct {
	key  string
	move map[string]any
}{
	{"dxy", map[string]any{"value": 111.566}},
	{"rates", map[string]any{"value": 5.5563}},
	{"vix", map[string]any{"value": 19.152}},
	{"spx", map[string]any{"value": 8534.38}},
	{"gold", map[string]any{"value": 4864.94}},
}

// TestHookMacroReadingIgnoresQuotes: on all three addresses a quote move
// changes the hash (it is sent, as before) and leaves the reading.
func TestHookMacroReadingIgnoresQuotes(t *testing.T) {
	th, setMacro := newMacroHook(t)
	base := macroReads(t, th, setMacro, macroPayload(t))
	for _, c := range macroQuoteOnly {
		got := macroReads(t, th, setMacro, macroPayload(t, setLamp(t, c.key, c.move)))
		for _, p := range macroHookPaths {
			if got[p].hash == base[p].hash {
				t.Errorf("%s %s %v: the hash must move", c.key, p, c.move)
			}
			if readingOf(t, got[p].body) != readingOf(t, base[p].body) {
				t.Errorf("%s %s %v: a quote move must leave the reading", c.key, p, c.move)
			}
		}
	}
}

// TestHookMacroReadingSeesRealChange: a changed reading is never held.
func TestHookMacroReadingSeesRealChange(t *testing.T) {
	th, setMacro := newMacroHook(t)
	base := macroReads(t, th, setMacro, macroPayload(t))
	riskPaths := macroHookPaths[:2]
	cases := []struct {
		name  string
		paths []string
		edit  func(m map[string]any)
	}{
		// On the gold view the gold lamp is the card's subject, not an input
		// (no condition, no contribution): its move there is a quote.
		{"a lamp crosses its threshold", riskPaths, setLamp(t, "gold", map[string]any{
			"delta_pct": -0.42, "status": "tailwind"})},
		{"an input lamp crosses its threshold", macroHookPaths, setLamp(t, "spx", map[string]any{
			"delta_pct": -1.2, "status": "headwind"})},
		{"a lamp's as_of moves", macroHookPaths, setLamp(t, "dxy", map[string]any{
			"as_of": "2026-09-15T05:00:00Z"})},
		{"a lamp loses its value", macroHookPaths, setLamp(t, "dxy", map[string]any{
			"value": nil, "ok": false, "status": ""})},
		{"a lamp loses its session change", macroHookPaths, setLamp(t, "dxy", map[string]any{
			"delta_pct": nil})},
		{"a lamp leaves the set", macroHookPaths, func(m map[string]any) {
			m["lamps"] = asAnySlice(m["lamps"])[1:]
		}},
		{"the composite moves", riskPaths, func(m map[string]any) { m["composite"] = 71 }},
		{"the regime changes", riskPaths, func(m map[string]any) { m["regime"] = "mixed" }},
		{"Fear & Greed moves", macroHookPaths, func(m map[string]any) {
			m["fng"].(map[string]any)["value"] = 31
			m["fng"].(map[string]any)["label"] = "Fear"
		}},
	}
	for _, tc := range cases {
		got := macroReads(t, th, setMacro, macroPayload(t, tc.edit))
		for _, p := range tc.paths {
			if readingOf(t, got[p].body) == readingOf(t, base[p].body) {
				t.Errorf("%s: %s must change the reading", p, tc.name)
			}
		}
	}
}

// TestHookMacroQuoteOnlyWaits: through process() — a quote-only change waits
// hookMacroQuoteEvery after the last send, then the latest body goes; a
// reading change goes on the sweep that sees it, pause or not.
func TestHookMacroQuoteOnlyWaits(t *testing.T) {
	sink := newHookSink(t, 201)
	ag, setMacro := mutableMacroAgents(t)
	th := newTestHook(t, ag, sink.srv.URL, nil)
	now := time.Date(2026, 9, 24, 14, 0, 0, 0, time.UTC)
	th.clock = func() time.Time { return now }
	tg := mustHookTarget(t, "/agents/macro")
	st := &hookTargetState{}
	step := func(payload string, at time.Duration) int {
		now = time.Date(2026, 9, 24, 14, 0, 0, 0, time.UTC).Add(at)
		setMacro(payload)
		th.process(context.Background(), tg, st)
		return len(sink.take())
	}
	quote1 := macroPayload(t, setLamp(t, "dxy", map[string]any{"value": 111.566}))
	quote2 := macroPayload(t, setLamp(t, "vix", map[string]any{"value": 19.152}))
	reading := macroPayload(t, func(m map[string]any) { m["composite"] = 71 })

	if n := step(macroPayload(t), 0); n != 1 {
		t.Fatalf("first body: %d events, want 1", n)
	}
	if n := step(quote1, time.Minute); n != 0 {
		t.Errorf("quote-only change 1 min after a send: %d events, want 0", n)
	}
	if n := step(quote2, 14*time.Minute); n != 0 {
		t.Errorf("quote-only change 14 min after a send: %d events, want 0", n)
	}
	if n := step(quote2, 15*time.Minute); n != 1 {
		t.Errorf("quote-only change 15 min after a send: %d events, want 1", n)
	}
	if n := step(reading, 16*time.Minute); n != 1 {
		t.Errorf("reading change 1 min after a send: %d events, want 1", n)
	}
	if n := step(reading, 17*time.Minute); n != 0 {
		t.Errorf("same body again: %d events, want 0", n)
	}
}

// TestHookMacroUndeliveredReadingIsNotHeld: a reading change the site did
// not take (500) is still new to it — the next body with that reading goes
// on the next sweep, not after the quote pause.
func TestHookMacroUndeliveredReadingIsNotHeld(t *testing.T) {
	sink := newHookSink(t, 201)
	ag, setMacro := mutableMacroAgents(t)
	th := newTestHook(t, ag, sink.srv.URL, nil)
	base := time.Date(2026, 9, 24, 20, 0, 0, 0, time.UTC)
	now := base
	th.clock = func() time.Time { return now }
	tg := mustHookTarget(t, "/agents/macro")
	st := &hookTargetState{}
	step := func(payload string, at time.Duration) int {
		now = base.Add(at)
		setMacro(payload)
		th.process(context.Background(), tg, st)
		return len(sink.take())
	}
	reading := func(m map[string]any) { m["composite"] = 71 }
	if n := step(macroPayload(t), 0); n != 1 {
		t.Fatalf("first body: %d events, want 1", n)
	}
	sink.setCode(500)
	if n := step(macroPayload(t, reading), time.Minute); n != 1 {
		t.Fatalf("reading change: %d attempts, want 1", n)
	}
	sink.setCode(201)
	moved := macroPayload(t, reading, setLamp(t, "dxy", map[string]any{"value": 111.566}))
	if n := step(moved, 2*time.Minute); n != 1 {
		t.Errorf("undelivered reading with a new quote 1 min later: %d events, want 1", n)
	}
	moved2 := macroPayload(t, reading, setLamp(t, "vix", map[string]any{"value": 19.152}))
	if n := step(moved2, 3*time.Minute); n != 0 {
		t.Errorf("quote-only change after the reading was delivered: %d events, want 0", n)
	}
}

// TestHookMacroThrottleIsMacroOnly: no other agent keeps a reading.
func TestHookMacroThrottleIsMacroOnly(t *testing.T) {
	sink := newHookSink(t, 201)
	ag, setBody := mutableWhaleAgents(t)
	th := newTestHook(t, ag, sink.srv.URL, nil)
	tg := mustHookTarget(t, "/agents/whale")
	st := &hookTargetState{}
	setBody(whaleLiveFixture, 200)
	th.process(context.Background(), tg, st)
	if st.sentReading != "" {
		t.Errorf("whale kept a reading %q", st.sentReading)
	}
}
