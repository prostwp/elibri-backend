package demobot

// gold_idea_test.go — Gold stage 2 (2026-09-16): the setup-structure block.
//
// The block is presentation over numbers the agent ALREADY computes: the day
// range edge (trigger), the trend card's invalidation level, and the nearest
// clustered level on the other side of price. No rule, threshold or model moves, and
// the card must not grow a promise: the history run found no edge the sample
// could detect, so the block describes structure and says so — at that
// strength, never stronger — on the card itself.

import (
	"encoding/json"
	"strings"
	"testing"
	"unicode/utf8"
)

// goldIdeaBlock is the tail of the card: everything from the "Setup structure"
// (or "No setup structure") line on.
func goldIdeaBlock(c Card) []string {
	for i, f := range c.Facts {
		if strings.HasPrefix(f, "Setup structure") || strings.HasPrefix(f, "No setup structure") {
			return c.Facts[i:]
		}
	}
	return nil
}

// The block sits AFTER the stage-1 facts and repeats their numbers — it never
// replaces a line and never introduces a level of its own.
func TestGoldIdeaLinesConfirmed(t *testing.T) {
	cases := map[string]struct {
		state string
		want  []string
		nums  []string // trigger, invalidation, reference — each already printed above
	}{
		"up": {trendUp, []string{
			"Setup structure: a daily close above 4396.80 (day range high) is the trigger",
			"Invalidated by a closed 1d candle below 4040.00; nearest level below price: support 4329.20",
			"Structure, not a forecast: 10 years of history showed no edge this sample could detect",
		}, []string{"4396.80", "4040.00", "4329.20"}},
		"down": {trendDown, []string{
			"Setup structure: a daily close below 4293.00 (day range low) is the trigger",
			"Invalidated by a closed 1d candle above 4660.00; nearest level above price: resistance 4364.50",
			"Structure, not a forecast: 10 years of history showed no edge this sample could detect",
		}, []string{"4293.00", "4660.00", "4364.50"}},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			c := goldCardFrom(goldFixture(tc.state))
			got := goldIdeaBlock(c)
			if strings.Join(got, "\n") != strings.Join(tc.want, "\n") {
				t.Errorf("idea block:\n%s\nwant:\n%s", strings.Join(got, "\n"), strings.Join(tc.want, "\n"))
			}
			// It is the tail of the card: the stage-1 lines keep their order.
			if n := len(c.Facts); n < 3 || strings.Join(c.Facts[n-3:], "\n") != strings.Join(tc.want, "\n") {
				t.Errorf("the block must close the card: %v", c.Facts)
			}
			// Every number in the block is one the card has already printed:
			// the block introduces no level of its own.
			body := strings.Join(c.Facts[:len(c.Facts)-3], "\n")
			for _, num := range tc.nums {
				if !strings.Contains(body, num) {
					t.Errorf("%s is not a number the card already printed:\n%s", num, body)
				}
				if !strings.Contains(strings.Join(got, "\n"), num) {
					t.Errorf("%s is missing from the block: %v", num, got)
				}
			}
			// blocks are stage 1 and must not grow the idea a second time.
			b, err := json.Marshal(c.Blocks)
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(string(b), "Setup structure") {
				t.Errorf("blocks must stay as shipped: %s", b)
			}
		})
	}
}

// The disclaimer is pinned word for word, because its STRENGTH is the point.
// The ten-year run distinguishes effects of about 12 pp and larger
// (Отчёт_прогона_золотой_агент.md, section 7), so "no edge this sample could
// detect" is what was earned. "No edge over the baseline" would turn a limit
// of the measurement into a finding about the market — the flattering
// direction, and the one an investor would read as proof.
func TestGoldIdeaDisclaimerIsTheWeakClaim(t *testing.T) {
	const want = "Structure, not a forecast: 10 years of history showed no edge this sample could detect"
	if goldIdeaDisclaimer != want {
		t.Errorf("disclaimer = %q, want %q", goldIdeaDisclaimer, want)
	}
	if n := utf8.RuneCountInString(goldIdeaDisclaimer); n > goldFactMaxRunes {
		t.Errorf("disclaimer is %d runes, over the %d budget", n, goldFactMaxRunes)
	}
	if !strings.Contains(goldIdeaDisclaimer, "this sample could detect") {
		t.Errorf("the claim must stay bounded by the sample: %q", goldIdeaDisclaimer)
	}
	// No path — card text or blocks — may carry the stronger version.
	strong := []string{"no edge over the baseline", "no edge exists", "no predictive edge", "proves", "proven"}
	for key, c := range goldAllCards() {
		b, err := json.Marshal(c.Blocks)
		if err != nil {
			t.Fatal(err)
		}
		all := c.RenderHTML() + "\n" + string(b)
		for _, s := range strong {
			if strings.Contains(all, s) {
				t.Errorf("%s: %q claims more than the run measured:\n%s", key, s, all)
			}
		}
	}
	// And it closes every card that names a structure.
	for _, st := range []string{trendUp, trendDown} {
		c := goldCardFrom(goldFixture(st))
		if c.Facts[len(c.Facts)-1] != goldIdeaDisclaimer {
			t.Errorf("%s: the card must end with the disclaimer: %v", st, c.Facts)
		}
	}
}

// Up and down are the same sentence mirrored — no side gets softer wording.
func TestGoldIdeaSymmetry(t *testing.T) {
	mirror := strings.NewReplacer(
		"above", "below", "below", "above", "high", "low", "low", "high",
		"support", "resistance", "resistance", "support",
		"4396.80", "4293.00", "4293.00", "4396.80",
		"4040.00", "4660.00", "4660.00", "4040.00",
		"4329.20", "4364.50", "4364.50", "4329.20")
	up := goldIdeaBlock(goldCardFrom(goldFixture(trendUp)))
	down := goldIdeaBlock(goldCardFrom(goldFixture(trendDown)))
	if len(up) != 3 || len(down) != 3 {
		t.Fatalf("both sides need the block: %v / %v", up, down)
	}
	for i := range up {
		if got := mirror.Replace(up[i]); got != down[i] {
			t.Errorf("line %d does not mirror:\n%q\n%q", i, got, down[i])
		}
	}
}

// No confirmed regime, no price, no day range, no invalidation level → no
// idea at all: a null field and a line that says why.
func TestGoldIdeaAbsent(t *testing.T) {
	unconfirmed := "No setup structure without a confirmed regime and a last closed 1h price"
	for _, st := range []string{trendGrey, trendFlat, trendConflict} {
		c := goldCardFrom(goldFixture(st))
		if got := goldIdeaBlock(c); len(got) != 1 || got[0] != unconfirmed {
			t.Errorf("%s: %v, want %q", st, got, unconfirmed)
		}
		if c.Gold == nil || c.Gold.Idea != nil {
			t.Errorf("%s: gold.idea must be null: %+v", st, c.Gold)
		}
	}

	noPx := goldFixture(trendUp)
	noPx.hasPx, noPx.px, noPx.sup, noPx.res = false, 0, nil, nil
	c := goldCardFrom(noPx)
	if got := goldIdeaBlock(c); len(got) != 1 || got[0] != unconfirmed {
		t.Errorf("no price: %v, want %q", got, unconfirmed)
	}
	if c.Gold.Idea != nil {
		t.Errorf("no price: gold.idea must be null: %+v", c.Gold.Idea)
	}

	noRange := goldFixture(trendUp)
	noRange.levels = goldDayLevels{}
	c = goldCardFrom(noRange)
	want := "No setup structure: no day range to trigger against"
	if got := goldIdeaBlock(c); len(got) != 1 || got[0] != want {
		t.Errorf("no range: %v, want %q", got, want)
	}
	if c.Gold.Idea != nil {
		t.Errorf("no range: gold.idea must be null: %+v", c.Gold.Idea)
	}

	noInv := goldFixture(trendDown)
	lv := noInv.trend.Levels.(TrendLevels)
	lv.Invalidation = nil
	noInv.trend.Levels = lv
	c = goldCardFrom(noInv)
	want = "No setup structure: this regime read carries no invalidation level"
	if got := goldIdeaBlock(c); len(got) != 1 || got[0] != want {
		t.Errorf("no invalidation: %v, want %q", got, want)
	}
	if c.Gold.Idea != nil {
		t.Errorf("no invalidation: gold.idea must be null: %+v", c.Gold.Idea)
	}
}

// The reference level is nearest to PRICE, and the words must say that. With
// price already beyond the trigger, the nearest cluster on its far side can
// sit beyond the trigger too — "on the other side" would then be a false
// statement about the trigger, which is what this pins (review 2026-09-16).
func TestGoldIdeaReferenceLevelIsRelativeToPrice(t *testing.T) {
	in := goldFixture(trendUp) // day range 4293.00 – 4396.80
	in.px = 4400.1             // already above the trigger
	sup := SRLevel{Raw: 4398.0, Touches: 3}
	in.sup = &sup
	if sup.Raw <= in.levels.High {
		t.Fatalf("precondition: the support must sit ABOVE the trigger %v", in.levels.High)
	}
	c := goldCardFrom(in)
	line := goldIdeaBlock(c)[1]
	want := "Invalidated by a closed 1d candle below 4040.00; nearest level below price: support 4398.00"
	if line != want {
		t.Errorf("line = %q, want %q", line, want)
	}
	if strings.Contains(strings.Join(c.Facts, "\n"), "other side") {
		t.Errorf("the level is nearest to price, not opposite the trigger: %v", c.Facts)
	}
	if r := c.Gold.Idea.ReferenceLevel; r == nil || r.Level != 4398.0 || r.Kind != "support" {
		t.Errorf("reference_level: %+v", c.Gold.Idea.ReferenceLevel)
	}
	if c.Gold.Idea.State != goldIdeaTriggerHit {
		t.Errorf("price is past the trigger: %q", c.Gold.Idea.State)
	}
}

// The trigger branch compares RAW, exactly like the stage-1 scenario tail —
// only the invalidation branch rounds to the printed tick. A close 0.004 above
// the edge prints the same as the edge and is still beyond it.
func TestGoldIdeaTriggerComparesRaw(t *testing.T) {
	in := goldFixture(trendUp)
	in.px = 4396.804 // the day range high is 4396.80
	if goldPx(in.px) != goldPx(in.levels.High) {
		t.Fatalf("precondition: both must print alike, got %s / %s", goldPx(in.px), goldPx(in.levels.High))
	}
	c := goldCardFrom(in)
	if c.Gold.Idea.State != goldIdeaTriggerHit {
		t.Errorf("state = %q, want %q (raw comparison, like the scenario tail)", c.Gold.Idea.State, goldIdeaTriggerHit)
	}
	// Found by content, not by index: a confirmed card carries no leading
	// "Regime:" line, so its scenarios sit one row higher than a grey one's.
	up := ""
	for _, f := range c.Facts {
		if strings.Contains(f, "classifies the day as an upside break") {
			up = f
		}
	}
	if !strings.HasSuffix(up, "; the last 1h close is already above it") {
		t.Errorf("the stage-1 scenario tail must agree with the state: %q", up)
	}
}

// Nothing clustered on the reference side is said, never padded with the day
// range (a different thing measured a different way).
func TestGoldIdeaNoReferenceLevel(t *testing.T) {
	in := goldFixture(trendUp)
	in.sup = nil
	c := goldCardFrom(in)
	want := "Invalidated by a closed 1d candle below 4040.00; nothing clustered below price"
	if got := goldIdeaBlock(c); len(got) != 3 || got[1] != want {
		t.Errorf("no support: %v, want %q", got, want)
	}
	if c.Gold.Idea == nil || c.Gold.Idea.ReferenceLevel != nil {
		t.Errorf("reference_level must be null: %+v", c.Gold.Idea)
	}
}

// state mirrors what the card already says: armed until the last 1h close is
// beyond a level, then which one it is beyond.
func TestGoldIdeaState(t *testing.T) {
	cases := []struct {
		name  string
		state string
		px    float64
		want  string
	}{
		{"up armed", trendUp, 4354.9, "armed"},
		{"up trigger", trendUp, 4400.1, "trigger_reached"},
		{"up at the edge is not beyond", trendUp, 4396.8, "armed"},
		{"down armed", trendDown, 4354.9, "armed"},
		{"down trigger", trendDown, 4290.0, "trigger_reached"},
		{"down at the edge is not beyond", trendDown, 4293.0, "armed"},
	}
	for _, tc := range cases {
		in := goldFixture(tc.state)
		in.px = tc.px
		g := goldCardFrom(in).Gold
		if g.Idea == nil || g.Idea.State != tc.want {
			t.Errorf("%s: %+v, want state %q", tc.name, g.Idea, tc.want)
		}
	}

	// Price already beyond the invalidation level outranks the trigger, and
	// the comparison is the printed tick — the same one the card's own
	// invalidation line uses.
	inv := goldFixture(trendUp)
	lv := inv.trend.Levels.(TrendLevels)
	lvl := 4360.0
	lv.Invalidation = &lvl
	inv.trend.Levels = lv
	if g := goldCardFrom(inv).Gold; g.Idea == nil || g.Idea.State != "invalidation_reached" {
		t.Errorf("below the invalidation level: %+v", g.Idea)
	}
	lvl = 4354.904 // prints 4354.90, the same as the close: not beyond
	if g := goldCardFrom(inv).Gold; g.Idea == nil || g.Idea.State != "armed" {
		t.Errorf("equal at the printed tick is not beyond: %+v", g.Idea)
	}
}

// The machine form the site draws the card from — additive, inside gold.
func TestGoldIdeaMachineFields(t *testing.T) {
	raw, err := json.Marshal(cardEnvelope(goldCardFrom(goldFixture(trendUp))))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	g, ok := m["gold"].(map[string]any)
	if !ok {
		t.Fatalf("no gold object: %s", raw)
	}
	idea, ok := g["idea"].(map[string]any)
	if !ok {
		t.Fatalf("no gold.idea: %s", raw)
	}
	if idea["state"] != "armed" {
		t.Errorf("state: %v", idea["state"])
	}
	trig, _ := idea["trigger"].(map[string]any)
	if trig["level"] != 4396.8 || trig["side"] != "above" || trig["basis"] != "day_range_high" {
		t.Errorf("trigger: %v", trig)
	}
	invl, _ := idea["invalidation"].(map[string]any)
	if invl["level"] != 4040.0 || invl["side"] != "below" || invl["basis"] != "ema_cluster_atr" {
		t.Errorf("invalidation: %v", invl)
	}
	ref, _ := idea["reference_level"].(map[string]any)
	if ref["level"] != 4329.2 || ref["kind"] != "support" || ref["class"] != "single_swing" {
		t.Errorf("reference_level: %v", ref)
	}

	// Stage 1 fields keep their place and their values.
	for k, v := range map[string]any{"regime": "up", "confirmed": true, "price": 4354.9, "price_position": "inside"} {
		if g[k] != v {
			t.Errorf("gold.%s = %v, want %v", k, g[k], v)
		}
	}
	// Down mirrors: the trigger is the range low, the reference a resistance.
	down, _ := json.Marshal(cardEnvelope(goldCardFrom(goldFixture(trendDown))))
	var dm map[string]any
	if err := json.Unmarshal(down, &dm); err != nil {
		t.Fatal(err)
	}
	di, _ := dm["gold"].(map[string]any)["idea"].(map[string]any)
	dt, _ := di["trigger"].(map[string]any)
	dr, _ := di["reference_level"].(map[string]any)
	if dt["level"] != 4293.0 || dt["side"] != "below" || dt["basis"] != "day_range_low" {
		t.Errorf("down trigger: %v", dt)
	}
	if dr["level"] != 4364.5 || dr["kind"] != "resistance" || dr["class"] != "candidate" {
		t.Errorf("down reference_level: %v", dr)
	}
	// The key exists and is null when there is no idea — the site branches on
	// presence, never on a missing key.
	grey, _ := json.Marshal(cardEnvelope(goldCardFrom(goldFixture(trendGrey))))
	var gm map[string]any
	if err := json.Unmarshal(grey, &gm); err != nil {
		t.Fatal(err)
	}
	gi, ok := gm["gold"].(map[string]any)["idea"]
	if !ok || gi != nil {
		t.Errorf("unconfirmed: idea must be present and null, got %v (%v): %s", gi, ok, grey)
	}
}
