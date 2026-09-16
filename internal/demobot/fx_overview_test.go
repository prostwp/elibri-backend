package demobot

// fx_overview_test.go — FX stage 2 (2026-09-16): the card becomes a COMPARISON
// of the instruments in one place. The rule is UNCHANGED (windows, EMA50 vs
// EMA200, RSI(14), the freshness bounds, gold's 3h bound, the instrument set):
// what changes is the presentation — one row per instrument under a column
// header, a deterministic signed order, the bar age on a row whose bar is not
// fresh, gold still in its own COMEX section. No leader, no recommendation,
// no forecast: USD normalisation and a dollar index are stage 3 and are NOT
// introduced here.

import (
	"encoding/json"
	"regexp"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

// ── the table ────────────────────────────────────────────────────────────────

// The whole card as a reader sees it: two caption lines (columns, then the
// order and what it is NOT), the pair rows ordered by the size of the 24h
// change, then gold's own section. Same numbers as the stage-1 golden.
func TestFXTableGoldenLive(t *testing.T) {
	c := fxCardFromReads(liveFXReads(), fxNow)
	want := "⚪ <b>FX Agent</b>\n" +
		"<b>FX overview · 1h · 3 pairs + gold read</b>\n" +
		"• " + fxTableHeader + "\n" +
		"• " + fxOrderNote + "\n" +
		"• USDJPY · 154.88 · +0.32% · 91% · below · 62.9 · Sep 15 07:00\n" +
		"• GBPUSD · 1.3480 · -0.09% · 55% · below · 41.2 · Sep 15 08:00\n" +
		"• EURUSD · 1.1543 · -0.05% · 43% · below · 40.8 · Sep 15 08:00\n" +
		"• " + fxGoldHeader + "\n" +
		"• GOLD · 4320.7 · -0.69% · 38% · below · 40.6 · Sep 15 08:00\n" +
		"\n<i>Analytics, not financial advice · AlphaVizor · 2026-09-15 07:00 UTC · data: Yahoo Finance</i>"
	if got := c.RenderHTML(); got != want {
		t.Fatalf("fx table golden mismatch:\ngot:\n%s\nwant:\n%s", got, want)
	}
}

// The order is signed on the card and it says what it is not: no leader, no
// importance, no ranking claim.
func TestFXOrderIsSignedNotARanking(t *testing.T) {
	c := fxCardFromReads(liveFXReads(), fxNow)
	if !strings.Contains(fxOrderNote, "not by importance") {
		t.Errorf("the order note must disclaim importance: %q", fxOrderNote)
	}
	found := false
	for _, f := range c.Facts {
		found = found || f == fxOrderNote
	}
	if !found {
		t.Errorf("the card must carry the order note:\n%s", strings.Join(c.Facts, "\n"))
	}
	// One instrument alone is not an order: the note stays away.
	one := fxCardFromReads(liveFXReads()[:1], fxNow)
	for _, f := range one.Facts {
		if f == fxOrderNote {
			t.Errorf("single row must not claim an order:\n%s", strings.Join(one.Facts, "\n"))
		}
	}
	if one.Facts[0] != fxTableHeader {
		t.Errorf("single row still needs the columns: %q", one.Facts[0])
	}
}

// fxFirstRow is the card's first instrument row, captions skipped.
func fxFirstRow(c Card) string {
	for _, f := range c.Facts {
		if !fxCaptionLine(f) {
			return f
		}
	}
	return ""
}

// fxRowFor is the shown row of one label; "" when the card has none.
func fxRowFor(c Card, label string) string {
	for _, f := range c.Facts {
		if !fxCaptionLine(f) && strings.HasPrefix(f, label+" · ") {
			return f
		}
	}
	return ""
}

// fxResultFor is one instrument's results[] entry, found by its printed label.
func fxResultFor(c Card, label string) AssetResult {
	for _, r := range c.Results {
		if r.Label == label {
			return r
		}
	}
	return AssetResult{}
}

// fxRowLabels: the instrument rows of a card, in shown order, by their label.
func fxRowLabels(c Card) []string {
	var out []string
	for _, f := range c.Facts {
		if fxCaptionLine(f) {
			continue
		}
		if i := strings.Index(f, " · "); i > 0 {
			out = append(out, f[:i])
		}
	}
	return out
}

// The order is deterministic: by the SIZE of the 24h change, whatever order
// the concurrent sweep leaves the reads in.
func TestFXOrderDeterministic(t *testing.T) {
	live := liveFXReads()
	shuffles := [][]fxRead{
		{live[0], live[1], live[2], live[3]},
		{live[3], live[2], live[1], live[0]},
		{live[2], live[0], live[3], live[1]},
	}
	want := []string{"USDJPY", "GBPUSD", "EURUSD", "GOLD"}
	for i, reads := range shuffles {
		if got := fxRowLabels(fxCardFromReads(reads, fxNow)); strings.Join(got, ",") != strings.Join(want, ",") {
			t.Errorf("shuffle %d: rows %v, want %v", i, got, want)
		}
	}
	// A tie keeps the registry order, so two equal moves never swap places
	// between two reads of the same data.
	tie := []fxRead{
		fxOK("usdjpy", 154.88, "down", 62.9, 0.4, 0.9, fxAt(15, 8)),
		fxOK("eurusd", 1.1543, "down", 40.8, -0.4, 0.43, fxAt(15, 8)),
		fxOK("gbpusd", 1.348, "down", 41.2, 0.4, 0.55, fxAt(15, 8)),
	}
	if got := fxRowLabels(fxCardFromReads(tie, fxNow)); strings.Join(got, ",") != "EURUSD,GBPUSD,USDJPY" {
		t.Errorf("tie order %v, want the registry order", got)
	}
}

// A row without a fresh bar is shown honestly with the bar's age and never
// floats to the top, however big its move: the biggest mover here is the
// delayed one.
func TestFXStaleRowsCarryAgeAndStayBelow(t *testing.T) {
	reads := liveFXReads()
	reads[2].CloseAt = fxAt(15, 5) // USDJPY 3.5h old at 08:30, market open
	reads[3].CloseAt = fxAt(15, 5) // gold past its 3h bound
	c := fxCardFromReads(reads, fxNow)
	if got := fxRowLabels(c); strings.Join(got, ",") != "GBPUSD,EURUSD,USDJPY,GOLD" {
		t.Errorf("stale rows must sink: %v", got)
	}
	rows := map[string]string{}
	for _, f := range c.Facts {
		if fxCaptionLine(f) {
			continue
		}
		rows[f[:strings.Index(f, " · ")]] = f
	}
	if want := "USDJPY · 154.88 · +0.32% · 91% · below · 62.9 · Sep 15 05:00 · delayed, bar 3h old"; rows["USDJPY"] != want {
		t.Errorf("delayed row:\n%q\nwant:\n%q", rows["USDJPY"], want)
	}
	if want := "GOLD · 4320.7 · -0.69% · 38% · below · 40.6 · Sep 15 05:00 · no recent bar, 3h old"; rows["GOLD"] != want {
		t.Errorf("stale gold row:\n%q\nwant:\n%q", rows["GOLD"], want)
	}
	// A pair inside the weekend window is not "stale": the banner dates it,
	// so the weekend does not reshuffle the table.
	wk := []fxRead{
		fxOK("eurusd", 1.1731, "up", 55.1, 0.12, 0.66, fxAt(11, 22)),
		fxOK("usdjpy", 147.33, "down", 47.0, -0.2, 0.3, fxAt(11, 22)),
	}
	if got := fxRowLabels(fxCardFromReads(wk, fxWeekend)); strings.Join(got, ",") != "USDJPY,EURUSD" {
		t.Errorf("weekend rows %v, want the change order", got)
	}
}

// Part of the sweep dead: every instrument keeps a row, the ones without a
// reading sit at the bottom, and their wording is the stage-1 one.
func TestFXTablePartlyDead(t *testing.T) {
	reads := []fxRead{
		fxOK("eurusd", 1.1543, "down", 40.8, -0.05, 0.43, fxAt(15, 8)),
		{Pair: "GBPUSD", spec: assetTable["gbpusd"], Insufficient: true},
		{Pair: "USDJPY", spec: assetTable["usdjpy"]},
		fxOK("xauusd", 4320.7, "down", 40.6, -0.69, 0.38, fxAt(15, 8)),
	}
	c := fxCardFromReads(reads, fxNow)
	if got := fxRowLabels(c); strings.Join(got, ",") != "EURUSD,GBPUSD,USDJPY,GOLD" {
		t.Errorf("rows %v", got)
	}
	joined := strings.Join(c.Facts, "\n")
	for _, want := range []string{
		"GBPUSD · insufficient history for EMA50/EMA200/RSI(14) on 1h bars",
		"USDJPY · data unavailable right now",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q:\n%s", want, joined)
		}
	}
	if c.Verdict != "FX overview · 1h · 1 of 3 pairs + gold read · 1 short history · 1 unavailable" {
		t.Errorf("verdict %q", c.Verdict)
	}
}

// A row with no 24h reference and a degenerate range says so in its own
// columns instead of borrowing a neighbour's number; after a session gap the
// change names the close it is measured from and the card explains it once.
func TestFXRowMissingPiecesAndGap(t *testing.T) {
	r := fxOK("eurusd", 1.1, "flat", 50.0, 0, 0.5, fxAt(15, 8))
	r.HasDay, r.HasRange = false, false
	if got := fxTableRow(r, fxNow); got != "EURUSD · 1.1000 · no 24h reference · no range · equal · 50.0 · Sep 15 08:00" {
		t.Errorf("missing pieces row: %q", got)
	}
	gap := liveFXReads()
	gap[0].SinceClose, gap[0].RefAt = true, fxAt(11, 22)
	c := fxCardFromReads(gap, fxNow)
	joined := strings.Join(c.Facts, "\n")
	if !strings.Contains(joined, "EURUSD · 1.1543 · -0.05% since Sep 11 22:00 · 43% · below · 40.8 · Sep 15 08:00") {
		t.Errorf("gap row:\n%s", joined)
	}
	if !strings.Contains(joined, fxGapNote) {
		t.Errorf("a gap row must be explained once:\n%s", joined)
	}
	if strings.Contains(strings.Join(fxCardFromReads(liveFXReads(), fxNow).Facts, "\n"), fxGapNote) {
		t.Error("no gap row, no gap note")
	}
}

// Gold keeps its own section and its contract disclosure right above its row.
func TestFXGoldStaysItsOwnSection(t *testing.T) {
	c := fxCardFromReads(liveFXReads(), fxNow)
	lines := strings.Split(c.RenderHTML(), "\n")
	for i, l := range lines {
		if strings.HasPrefix(l, "• GOLD · ") {
			if i == 0 || !strings.Contains(lines[i-1], "COMEX GC=F futures, not spot XAUUSD") {
				t.Fatalf("the line above the gold row must name the contract:\n%s", strings.Join(lines, "\n"))
			}
			return
		}
	}
	t.Fatalf("no gold row:\n%s", strings.Join(lines, "\n"))
}

// ── no recommendation, no forecast ───────────────────────────────────────────

// fxForbiddenWords: words the card may never use — a recommendation, a
// ranking claim or a forecast. Whole words only ("short history" must stay
// legal), matched on the lower-cased line.
var fxForbiddenWords = regexp.MustCompile(`\b(buy|sell|enter|entry|target|forecast|predict|prediction|expect|expected|likely|probably|should|recommend|recommended|advice|leader|leading|best|strongest|weakest|winner|opportunity|outperform|underperform|bullish|bearish)\b`)

// Every line the sweep puts in front of a reader, on every fixture — the
// landing's three parts included: the rule is about what the reader is told,
// not about which function printed it.
func TestFXTextHasNoRecommendationOrForecast(t *testing.T) {
	for name, reads := range fxTextFixtures() {
		for _, now := range []time.Time{fxNow, fxWeekend} {
			for _, l := range fxAllTextLines(t, reads, now) {
				if m := fxForbiddenWords.FindString(strings.ToLower(l)); m != "" {
					t.Errorf("%s @%s: %q carries %q", name, now.Format("Mon 15:04"), l, m)
				}
			}
		}
	}
}

// fxStage3Claim reports a dollar-normalisation or dollar-strength statement —
// the stage-3 rule this card must not sneak in. Matched by MEANING, not by a
// fixed phrase: a literal substring list let "Dollar strength today: +0.31%"
// and "USD score 61/100" straight through.
//
// The rule is not a list of bad phrases but a property of this card: it names
// INSTRUMENTS (EURUSD, GBPUSD, USDJPY, GOLD), never a currency standing on its
// own. So a bare dollar word or a dollar index IS the claim, whatever follows
// it — and every shape nobody has thought of yet with it. An AND-gate against
// a list of aggregating words was the first attempt and it let "DXY 97.4",
// "Dollar up across the board +0.2%", "USD firmer on 2 of 3 pairs" and
// "Greenback +0.18% on the day" straight through: the aggregation lives in the
// sentence, not in a word you can enumerate.
//
// Pair labels are safe by word boundary: \busd\b matches neither USDJPY nor
// XAUUSD. Nothing this card says legitimately carries one of these words —
// the conclusion says "one read across them", not "a read on the dollar" —
// so stage 3 cannot arrive without this test being changed on purpose.
var fxDollarClaim = regexp.MustCompile(`\b(usd|dollars?|greenback|buck|dxy)\b`)

func fxStage3Claim(line string) bool {
	return fxDollarClaim.MatchString(strings.ToLower(line))
}

// The detector itself, first: these are the shapes stage 3 would arrive in,
// and the ones the card may keep saying.
func TestFXStage3ClaimDetector(t *testing.T) {
	for _, bad := range []string{
		"Dollar strength today: +0.31%",
		"USD basket +0.22% vs the three pairs",
		"Trade-weighted USD +0.18%",
		"Average move against the dollar +0.3%",
		"USD score 61/100",
		"DXY 97.4 · index up",
		"USD mixed: stronger against JPY, weaker elsewhere",
		"Normalised USD move +0.12%",
		// The index names itself: no second word to lean on.
		"DXY 97.4",
		"Dollar index 97.4",
		"USD index up 0.2%",
		// Aggregation without the arithmetic words.
		"Dollar up across the board +0.2%",
		"USD firmer on 2 of 3 pairs",
		"Greenback +0.18% on the day",
		"USD tilt: up",
		"Dollar up against all three",
	} {
		if !fxStage3Claim(bad) {
			t.Errorf("detector missed the stage-3 claim %q", bad)
		}
	}
	for _, ok := range []string{
		fxTableHeader, fxOrderNote, fxGapNote, fxGoldHeader, fxClosedBanner,
		"USDJPY · 154.88 · +0.32% · 91% · below · 62.9 · Sep 15 07:00",
		"USDCHF · 0.7912 · -0.08% · 44% · below · 47.1 · Sep 15 07:00",
		"EURUSD · insufficient history for EMA50/EMA200/RSI(14) on 1h bars",
		"GOLD (COMEX GC=F futures) · 4320.7 · -0.69% · 38% · below · 40.6 · Sep 15 08:00",
		"FX overview · 1h · 3 pairs + gold read · 1 data delayed",
		howTexts[keyFX], fxConclusion(),
	} {
		if fxStage3Claim(ok) {
			t.Errorf("detector flags a legitimate line: %q", ok)
		}
	}
}

// Stage 3 is not sneaking in anywhere the reader looks: no USD normalisation,
// no dollar-strength verdict, no dollar index — on every fixture, including
// the weekend and the degraded paths.
func TestFXHasNoDollarStrengthClaim(t *testing.T) {
	for name, reads := range fxTextFixtures() {
		for _, now := range []time.Time{fxNow, fxWeekend} {
			for _, l := range fxAllTextLines(t, reads, now) {
				if fxStage3Claim(l) {
					t.Errorf("%s @%s: %q is a stage-3 claim", name, now.Format("Mon 15:04"), l)
				}
			}
		}
	}
}

// fxTextFixtures are the sweeps every text rule is checked against.
func fxTextFixtures() map[string][]fxRead {
	short := func(k string) fxRead {
		return fxRead{Pair: assetTable[k].Display, spec: assetTable[k], Insufficient: true}
	}
	dead := func(k string) fxRead { return fxRead{Pair: assetTable[k].Display, spec: assetTable[k]} }
	stale := liveFXReads()
	stale[2].CloseAt, stale[3].CloseAt = fxAt(15, 5), fxAt(15, 5)
	gap := liveFXReads()
	gap[0].SinceClose, gap[0].RefAt = true, fxAt(11, 22)
	return map[string][]fxRead{
		"live":      liveFXReads(),
		"stale":     stale,
		"gap":       gap,
		"mixed":     {liveFXReads()[0], short("gbpusd"), dead("usdjpy"), liveFXReads()[3]},
		"all short": {short("eurusd"), short("xauusd")},
		"all dead":  {dead("eurusd"), dead("xauusd")},
	}
}

// fxAllTextLines is every line the sweep puts in front of a reader: the card,
// the how-text, the landing's three parts (data block, leading fact,
// conclusion) and the digest block. The conclusion belongs here because it is
// the one sentence on the landing that draws a conclusion at all.
func fxAllTextLines(t *testing.T, reads []fxRead, now time.Time) []string {
	t.Helper()
	c := fxCardFromReads(reads, now)
	lines := append([]string{c.Verdict, c.Short, howTexts[keyFX]}, c.Facts...)
	// The content blocks are text a reader sees too (the site's CURRENT READING
	// and SCOPE AND LIMITATIONS sections), so they face the same rules as the
	// rows: no recommendation, no forecast, no stage-3 dollar claim.
	if b := c.Blocks; b != nil {
		lines = append(lines, b.WhatHappened, b.WhyLevel, b.Regime, b.Limitations, b.Context, b.StateChanges, b.Source)
		lines = append(lines, b.Scenarios...)
		if b.Invalidates != nil {
			lines = append(lines, *b.Invalidates)
		}
	}
	lines = append(lines, exampleFacts(c)...)
	lines = append(lines, strongestFact(c), conclusionFor(c), detectedSentence(c))
	g := gathered{fx: reads, fxAnyOK: c.effectiveStatus() == statusOK, cards: map[string]Card{}}
	for _, s := range digestSections(g, "", now) {
		if s.key == keyFX {
			lines = append(append(lines, htmlToPlain(s.title)), s.lines...)
		}
	}
	return lines
}

// ── regression: everything that is not a table row is the stage-1 line ───────

// The card's non-row lines must be byte-identical to stage 1 on the same
// fixtures: the coverage header, the weekend banner, the gold disclosure and
// every degraded wording. Only the rows were redesigned.
func TestFXNonTableLinesUnchanged(t *testing.T) {
	short := func(k string) fxRead {
		return fxRead{Pair: assetTable[k].Display, spec: assetTable[k], Insufficient: true}
	}
	dead := func(k string) fxRead { return fxRead{Pair: assetTable[k].Display, spec: assetTable[k]} }
	delayed := func() []fxRead {
		r := liveFXReads()
		r[0].CloseAt = fxAt(15, 5)
		return r
	}
	staleGold := func() []fxRead {
		r := liveFXReads()
		r[3].CloseAt = fxAt(15, 5)
		return r
	}
	cases := []struct {
		name           string
		reads          []fxRead
		now            time.Time
		verdict, short string
	}{
		{"live", liveFXReads(), fxNow, "FX overview · 1h · 3 pairs + gold read", "3 pairs + gold read"},
		{"delayed", delayed(), fxNow, "FX overview · 1h · 3 pairs + gold read · 1 data delayed", "3 pairs + gold read · 1 data delayed"},
		{"stale gold", staleGold(), fxNow, "FX overview · 1h · 3 pairs read · gold: no recent bar", "3 pairs read · gold: no recent bar"},
		{"weekend", liveFXReads(), fxWeekend, "FX overview · 1h · 3 pairs + gold read", "3 pairs + gold read"},
		{"mixed degraded", []fxRead{short("eurusd"), short("gbpusd"), dead("usdjpy"), dead("xauusd")}, fxNow,
			"FX overview · 1h · 0 of 3 pairs read · 2 short history · 2 unavailable",
			"0 of 3 pairs read · 2 short history · 2 unavailable"},
		{"all short", []fxRead{short("eurusd"), short("xauusd")}, fxNow,
			"Insufficient history on 1h bars — no FX overview", "insufficient history"},
		{"all dead", []fxRead{dead("eurusd"), dead("xauusd")}, fxNow, fxOfflineVerdict, "offline"},
	}
	for _, tc := range cases {
		c := fxCardFromReads(tc.reads, tc.now)
		if c.Verdict != tc.verdict || c.Short != tc.short {
			t.Errorf("%s: verdict %q short %q", tc.name, c.Verdict, c.Short)
		}
		if c.Emoji != emojiNeutral {
			t.Errorf("%s: semaphore %q", tc.name, c.Emoji)
		}
	}
	// The weekend banner is still the first line, above the table.
	wk := fxCardFromReads(liveFXReads(), fxWeekend)
	if wk.Facts[0] != fxClosedBanner || wk.Facts[1] != fxTableHeader {
		t.Errorf("weekend facts: %q / %q", wk.Facts[0], wk.Facts[1])
	}
	// The degraded card (no reading anywhere) keeps the stage-1 lines verbatim.
	degraded := []fxRead{short("eurusd"), dead("usdjpy")}
	d := fxCardFromReads(degraded, fxNow)
	for i, r := range degraded {
		if want := fxMarketLine(r, fxNow); d.Facts[i] != want {
			t.Errorf("degraded fact %d: %q, want %q", i, d.Facts[i], want)
		}
	}
	// The gold disclosure is untouched and still opens gold's section.
	if fxGoldHeader != "Gold: COMEX GC=F futures, not spot XAUUSD; Forex hours and the weekend banner do not apply" {
		t.Errorf("gold disclosure changed: %q", fxGoldHeader)
	}
	// The AI payload keeps the stage-1 rows: the model is not part of this
	// redesign.
	g := gathered{fx: liveFXReads(), fxAnyOK: true, cards: map[string]Card{}, at: fxNow}
	var p struct {
		FX []string `json:"fx"`
	}
	if err := json.Unmarshal([]byte(aiPayload(g)), &p); err != nil {
		t.Fatal(err)
	}
	if len(p.FX) != 4 || p.FX[0] != "EURUSD 1.1543 · 24h -0.05% · 43% of the 24h range · EMA50 below EMA200 · RSI(1h) 40.8" {
		t.Errorf("AI rows changed: %v", p.FX)
	}
}

// ── machine fields ───────────────────────────────────────────────────────────

// results[] is additive: the shown order (row), the section a row belongs to
// and the label the row prints — everything the site needs to redraw the
// table. The stage-1 numbers keep their names and meaning.
func TestFXResultsCarryTheTable(t *testing.T) {
	env := cardEnvelope(fxCardFromReads(liveFXReads(), fxNow))
	raw, err := json.Marshal(env)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Results []map[string]any `json:"results"`
	}
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Results) != 4 {
		t.Fatalf("results %d", len(got.Results))
	}
	want := []struct {
		asset, label, section string
		row                   float64
	}{
		{"USDJPY", "USDJPY", "pairs", 1},
		{"GBPUSD", "GBPUSD", "pairs", 2},
		{"EURUSD", "EURUSD", "pairs", 3},
		{"GOLD · COMEX GC=F", "GOLD", "gold", 4},
	}
	for i, w := range want {
		r := got.Results[i]
		if r["asset"] != w.asset || r["label"] != w.label || r["section"] != w.section || r["row"] != w.row {
			t.Errorf("results[%d] = %v, want asset %s label %s section %s row %v", i, r, w.asset, w.label, w.section, w.row)
		}
		if r["price"] == nil || r["change_pct"] == nil || r["range_position_pct"] == nil || r["ema_relation"] == nil {
			t.Errorf("results[%d] lost a stage-1 number: %v", i, r)
		}
	}
	// A row without a reading still carries its place in the table.
	reads := liveFXReads()
	reads[1] = fxRead{Pair: "GBPUSD", spec: assetTable["gbpusd"], Insufficient: true}
	res := fxCardFromReads(reads, fxNow).Results
	last := res[len(res)-1]
	if last.Asset != "GOLD · COMEX GC=F" || last.Section != "gold" {
		t.Errorf("gold stays last: %+v", last)
	}
	var gbp AssetResult
	for _, r := range res {
		if r.Asset == "GBPUSD" {
			gbp = r
		}
	}
	if gbp.OK || gbp.Row == 0 || gbp.Section != "pairs" || gbp.Label != "GBPUSD" {
		t.Errorf("unread row: %+v", gbp)
	}
	// results[] follows the shown rows one-to-one.
	c := fxCardFromReads(reads, fxNow)
	labels := fxRowLabels(c)
	if len(labels) != len(c.Results) {
		t.Fatalf("rows %v vs results %d", labels, len(c.Results))
	}
	for i, l := range labels {
		if c.Results[i].Label != l || c.Results[i].Row != i+1 {
			t.Errorf("row %d: %q vs %+v", i+1, l, c.Results[i])
		}
	}
}

// ── budgets ──────────────────────────────────────────────────────────────────

// Every table line fits the 110-rune budget, worst case included: the longest
// label, a four-figure price, a since-close change, a full range, a three-digit
// RSI and the longest staleness flag at once.
func TestFXTableLineBudget(t *testing.T) {
	worst := func(key string, price float64, gold bool) string {
		r := fxOK(key, price, "flat", 100, -12.34, 1, fxAt(11, 22))
		r.SinceClose, r.RefAt = true, fxAt(4, 22)
		now := r.CloseAt.Add(150 * 24 * time.Hour)
		for !isForexOpen(now) {
			now = now.Add(time.Hour)
		}
		line := fxTableRow(r, now)
		if gold != strings.Contains(line, "no recent bar") {
			t.Errorf("%s: staleness flag missing: %q", key, line)
		}
		return line
	}
	for _, l := range []string{
		fxTableHeader, fxOrderNote, fxGapNote,
		worst("usdjpy", 12345.67, false),
		worst("xauusd", 12345.67, true),
	} {
		if n := utf8.RuneCountInString(l); n > fxLineMaxRunes {
			t.Errorf("%d runes: %q", n, l)
		}
	}
}

// ── digest ───────────────────────────────────────────────────────────────────

// The digest block shows the same table: the column header and the card's own
// rows, verbatim and in card order. No separate "newer bars" line is needed —
// every row carries its bar.
func TestFXDigestShowsTheSameTable(t *testing.T) {
	reads := liveFXReads()
	reads[1] = fxRead{Pair: "GBPUSD", spec: assetTable["gbpusd"]}
	g := gathered{fx: reads, fxAnyOK: true, cards: map[string]Card{}}
	fx := fxDigestBlock(t, g, fxNow)
	card := fxCardFromReads(reads, fxNow)
	var want []string
	for _, f := range card.Facts {
		if f == fxClosedBanner || f == fxGoldHeader {
			continue
		}
		want = append(want, f)
	}
	if want[0] != fxTableHeader || want[1] != fxOrderNote {
		t.Fatalf("the digest must carry the same captions: %q / %q", want[0], want[1])
	}
	if strings.Join(fx.lines, "\n") != strings.Join(want, "\n") {
		t.Errorf("digest lines:\n%s\ncard table:\n%s", strings.Join(fx.lines, "\n"), strings.Join(want, "\n"))
	}
	for _, l := range fx.lines {
		if strings.HasPrefix(l, "Newer bars:") {
			t.Errorf("rows carry their own bar now: %q", l)
		}
	}
	if !fx.asOf.Equal(fxAt(15, 7)) || !fx.asOf.Equal(card.DataTime) {
		t.Errorf("as_of %v, want the oldest bar %v", fx.asOf, card.DataTime)
	}
	if got := htmlToPlain(fx.title); got != "FX (as of Sep 15 07:00 UTC · gold = COMEX GC=F futures)" {
		t.Errorf("title %q", got)
	}
	w := fxDigestBlock(t, g, fxWeekend)
	if got := htmlToPlain(w.title); !strings.HasPrefix(got, "FX (forex closed: pairs show Friday data · as of ") {
		t.Errorf("weekend title %q", got)
	}
}

// ── the landing's data block ─────────────────────────────────────────────────

func fxHasLine(lines []string, want string) bool {
	for _, l := range lines {
		if l == want {
			return true
		}
	}
	return false
}

// The landing quotes a PREFIX of the card's table, so it must keep every line
// that qualifies the values it quotes: the weekend banner (without it the
// reader takes Friday prices for today's), gold's COMEX disclosure (without it
// a futures price reads as spot XAUUSD), the column header (without it
// `91% · below · 62.9` reaches the reader unlabelled) and the order note
// (without it a list cut at the top of a sort reads as a ranking). Only the
// gap note is dropped: the row it explains names its own reference close.
func TestFXShowcaseDataKeepsWhatQualifiesTheValues(t *testing.T) {
	wk := fxCardFromReads(liveFXReads(), fxWeekend)
	data := exampleFacts(wk)
	if len(data) == 0 || data[0] != fxClosedBanner {
		t.Fatalf("a weekend data block must open with the banner: %v", data)
	}
	if len(data) > showcaseFactsMaxFX {
		t.Errorf("data block %d lines, cap %d: %v", len(data), showcaseFactsMaxFX, data)
	}
	if !fxHasLine(data, fxTableHeader) {
		t.Errorf("values without their column header: %v", data)
	}
	for _, f := range data {
		if f == fxGapNote {
			t.Errorf("the gap note is a service line: %q in %v", f, data)
		}
		if fxUnreadLine(f) {
			t.Errorf("an unread row on the landing: %q in %v", f, data)
		}
	}
	// The order note is not filtered at all: whenever the card signs its
	// order, the block quoting a cut of that order carries the signature.
	// Otherwise a list cut at the top of a sort reads as a ranking.
	if fxHasLine(wk.Facts, fxOrderNote) != fxHasLine(data, fxOrderNote) {
		t.Errorf("order note on the card %v, in the block %v: %v",
			fxHasLine(wk.Facts, fxOrderNote), fxHasLine(data, fxOrderNote), data)
	}
	// The card itself keeps every caption — the landing filter must not reach
	// /agents/fx.
	if !fxHasLine(wk.Facts, fxOrderNote) || !fxHasLine(wk.Facts, fxTableHeader) {
		t.Errorf("the card lost a caption: %v", wk.Facts)
	}

	// Gold reaching the block brings its disclosure, immediately above it.
	reads := liveFXReads()
	for i := 0; i < 3; i++ {
		reads[i] = fxRead{Pair: reads[i].Pair, spec: reads[i].spec}
	}
	gold := exampleFacts(fxCardFromReads(reads, fxNow))
	gi, ri := -1, -1
	for i, f := range gold {
		if f == fxGoldHeader {
			gi = i
		}
		if strings.HasPrefix(f, "GOLD · 4320.7") {
			ri = i
		}
	}
	if ri < 0 {
		t.Fatalf("no gold row on the landing: %v", gold)
	}
	if gi != ri-1 {
		t.Errorf("the COMEX disclosure must sit right above the gold row: %v", gold)
	}
	// A disclosure is never the last line: it would disclose nothing.
	if gold[len(gold)-1] == fxGoldHeader {
		t.Errorf("trailing disclosure with no row under it: %v", gold)
	}

	// The ordinary live block — what a reader actually meets: the header, the
	// order note, and the rows that fit under them.
	live := exampleFacts(fxCardFromReads(liveFXReads(), fxNow))
	if live[0] != fxTableHeader || live[1] != fxOrderNote {
		t.Errorf("a live block opens with the header and the order note: %v", live)
	}
	// Two captions must not cost the comparison: all three pairs still show.
	for _, want := range []string{"USDJPY · ", "GBPUSD · ", "EURUSD · "} {
		found := false
		for _, f := range live {
			found = found || strings.HasPrefix(f, want)
		}
		if !found {
			t.Errorf("captions ate the %s row: %v", want, live)
		}
	}

	// And the disclosure stays REACHABLE beside live pairs, not only when
	// every pair is down: at cap 4 the two captions made that impossible.
	one := liveFXReads()
	one[1] = fxRead{Pair: one[1].Pair, spec: one[1].spec}
	one[2] = fxRead{Pair: one[2].Pair, spec: one[2].spec}
	mixed := exampleFacts(fxCardFromReads(one, fxNow))
	if !fxHasLine(mixed, fxGoldHeader) {
		t.Errorf("no COMEX disclosure beside a live pair: %v", mixed)
	}
}

// The example's leading fact stands alone, above the data block, so when it is
// the gold row it carries the contract in it: quoted bare, "GOLD · 4320.7 · …"
// reads as spot XAUUSD — the one thing this card may never imply.
func TestFXShowcaseLeadingGoldRowNamesTheContract(t *testing.T) {
	reads := liveFXReads()
	for i := 0; i < 3; i++ { // only gold answers, so gold leads
		reads[i] = fxRead{Pair: reads[i].Pair, spec: reads[i].spec}
	}
	got := strongestFact(fxCardFromReads(reads, fxNow))
	if !strings.HasPrefix(got, "GOLD (COMEX GC=F futures) · ") {
		t.Errorf("leading fact %q does not name the contract", got)
	}
	if strings.Contains(got, "XAUUSD") {
		t.Errorf("leading fact names spot: %q", got)
	}
	// A pair row is quoted as the card prints it.
	pair := strongestFact(fxCardFromReads(liveFXReads(), fxNow))
	if !strings.HasPrefix(pair, "USDJPY · ") {
		t.Errorf("a pair row must be quoted verbatim: %q", pair)
	}

	// The contract costs runes. The longest gold row carries BOTH qualifiers
	// at once — measured from the close before a gap AND behind on its bar.
	reads[3].CloseAt = fxAt(15, 4)
	reads[3].SinceClose, reads[3].RefAt = true, fxAt(11, 22)
	worst := strongestFact(fxCardFromReads(reads, fxNow))
	for _, want := range []string{"since Sep 11 22:00", "no recent bar"} {
		if !strings.Contains(worst, want) {
			t.Fatalf("fixture lacks %q: %q", want, worst)
		}
	}
	if n := utf8.RuneCountInString(worst); n > fxQuotedMaxRunes {
		t.Errorf("leading fact %d runes, cap %d: %q", n, fxQuotedMaxRunes, worst)
	}
	// The card's own row stays inside the table budget: the contract is added
	// on the way out, never printed into the table.
	row := fxTableRow(reads[3], fxNow)
	if n := utf8.RuneCountInString(row); n > fxLineMaxRunes {
		t.Errorf("card row %d runes, cap %d: %q", n, fxLineMaxRunes, row)
	}
}

// The quoted-line budget is a sum, so its parts have to hold: the contract is
// ASCII (fxQuotedMaxRunes adds its BYTE length to a rune budget), and the
// worst line the rule permits — a row already at the card's budget, quoted
// with the contract and closed with a full stop — lands exactly on the cap,
// never past it. A pair row costs only the full stop.
func TestFXQuotedLineBudgetIsExact(t *testing.T) {
	if got, want := utf8.RuneCountInString(fxGoldContract), len(fxGoldContract); got != want {
		t.Fatalf("fxGoldContract is not ASCII: %d runes, %d bytes", got, want)
	}
	pad := func(label string) string {
		row := label + " · 9999.9 · -99.99% since Dec 31 23:59 · no range · below · 100.0 · Jun 14 23:30"
		for utf8.RuneCountInString(row) < fxLineMaxRunes {
			row += "x"
		}
		return row
	}
	gold := endSentence(fxNamedRow(pad("GOLD")))
	if n := utf8.RuneCountInString(gold); n != fxQuotedMaxRunes {
		t.Errorf("worst permitted gold quote %d runes, cap %d: %q", n, fxQuotedMaxRunes, gold)
	}
	pair := endSentence(fxNamedRow(pad("GBPUSD")))
	if n := utf8.RuneCountInString(pair); n != fxLineMaxRunes+1 {
		t.Errorf("worst permitted pair quote %d runes, want %d: %q", n, fxLineMaxRunes+1, pair)
	}
}

// The landing's conclusion is the only sentence there that concludes
// anything. The generic one read the neutral semaphore as a combined verdict
// over the pairs ("nothing leans either way on the broader market") and sent
// the reader to a "level structure" this card has no levels for.
func TestFXShowcaseConclusionMakesNoCombinedVerdict(t *testing.T) {
	for name, reads := range fxTextFixtures() {
		for _, now := range []time.Time{fxNow, fxWeekend} {
			got := conclusionFor(fxCardFromReads(reads, now))
			if got != fxConclusion() {
				t.Errorf("%s @%s: conclusion %q", name, now.Format("Mon 15:04"), got)
			}
			for _, bad := range []string{"leans either way", "level structure", "broader market"} {
				if strings.Contains(got, bad) {
					t.Errorf("%s: conclusion carries %q: %s", name, bad, got)
				}
			}
		}
	}
}

// ── the degraded card serves no table coordinates ────────────────────────────

// row/section/label describe a place in the shown table. A card that renders
// no table (no instrument produced a reading) has no such place, so it serves
// none of the three — the fields are not "0 / empty", they are absent.
func TestFXDegradedResultsCarryNoTableFields(t *testing.T) {
	short := func(k string) fxRead {
		return fxRead{Pair: assetTable[k].Display, spec: assetTable[k], Insufficient: true}
	}
	dead := func(k string) fxRead { return fxRead{Pair: assetTable[k].Display, spec: assetTable[k]} }
	for name, reads := range map[string][]fxRead{
		"mixed degraded": {short("eurusd"), dead("usdjpy")},
		"all short":      {short("eurusd"), short("xauusd")},
		"all dead":       {dead("eurusd"), dead("xauusd")},
	} {
		c := fxCardFromReads(reads, fxNow)
		if len(c.Results) != len(reads) {
			t.Fatalf("%s: results %d, want one per instrument", name, len(c.Results))
		}
		for _, r := range c.Results {
			if r.Row != 0 || r.Section != "" || r.Label != "" {
				t.Errorf("%s: no table, yet the row carries coordinates: %+v", name, r)
			}
		}
		raw, err := json.Marshal(cardEnvelope(c))
		if err != nil {
			t.Fatal(err)
		}
		for _, field := range []string{`"row"`, `"section"`, `"label"`} {
			if strings.Contains(string(raw), field) {
				t.Errorf("%s: %s served on a card with no table: %s", name, field, raw)
			}
		}
	}
	// A card WITH a table keeps them on every row, an unread one included.
	reads := liveFXReads()
	reads[1] = fxRead{Pair: "GBPUSD", spec: assetTable["gbpusd"], Insufficient: true}
	for _, r := range fxCardFromReads(reads, fxNow).Results {
		if r.Row == 0 || r.Section == "" || r.Label == "" {
			t.Errorf("table row lost its coordinates: %+v", r)
		}
	}
}
