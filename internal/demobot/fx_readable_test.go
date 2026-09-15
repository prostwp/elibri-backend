package demobot

// fx_readable_test.go — FX stage 1 (2026-09-15): an honest, readable /fx card
// with the rule UNCHANGED (EMA50 vs EMA200 on 1h, RSI(14), change vs the bar
// ~24h back, place in the trailing-24h range). Goldens with live-shaped
// numbers, the oldest-bar data time, per-row freshness (pairs by the Forex
// window, gold by bar age only), the 24h vs since-previous-close window,
// short history kept apart from "no data", results[] JSON, the digest block
// and the 110-character line budget on every path.

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// Tuesday 2026-09-15 08:30 UTC — Forex week open.
var fxNow = time.Date(2026, 9, 15, 8, 30, 0, 0, time.UTC)

// Saturday 2026-09-12 12:00 UTC — inside the fixed weekend window.
var fxWeekend = time.Date(2026, 9, 12, 12, 0, 0, 0, time.UTC)

func fxAt(day, hour int) time.Time { return time.Date(2026, 9, day, hour, 0, 0, 0, time.UTC) }

// fxOK is one live read of a registry asset, as fxReadFromCandles leaves it.
func fxOK(key string, price float64, dir string, rsi, chg, pos float64, closeAt time.Time) fxRead {
	spec := assetTable[key]
	return fxRead{
		Pair: spec.Display, spec: spec, OK: true, Dir: dir, RSI: rsi, Price: price,
		DayChangePct: chg, HasDay: true, RefAt: closeAt.Add(-24 * time.Hour),
		DayPos: pos, HasRange: true, CloseAt: closeAt,
	}
}

// liveFXReads mirrors the live /fx of 2026-09-15 ~09:00 UTC (EURUSD -0.05%,
// USDJPY +0.32% near the day high while EMA50 < EMA200 — the row the old card
// painted red). USDJPY's bar is one hour older than the rest.
func liveFXReads() []fxRead {
	return []fxRead{
		fxOK("eurusd", 1.154334545, "down", 40.8, -0.05, 0.43, fxAt(15, 8)),
		fxOK("gbpusd", 1.347981333, "down", 41.2, -0.09, 0.55, fxAt(15, 8)),
		fxOK("usdjpy", 154.884002, "down", 62.9, 0.32, 0.91, fxAt(15, 7)),
		fxOK("xauusd", 4320.7001953, "down", 40.6, -0.69, 0.38, fxAt(15, 8)),
	}
}

func TestFXCardGoldenLive(t *testing.T) {
	c := fxCardFromReads(liveFXReads(), fxNow)
	want := "⚪ <b>FX Agent</b>\n" +
		"<b>FX overview · 1h · 3 pairs + gold read</b>\n" +
		"• EURUSD 1.1543 · 24h -0.05% · 43% of the 24h range\n" +
		"• EURUSD: EMA50 below EMA200 · RSI(1h) 40.8 · last bar Sep 15 08:00 UTC\n" +
		"• GBPUSD 1.3480 · 24h -0.09% · 55% of the 24h range\n" +
		"• GBPUSD: EMA50 below EMA200 · RSI(1h) 41.2 · last bar Sep 15 08:00 UTC\n" +
		"• USDJPY 154.88 · 24h +0.32% · 91% of the 24h range\n" +
		"• USDJPY: EMA50 below EMA200 · RSI(1h) 62.9 · last bar Sep 15 07:00 UTC\n" +
		"• " + fxGoldHeader + "\n" +
		"• GOLD 4320.7 · 24h -0.69% · 38% of the 24h range\n" +
		"• GOLD: EMA50 below EMA200 · RSI(1h) 40.6 · last bar Sep 15 08:00 UTC\n" +
		"\n<i>Analytics, not financial advice · AlphaVizor · 2026-09-15 07:00 UTC · data: Yahoo Finance</i>"
	if got := c.RenderHTML(); got != want {
		t.Fatalf("fx live golden mismatch:\ngot:\n%s\nwant:\n%s", got, want)
	}
	if c.Short != "3 pairs + gold read" {
		t.Errorf("short %q", c.Short)
	}
}

// Item 1: the card's data time is the OLDEST bar shown — one fresh pair must
// not hide a lagging one.
func TestFXDataTimeIsOldestShownBar(t *testing.T) {
	reads := liveFXReads()
	reads[1].CloseAt = fxAt(15, 5) // GBPUSD three hours behind
	reads[1].RefAt = reads[1].CloseAt.Add(-24 * time.Hour)
	c := fxCardFromReads(reads, fxNow)
	if !c.DataTime.Equal(fxAt(15, 5)) {
		t.Errorf("DataTime %v, want the oldest shown bar %v", c.DataTime, fxAt(15, 5))
	}
	// A dead row carries no bar and cannot move the stamp.
	reads = liveFXReads()
	reads[0] = fxRead{Pair: "EURUSD", spec: assetTable["eurusd"]}
	if c := fxCardFromReads(reads, fxNow); !c.DataTime.Equal(fxAt(15, 7)) {
		t.Errorf("DataTime %v with a dead row, want %v", c.DataTime, fxAt(15, 7))
	}
}

// Item 2: a pair whose last bar is older than two 1h bars while the market
// has been open reads "data delayed" on its line, in the header and in JSON.
func TestFXPairDataDelayed(t *testing.T) {
	reads := liveFXReads()
	reads[0].CloseAt = fxAt(15, 5) // 3.5h old at 08:30, market open since Sunday
	c := fxCardFromReads(reads, fxNow)
	if c.Facts[0] != "EURUSD 1.1543 · 24h -0.05% · 43% of the 24h range · data delayed" {
		t.Errorf("delayed line: %q", c.Facts[0])
	}
	if c.Verdict != "FX overview · 1h · 3 pairs + gold read · 1 data delayed" {
		t.Errorf("verdict %q", c.Verdict)
	}
	if c.Results[0].Freshness != "data_delayed" || c.Results[1].Freshness != "on_time" {
		t.Errorf("freshness %q / %q", c.Results[0].Freshness, c.Results[1].Freshness)
	}
	// Right after the Sunday reopen a missing bar is not yet due.
	sun := time.Date(2026, 9, 13, 22, 30, 0, 0, time.UTC)
	if f := fxFreshness(reads[0], sun); f != "on_time" {
		t.Errorf("Sunday 22:30 reopen grace: %q", f)
	}
}

// Item 4: gold is its own section, named as futures, and never told the
// Forex market is closed — on a weekend its line states only what the bar
// age shows.
func TestFXGoldSectionWeekend(t *testing.T) {
	reads := []fxRead{
		fxOK("eurusd", 1.1731, "up", 55.1, 0.12, 0.66, fxAt(11, 22)),
		fxOK("gbpusd", 1.3561, "up", 52.0, 0.08, 0.5, fxAt(11, 22)),
		fxOK("usdjpy", 147.33, "down", 47.0, -0.2, 0.3, fxAt(11, 22)),
		fxOK("xauusd", 3650.4, "up", 61.0, 0.9, 0.95, fxAt(11, 21)),
	}
	c := fxCardFromReads(reads, fxWeekend)
	if c.Facts[0] != fxClosedBanner {
		t.Fatalf("weekend: banner first, got %q", c.Facts[0])
	}
	gi := -1
	for i, f := range c.Facts {
		if f == fxGoldHeader {
			gi = i
		}
	}
	if gi < 0 || gi != len(c.Facts)-3 {
		t.Fatalf("gold header must open the last section:\n%s", strings.Join(c.Facts, "\n"))
	}
	for _, f := range c.Facts[gi:] {
		if strings.Contains(strings.ToLower(f), "market closed") {
			t.Errorf("gold section claims a closed market: %q", f)
		}
	}
	// What the reader sees: the rendered line right above the gold row names
	// the contract as futures, not spot.
	rendered := strings.Split(c.RenderHTML(), "\n")
	named := false
	for i, l := range rendered {
		if strings.HasPrefix(l, "• GOLD 3650.4 ") && i > 0 {
			named = strings.Contains(rendered[i-1], "COMEX GC=F futures, not spot XAUUSD")
		}
	}
	if !named {
		t.Errorf("the line above the gold row must name the contract:\n%s", strings.Join(rendered, "\n"))
	}
	if c.Facts[gi+1] != "GOLD 3650.4 · 24h +0.90% · 95% of the 24h range · no bar in the last 3h" {
		t.Errorf("gold weekend line: %q", c.Facts[gi+1])
	}
	byAsset := map[string]AssetResult{}
	for _, r := range c.Results {
		byAsset[r.Asset] = r
	}
	if f := byAsset["GOLD · COMEX GC=F"].Freshness; f != "no_recent_bar" {
		t.Errorf("gold freshness %q, want no_recent_bar (never market_closed)", f)
	}
	if f := byAsset["EURUSD"].Freshness; f != "market_closed" {
		t.Errorf("pair weekend freshness %q", f)
	}
	// Gold with a fresh bar is on time whatever the Forex clock says.
	g := reads[3]
	g.CloseAt = fxWeekend.Add(-30 * time.Minute)
	if f := fxFreshness(g, fxWeekend); f != "on_time" {
		t.Errorf("fresh gold on a weekend: %q", f)
	}
	// The data time is the oldest bar — gold's Friday 21:00 here.
	if !c.DataTime.Equal(fxAt(11, 21)) {
		t.Errorf("weekend DataTime %v", c.DataTime)
	}
}

// Item 5: short history is its own state — never "no data" in the header —
// and a card with nothing but short history says so (insufficient_history).
func TestFXInsufficientHistoryNotNoData(t *testing.T) {
	reads := liveFXReads()
	reads[2] = fxRead{Pair: "USDJPY", spec: assetTable["usdjpy"], Insufficient: true}
	reads[3] = fxRead{Pair: "GOLD · COMEX GC=F", spec: assetTable["xauusd"]}
	c := fxCardFromReads(reads, fxNow)
	if c.Verdict != "FX overview · 1h · 2 of 3 pairs read · 1 short history · 1 unavailable" {
		t.Errorf("verdict %q", c.Verdict)
	}
	if strings.Contains(c.Verdict, "no data") {
		t.Errorf("short history counted as no data: %q", c.Verdict)
	}
	joined := strings.Join(c.Facts, "\n")
	for _, want := range []string{
		"USDJPY: insufficient history for EMA50/EMA200/RSI(14) on 1h bars",
		"GOLD: data unavailable right now",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q:\n%s", want, joined)
		}
	}
	if c.effectiveStatus() != statusOK {
		t.Errorf("a card with live pairs is ok, got %v", c.effectiveStatus())
	}

	all := []fxRead{
		{Pair: "EURUSD", spec: assetTable["eurusd"], Insufficient: true},
		{Pair: "GBPUSD", spec: assetTable["gbpusd"], Insufficient: true},
	}
	d := fxCardFromReads(all, fxNow)
	if d.effectiveStatus() != statusInsufficientHistory {
		t.Errorf("all short history: status %v, want insufficient_history", d.effectiveStatus())
	}
	if strings.Contains(d.Verdict, "unavailable") {
		t.Errorf("all short history must not read as a dead source: %q", d.Verdict)
	}
	dead := fxCardFromReads([]fxRead{{Pair: "EURUSD", spec: assetTable["eurusd"]}}, fxNow)
	if dead.effectiveStatus() != statusSourceOffline || dead.Verdict != fxOfflineVerdict {
		t.Errorf("dead source: %v %q", dead.effectiveStatus(), dead.Verdict)
	}
}

// Items 7 and 9: no "EMA trend" claim, no coloured row, no "N down" counter,
// a neutral semaphore — the card has no combined verdict.
func TestFXNoCombinedVerdictOrRowColour(t *testing.T) {
	c := fxCardFromReads(liveFXReads(), fxNow)
	if c.Emoji != emojiNeutral {
		t.Errorf("semaphore %q, want neutral", c.Emoji)
	}
	for _, f := range append([]string{c.Verdict, c.Short}, c.Facts...) {
		for _, bad := range []string{"EMA trend", "mid-range", "near day", "down ·", " down", " up", emojiBull, emojiBear} {
			if strings.Contains(f, bad) {
				t.Errorf("%q carries %q", f, bad)
			}
		}
	}
	for _, dir := range []struct{ d, want string }{{"up", "above"}, {"down", "below"}, {"flat", "equal to"}} {
		r := fxOK("eurusd", 1.1, dir.d, 50, 0, 0.5, fxAt(15, 8))
		if got := fxContextLine(r); !strings.Contains(got, "EMA50 "+dir.want+" EMA200") {
			t.Errorf("%s: %q", dir.d, got)
		}
	}
}

// Item 3: "24h" only when the reference close is 24–26h back; after a
// session gap the change is labelled with the close it is measured from.
func TestFXChangeWindow(t *testing.T) {
	spec := assetTable["eurusd"]
	last := fxAt(15, 7) // bar open Tue 07:00, closes 08:00

	// Continuous series: the bar exactly 24h back.
	r := fxReadFromCandles(spec, fxSeries(last, 300, nil))
	if !r.OK || !r.HasDay || r.SinceClose || !r.RefAt.Equal(fxAt(14, 8)) {
		t.Fatalf("continuous: ok=%v hasDay=%v since=%v ref=%v", r.OK, r.HasDay, r.SinceClose, r.RefAt)
	}
	if !strings.Contains(fxMarketLine(r, fxNow), " · 24h ") {
		t.Errorf("continuous line: %q", fxMarketLine(r, fxNow))
	}

	// One missing bar at T-24h: the reference is 25h back — inside the
	// tolerance, still "24h".
	r = fxReadFromCandles(spec, fxSeries(last, 300, func(t int64) bool { return t == last.Unix()-86400 }))
	if !r.HasDay || r.SinceClose || !r.RefAt.Equal(fxAt(14, 7)) {
		t.Errorf("25h: since=%v ref=%v", r.SinceClose, r.RefAt)
	}

	// Three missing bars: 27h back — outside the tolerance.
	r = fxReadFromCandles(spec, fxSeries(last, 300, func(t int64) bool {
		d := last.Unix() - t
		return d >= 86400 && d <= 86400+7200
	}))
	if !r.HasDay || !r.SinceClose || !r.RefAt.Equal(fxAt(14, 5)) {
		t.Errorf("27h: since=%v ref=%v", r.SinceClose, r.RefAt)
	}

	// Monday after the weekend: Friday's last bar (open 21:00, close 22:00)
	// is the reference — "since Sep 11 22:00 UTC close", and the range is the
	// bars since then.
	mon := fxAt(14, 7)
	weekend := func(t int64) bool {
		return t > fxAt(11, 21).Unix() && t < time.Date(2026, 9, 13, 23, 0, 0, 0, time.UTC).Unix()
	}
	candles := fxSeries(mon, 300, weekend)
	r = fxReadFromCandles(spec, candles)
	if !r.HasDay || !r.SinceClose || !r.RefAt.Equal(fxAt(11, 22)) {
		t.Fatalf("weekend: since=%v ref=%v", r.SinceClose, r.RefAt)
	}
	ref := candles[0]
	for _, c := range candles {
		if c.Time == fxAt(11, 21).Unix() {
			ref = c
		}
	}
	lastC := candles[len(candles)-1]
	if want := (lastC.Close - ref.Close) / ref.Close * 100; r.DayChangePct != want {
		t.Errorf("weekend change %v, want %v (vs Friday's last close)", r.DayChangePct, want)
	}
	line := fxMarketLine(r, fxAt(14, 8))
	if !strings.Contains(line, " · since Sep 11 22:00 UTC close ") || !strings.Contains(line, "% of the range since then") {
		t.Errorf("weekend line: %q", line)
	}
	if strings.Contains(line, "24h") {
		t.Errorf("a Friday reference must not be called 24h: %q", line)
	}
	// The raw read is the same calculation as before: last close, the
	// 1h EMA cross and the trailing-24h range position.
	if r.Price != lastC.Close || !r.CloseAt.Equal(fxAt(14, 8)) {
		t.Errorf("price %v closeAt %v", r.Price, r.CloseAt)
	}
	if pos, ok := dayRange(candles); !ok || pos != r.DayPos {
		t.Errorf("range pos %v, want dayRange %v", r.DayPos, pos)
	}

	// Too little history: explicit, never a flat read.
	short := fxReadFromCandles(spec, fxSeries(last, 120, nil))
	if short.OK || !short.Insufficient {
		t.Errorf("120 bars: ok=%v insufficient=%v", short.OK, short.Insufficient)
	}
}

// fxSeries builds n hourly bars ending at the bar that opens at last,
// skipping the open times drop reports (session gaps, missing bars). Closes
// oscillate so RSI and the range are defined.
func fxSeries(last time.Time, n int, drop func(int64) bool) []types.OHLCVCandle {
	var out []types.OHLCVCandle
	for ts := last.Unix(); len(out) < n; ts -= 3600 {
		if drop != nil && drop(ts) {
			continue
		}
		i := float64(len(out))
		c := 1.10 + 0.002*float64(int(i)%7) - 0.00001*i
		out = append(out, types.OHLCVCandle{Time: ts, Open: c, High: c + 0.001, Low: c - 0.001, Close: c})
	}
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

// results[]: the numbers behind every row, additive to {asset, ok, reason}.
func TestFXResultsJSON(t *testing.T) {
	reads := liveFXReads()
	reads[2] = fxRead{Pair: "USDJPY", spec: assetTable["usdjpy"], Insufficient: true}
	env := cardEnvelope(fxCardFromReads(reads, fxNow))
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
		t.Fatalf("results: %d, want one per instrument", len(got.Results))
	}
	eur := got.Results[0]
	want := map[string]any{
		"asset": "EURUSD", "ok": true, "timeframe": "1h", "price": 1.154334545,
		"change_pct": -0.05, "change_window": "24h", "change_from": "2026-09-14T08:00:00Z",
		"rsi": 40.8, "ema_relation": "below", "range_position_pct": 43.0,
		"data_as_of": "2026-09-15T08:00:00Z", "freshness": "on_time",
	}
	for k, v := range want {
		if f, isNum := v.(float64); isNum {
			if g, ok := eur[k].(float64); !ok || math.Abs(g-f) > 1e-9 {
				t.Errorf("results[0].%s = %v, want %v", k, eur[k], v)
			}
			continue
		}
		if eur[k] != v {
			t.Errorf("results[0].%s = %v, want %v", k, eur[k], v)
		}
	}
	if eur["reason"] != nil {
		t.Errorf("ok row reason %v", eur["reason"])
	}
	jpy := got.Results[2]
	if jpy["ok"] != false || jpy["reason"] != "insufficient_history" || jpy["price"] != nil {
		t.Errorf("short-history row: %v", jpy)
	}
	if g := got.Results[3]; g["asset"] != "GOLD · COMEX GC=F" || g["ema_relation"] != "below" {
		t.Errorf("gold row: %v", g)
	}
	// Since-previous-close rows say so.
	r := liveFXReads()
	r[0].SinceClose, r[0].RefAt = true, fxAt(11, 22)
	res := fxCardFromReads(r, fxNow).Results[0]
	if res.ChangeWindow != "since_previous_close" || res.ChangeFrom != "2026-09-11T22:00:00Z" {
		t.Errorf("since-close row: %q %q", res.ChangeWindow, res.ChangeFrom)
	}
}

// fxDigestBlock is the FX section of one digest sweep.
func fxDigestBlock(t *testing.T, g gathered, now time.Time) digestSection {
	t.Helper()
	for _, s := range digestSections(g, "", now) {
		if s.key == keyFX {
			return s
		}
	}
	t.Fatal("digest has no FX section")
	return digestSection{}
}

// fxCardMarketLines is what the /fx card itself shows as each instrument's
// first line, in card order: its facts minus the weekend banner, the gold
// header and the indicator lines ("<label>: EMA50 …").
func fxCardMarketLines(c Card) []string {
	var out []string
	for _, f := range c.Facts {
		if f == fxClosedBanner || f == fxGoldHeader || strings.Contains(f, ": EMA50 ") {
			continue
		}
		out = append(out, f)
	}
	return out
}

// The digest FX block shows the market lines of the card fxCardFromReads
// really builds — verbatim, in card order, a dead row included — under a
// title with the oldest bar ("as of"), gold's contract and, on a weekend,
// the Forex closure; rows with a newer bar are named with it, and the card's
// own "last bar" times agree.
func TestFXDigestSectionMatchesCard(t *testing.T) {
	reads := liveFXReads() // USDJPY 07:00, the rest 08:00
	reads[1] = fxRead{Pair: "GBPUSD", spec: assetTable["gbpusd"]}
	g := gathered{fx: reads, fxAnyOK: true, cards: map[string]Card{}}
	fx := fxDigestBlock(t, g, fxNow)
	card := fxCardFromReads(reads, fxNow)
	want := fxCardMarketLines(card)
	if len(want) != len(reads) || len(fx.lines) != len(want)+1 {
		t.Fatalf("digest lines:\n%s\ncard market lines:\n%s", strings.Join(fx.lines, "\n"), strings.Join(want, "\n"))
	}
	if got := strings.Join(fx.lines[:len(want)], "\n"); got != strings.Join(want, "\n") {
		t.Errorf("digest rows != the card's market lines\n%s\n--\n%s", got, strings.Join(want, "\n"))
	}
	if got := fx.lines[len(want)]; got != "Newer bars: EURUSD, GOLD Sep 15 08:00 UTC" {
		t.Errorf("newer-bar line %q", got)
	}
	facts := strings.Join(card.Facts, "\n")
	for _, row := range []struct{ label, at string }{{"EURUSD", "Sep 15 08:00"}, {"GOLD", "Sep 15 08:00"}, {"USDJPY", "Sep 15 07:00"}} {
		if !fxFactHasBar(card, row.label, row.at) {
			t.Errorf("card row %s must print last bar %s UTC:\n%s", row.label, row.at, facts)
		}
	}
	if got := htmlToPlain(fx.title); got != "FX (as of Sep 15 07:00 UTC · gold = COMEX GC=F futures)" {
		t.Errorf("open title %q", got)
	}
	if !fx.asOf.Equal(fxAt(15, 7)) || !fx.asOf.Equal(card.DataTime) {
		t.Errorf("section as_of %v, want the oldest bar = the card's data time %v", fx.asOf, card.DataTime)
	}
	// Every row on one bar: no extra line, the rows alone.
	same := liveFXReads()
	same[2].CloseAt = fxAt(15, 8)
	s := fxDigestBlock(t, gathered{fx: same, fxAnyOK: true, cards: map[string]Card{}}, fxNow)
	if strings.Join(s.lines, "\n") != strings.Join(fxCardMarketLines(fxCardFromReads(same, fxNow)), "\n") {
		t.Errorf("one bar time: digest lines %q", s.lines)
	}
	w := fxDigestBlock(t, g, fxWeekend)
	if got := htmlToPlain(w.title); !strings.HasPrefix(got, "FX (forex closed: pairs show Friday data · as of ") {
		t.Errorf("weekend title %q", got)
	}
	// All short history: the block and the health say so, not "offline".
	short := gathered{fx: []fxRead{{Pair: "EURUSD", spec: assetTable["eurusd"], Insufficient: true}}, cards: map[string]Card{}}
	for _, s := range digestSections(short, "", fxNow) {
		if s.key == keyFX && s.status != statusInsufficientHistory {
			t.Errorf("short-history block status %v", s.status)
		}
	}
	if h := short.health(); h.Reason != statusInsufficientHistory {
		t.Errorf("health reason %v", h.Reason)
	}
}

// Item 10: every line at most 110 characters on every path.
func TestFXLineBudget(t *testing.T) {
	worst := func(key string, price float64) fxRead {
		r := fxOK(key, price, "flat", 100, -12.34, 1, fxAt(11, 22))
		r.SinceClose, r.RefAt = true, fxAt(4, 22)
		return r
	}
	cases := map[string][]fxRead{
		"live":    liveFXReads(),
		"worst":   {worst("eurusd", 12.34567), worst("gbpusd", 1.3), worst("usdjpy", 1234.567), worst("xauusd", 12345.67)},
		"mixed":   {worst("eurusd", 1.1), {Pair: "GBPUSD", spec: assetTable["gbpusd"], Insufficient: true}, {Pair: "USDJPY", spec: assetTable["usdjpy"]}, worst("xauusd", 4000)},
		"allbad":  {{Pair: "EURUSD", spec: assetTable["eurusd"], Insufficient: true}, {Pair: "GOLD · COMEX GC=F", spec: assetTable["xauusd"]}},
		"allshrt": {{Pair: "EURUSD", spec: assetTable["eurusd"], Insufficient: true}},
		"dead":    {{Pair: "EURUSD", spec: assetTable["eurusd"]}},
	}
	for name, reads := range cases {
		for _, now := range []time.Time{fxNow, fxWeekend, fxAt(15, 23).Add(100 * time.Hour)} {
			c := fxCardFromReads(reads, now)
			lines := append([]string{c.Verdict, c.Short}, c.Facts...)
			g := gathered{fx: reads, fxAnyOK: c.effectiveStatus() == statusOK, cards: map[string]Card{}}
			for _, s := range digestSections(g, "", now) {
				if s.key == keyFX {
					lines = append(lines, htmlToPlain(s.title))
					lines = append(lines, s.lines...)
				}
			}
			for _, l := range lines {
				if n := utf8.RuneCountInString(l); n > 110 {
					t.Errorf("%s @%s: %d runes: %q", name, now.Format("Mon 15:04"), n, l)
				}
			}
		}
	}
	if n := len(howTexts[keyFX]); n > 200 {
		t.Errorf("howTexts[fx] %d chars > 200", n)
	}
	if strings.Contains(howTexts[keyFX], "trend") {
		t.Errorf("how text still calls the EMA cross a trend: %q", howTexts[keyFX])
	}
}

// fxFactHasBar: the card's indicator line for label names the bar at.
func fxFactHasBar(c Card, label, at string) bool {
	for _, f := range c.Facts {
		if strings.HasPrefix(f, label+": EMA50 ") && strings.HasSuffix(f, " · last bar "+at+" UTC") {
			return true
		}
	}
	return false
}

// Sunday after the fixed-window reopen (21:00 UTC) Yahoo still serves
// Friday's pair bars (its first Sunday bar comes later), so the title says
// nothing about a closed market: the block's "as of" is what tells the reader
// the rows are Friday's. Gold, back earlier, is named with its newer bar.
func TestFXDigestSundayReopenShowsBarTime(t *testing.T) {
	fri := fxAt(11, 21)
	base := []fxRead{
		fxOK("eurusd", 1.1731, "up", 55.1, 0.12, 0.66, fri),
		fxOK("gbpusd", 1.3561, "up", 52.0, 0.08, 0.5, fri),
		fxOK("usdjpy", 147.33, "down", 47.0, -0.2, 0.3, fri),
		fxOK("xauusd", 3650.4, "up", 61.0, 0.9, 0.95, fri),
	}
	cases := []struct {
		now, goldAt time.Time
		newer       string
	}{
		{fxAt(13, 21).Add(30 * time.Minute), fri, ""},
		{fxAt(13, 23).Add(30 * time.Minute), fxAt(13, 23), "Newer bars: GOLD Sep 13 23:00 UTC"},
		{fxAt(14, 0).Add(10 * time.Minute), fxAt(14, 0), "Newer bars: GOLD Sep 14 00:00 UTC"},
	}
	for _, tc := range cases {
		if !isForexOpen(tc.now) {
			t.Fatalf("%v must be inside the open window", tc.now)
		}
		reads := append([]fxRead{}, base...)
		reads[3].CloseAt, reads[3].RefAt = tc.goldAt, tc.goldAt.Add(-24*time.Hour)
		fx := fxDigestBlock(t, gathered{fx: reads, fxAnyOK: true, cards: map[string]Card{}}, tc.now)
		title := htmlToPlain(fx.title)
		if strings.Contains(title, "closed") || !strings.Contains(title, "as of Sep 11 21:00 UTC") {
			t.Errorf("%s: title %q", tc.now.Format("Mon 15:04"), title)
		}
		var newer string
		for _, l := range fx.lines {
			if strings.HasPrefix(l, "Newer bars:") {
				newer = l
			}
		}
		if newer != tc.newer {
			t.Errorf("%s: newer-bar line %q, want %q", tc.now.Format("Mon 15:04"), newer, tc.newer)
		}
		for _, l := range append([]string{title}, fx.lines...) {
			if n := utf8.RuneCountInString(l); n > fxLineMaxRunes {
				t.Errorf("%s: %d runes: %q", tc.now.Format("Mon 15:04"), n, l)
			}
		}
	}
	// Four rows on four different bars still fit one line.
	spread := liveFXReads()
	for i := range spread {
		spread[i].CloseAt = fxAt(15, 5+i)
	}
	fx := fxDigestBlock(t, gathered{fx: spread, fxAnyOK: true, cards: map[string]Card{}}, fxNow)
	last := fx.lines[len(fx.lines)-1]
	if last != "Newer bars: GBPUSD Sep 15 06:00 · USDJPY Sep 15 07:00 · GOLD Sep 15 08:00 UTC" || utf8.RuneCountInString(last) > fxLineMaxRunes {
		t.Errorf("spread newer-bar line %q", last)
	}
}

// A gold row without a recent bar is not counted as "+ gold" read; the
// header names it instead, on weekdays and weekends, and within budget.
func TestFXHeaderNamesStaleGold(t *testing.T) {
	reads := liveFXReads()
	reads[3].CloseAt = fxAt(15, 5) // 3.5h old at 08:30
	if c := fxCardFromReads(reads, fxNow); c.Verdict != "FX overview · 1h · 3 pairs read · gold: no recent bar" {
		t.Errorf("stale gold verdict %q", c.Verdict)
	}
	if c := fxCardFromReads(liveFXReads(), fxNow); strings.Contains(c.Verdict, "no recent bar") {
		t.Errorf("fresh gold flagged: %q", c.Verdict)
	}
	only := []fxRead{reads[3]}
	if c := fxCardFromReads(only, fxNow); c.Verdict != "FX overview · 1h · gold read · gold: no recent bar" {
		t.Errorf("gold-only stale verdict %q", c.Verdict)
	}
	// Worst case: every part of the header at once.
	worst := []fxRead{
		fxOK("eurusd", 1.1, "up", 50, 0.1, 0.5, fxAt(15, 4)), // data delayed
		{Pair: "GBPUSD", spec: assetTable["gbpusd"], Insufficient: true},
		{Pair: "USDJPY", spec: assetTable["usdjpy"]},
		reads[3],
	}
	c := fxCardFromReads(worst, fxNow)
	if c.Verdict != "FX overview · 1h · 1 of 3 pairs read · 1 short history · 1 unavailable · 1 data delayed · gold: no recent bar" {
		t.Errorf("worst verdict %q", c.Verdict)
	}
	if n := utf8.RuneCountInString(c.Verdict); n > fxLineMaxRunes {
		t.Errorf("worst verdict %d runes", n)
	}
}

// The landing example quotes a reading, never a dead or short row: with every
// pair down and gold live the strongest fact is gold's line. The card itself
// keeps its order.
func TestFXShowcaseExampleSkipsUnreadRows(t *testing.T) {
	for _, dead := range []func(fxRead) fxRead{
		func(r fxRead) fxRead { return fxRead{Pair: r.Pair, spec: r.spec} },
		func(r fxRead) fxRead { return fxRead{Pair: r.Pair, spec: r.spec, Insufficient: true} },
	} {
		reads := liveFXReads()
		for i := 0; i < 3; i++ {
			reads[i] = dead(reads[i])
		}
		c := fxCardFromReads(reads, fxNow)
		if !fxUnreadLine(c.Facts[0]) || !strings.HasPrefix(c.Facts[0], "EURUSD: ") {
			t.Errorf("card order changed: first fact %q", c.Facts[0])
		}
		if got := strongestFact(c); !strings.HasPrefix(got, "GOLD 4320.7 · 24h -0.69% · 38% of the 24h range") {
			t.Errorf("strongest fact %q", got)
		}
		ex := exampleFacts(c)
		if len(ex) == 0 {
			t.Fatal("no example facts")
		}
		for _, f := range ex {
			if strings.Contains(f, "data unavailable") || strings.Contains(f, "insufficient history") {
				t.Errorf("example quotes an unread row: %q", ex)
			}
		}
	}
	if got := strongestFact(fxCardFromReads(liveFXReads(), fxNow)); !strings.HasPrefix(got, "EURUSD 1.1543 · 24h") {
		t.Errorf("live card strongest fact %q", got)
	}
}

// The AI payload names gold as the futures contract (the model sees no gold
// section header); pair rows stay the card's own words.
func TestFXAIPayloadNamesGoldFutures(t *testing.T) {
	g := gathered{fx: liveFXReads(), fxAnyOK: true, cards: map[string]Card{}, at: fxNow}
	var p struct {
		FX []string `json:"fx"`
	}
	if err := json.Unmarshal([]byte(aiPayload(g)), &p); err != nil {
		t.Fatal(err)
	}
	want := []string{
		"EURUSD 1.1543 · 24h -0.05% · 43% of the 24h range · EMA50 below EMA200 · RSI(1h) 40.8",
		"GBPUSD 1.3480 · 24h -0.09% · 55% of the 24h range · EMA50 below EMA200 · RSI(1h) 41.2",
		"USDJPY 154.88 · 24h +0.32% · 91% of the 24h range · EMA50 below EMA200 · RSI(1h) 62.9",
		"GOLD (COMEX GC=F futures) 4320.7 · 24h -0.69% · 38% of the 24h range · EMA50 below EMA200 · RSI(1h) 40.6",
	}
	if strings.Join(p.FX, "\n") != strings.Join(want, "\n") {
		t.Errorf("AI fx rows:\n%s\nwant:\n%s", strings.Join(p.FX, "\n"), strings.Join(want, "\n"))
	}
}

// The push hook's hash of /agents/fx stays put across two reads with the data
// unchanged in the everyday on_time state (every row's last bar closed within
// the hour before the card's clock) — the shared fixture's bars are two hours
// old, so its reads are delayed or closed, never on_time.
func TestFXHookHashStableOnTime(t *testing.T) {
	// The latest past hour with the Forex week open for the bars before it.
	end := time.Now().UTC().Truncate(time.Hour)
	for !isForexOpen(end) || !isForexOpen(end.Add(-3*time.Hour)) {
		end = end.Add(-time.Hour)
	}
	ag := liveHookAgents(t)
	stubYahooWave(t, end, 600) // last bar closes at end
	base := end.Add(30 * time.Minute)
	clk := &stepClock{t: base}
	ag.now = clk.now
	th := newTestHook(t, ag, "http://127.0.0.1:1", nil)
	tg := mustHookTarget(t, "/agents/fx")
	read := func() (int, []byte, string) {
		start := time.Now()
		st, body := th.fetch(t.Context(), tg)
		return st, body, hashOf(t, tg.Agent, body, start, time.Now())
	}
	st1, b1, h1 := read()
	nextWallSecond()
	clk.set(base.Add(61 * time.Second))
	st2, b2, h2 := read()
	if st1 != 200 || st2 != 200 {
		t.Fatalf("status %d / %d: %s", st1, st2, b1)
	}
	var env struct {
		Results []struct {
			Asset, Freshness string
		} `json:"results"`
	}
	if err := json.Unmarshal(b1, &env); err != nil {
		t.Fatal(err)
	}
	if len(env.Results) != len(fxPairs) {
		t.Fatalf("results %d: %s", len(env.Results), b1)
	}
	for _, r := range env.Results {
		if r.Freshness != momentumOnTime {
			t.Errorf("%s freshness %q, want on_time: %s", r.Asset, r.Freshness, b1)
		}
	}
	if h1 != h2 {
		t.Errorf("hash moved with unchanged on_time data\nA: %s\nB: %s", b1, b2)
	}
}

// Gold's bar-age bound is 3h (two bars plus the daily COMEX break): 2h59m is
// on time, 3h01m is not. Pairs keep the Momentum rule (two bars).
func TestFXGoldNoRecentBarBoundary(t *testing.T) {
	g := liveFXReads()[3] // closes Sep 15 08:00
	for _, tc := range []struct {
		age  time.Duration
		want string
	}{
		{2*time.Hour + 59*time.Minute, momentumOnTime},
		{3*time.Hour + time.Minute, fxNoRecentBar},
	} {
		now := g.CloseAt.Add(tc.age)
		if f := fxFreshness(g, now); f != tc.want {
			t.Errorf("gold at %v: %q, want %q", tc.age, f, tc.want)
		}
		line := fxMarketLine(g, now)
		if flagged := strings.HasSuffix(line, " · no bar in the last 3h"); flagged != (tc.want == fxNoRecentBar) {
			t.Errorf("gold at %v: line %q", tc.age, line)
		}
		c := fxCardFromReads(liveFXReads()[3:], now)
		if flagged := strings.Contains(c.Verdict, "gold: no recent bar"); flagged != (tc.want == fxNoRecentBar) {
			t.Errorf("gold at %v: verdict %q", tc.age, c.Verdict)
		}
	}
	p := liveFXReads()[0]
	if f := fxFreshness(p, p.CloseAt.Add(2*time.Hour+59*time.Minute)); f != momentumDataDelayed {
		t.Errorf("pair at 2h59m: %q, want data_delayed (Momentum rule unchanged)", f)
	}
}

// No reading, some short history, some dead: the source answered, so the
// reason is insufficient_history, and the card and the digest state the
// coverage instead of "data source unavailable". All-short and all-dead keep
// their own wording.
func TestFXMixedDegradedStatesCoverage(t *testing.T) {
	short := func(k string) fxRead {
		return fxRead{Pair: assetTable[k].Display, spec: assetTable[k], Insufficient: true}
	}
	dead := func(k string) fxRead { return fxRead{Pair: assetTable[k].Display, spec: assetTable[k]} }
	cases := []struct {
		name          string
		reads         []fxRead
		status        cardStatus
		verdict, line string
	}{
		{"mixed", []fxRead{short("eurusd"), short("gbpusd"), dead("usdjpy"), dead("xauusd")}, statusInsufficientHistory,
			"FX overview · 1h · 0 of 3 pairs read · 2 short history · 2 unavailable",
			"⚪ FX: 0 of 3 pairs read · 2 short history · 2 unavailable"},
		{"all short", []fxRead{short("eurusd"), short("xauusd")}, statusInsufficientHistory,
			"Insufficient history on 1h bars — no FX overview", "⚪ FX: insufficient history on 1h bars"},
		{"all dead", []fxRead{dead("eurusd"), dead("xauusd")}, statusSourceOffline,
			fxOfflineVerdict, "⚪ FX: data unavailable right now"},
	}
	for _, tc := range cases {
		for _, now := range []time.Time{fxNow, fxWeekend} {
			c := fxCardFromReads(tc.reads, now)
			if c.effectiveStatus() != tc.status || c.Verdict != tc.verdict {
				t.Errorf("%s: %v %q", tc.name, c.effectiveStatus(), c.Verdict)
			}
			raw, err := json.Marshal(cardEnvelope(c))
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(raw), `"reason":"`+tc.status.reason()+`"`) {
				t.Errorf("%s: envelope reason: %s", tc.name, raw)
			}
			g := gathered{fx: tc.reads, cards: map[string]Card{}}
			fx := fxDigestBlock(t, g, now)
			if fx.status != tc.status || strings.Join(fx.lines, "\n") != tc.line {
				t.Errorf("%s: digest %v %q", tc.name, fx.status, fx.lines)
			}
			if strings.Contains(fx.title, "forex closed") {
				t.Errorf("%s: no pair read, yet %q", tc.name, fx.title)
			}
			if h := g.health(); h.Reason != tc.status {
				t.Errorf("%s: health reason %v", tc.name, h.Reason)
			}
		}
	}
	if c := fxCardFromReads(cases[0].reads, fxNow); c.Short != "0 of 3 pairs read · 2 short history · 2 unavailable" {
		t.Errorf("mixed short %q", c.Short)
	}
}

// The weekend wording dates the pairs' data, so it needs a pair that was
// read: with every pair down (gold live) neither the card's banner nor the
// digest's "forex closed" appears.
func TestFXWeekendWordingNeedsAPairRead(t *testing.T) {
	gold := liveFXReads()[3]
	gold.CloseAt, gold.RefAt = fxWeekend.Add(-30*time.Minute), fxWeekend.Add(-24*time.Hour-30*time.Minute)
	reads := []fxRead{
		{Pair: "EURUSD", spec: assetTable["eurusd"]},
		{Pair: "GBPUSD", spec: assetTable["gbpusd"], Insufficient: true},
		{Pair: "USDJPY", spec: assetTable["usdjpy"]},
		gold,
	}
	title := func(rs []fxRead) string {
		return fxDigestBlock(t, gathered{fx: rs, fxAnyOK: true, cards: map[string]Card{}}, fxWeekend).title
	}
	c := fxCardFromReads(reads, fxWeekend)
	for _, f := range c.Facts {
		if f == fxClosedBanner {
			t.Errorf("banner with no pair read:\n%s", strings.Join(c.Facts, "\n"))
		}
	}
	if tt := title(reads); strings.Contains(tt, "forex closed") {
		t.Errorf("digest title with no pair read: %q", tt)
	}
	reads[0] = fxOK("eurusd", 1.1731, "up", 55.1, 0.12, 0.66, fxAt(11, 21))
	if c := fxCardFromReads(reads, fxWeekend); c.Facts[0] != fxClosedBanner {
		t.Errorf("one pair read on a weekend: first fact %q", c.Facts[0])
	}
	if tt := title(reads); !strings.Contains(tt, "forex closed: pairs show Friday data") {
		t.Errorf("one pair read on a weekend: title %q", tt)
	}
}
