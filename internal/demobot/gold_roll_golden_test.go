package demobot

// gold_roll_golden_test.go — proof that the roll-window change leaves the gold
// card byte-identical OUTSIDE a roll window, except the one disclosure line.
//
// testdata/gold_roll_base_437c6a4.json was generated on 437c6a4 (before the
// roll-window change) by this very file:
//
//	UPDATE_GOLD_ROLL_BASE=1 go test -run TestGoldRollBaseGolden ./internal/demobot/
//
// It holds the full HTTP envelope (facts, blocks, gold, card_html …) of every
// card in goldAllCards() — every wording path of the pure builder — plus a set
// of cards built through the whole pipeline (GoldCard over a scripted Yahoo
// at fixed dates). Never regenerate it for the roll work: the point is that it
// predates it.

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

const goldRollBasePath = "gold_roll_base_437c6a4.json"

// yahooSeries is one symbol's scripted answer: daily and hourly bars, or an
// HTTP status other than 200 for both.
type yahooSeries struct {
	d1, h1 []types.OHLCVCandle
	status int // 0 = 200
}

// goldYahooStub serves a Yahoo chart API by symbol and interval, and counts
// the requests per "symbol|interval". Unknown symbols answer 404 like Yahoo.
type goldYahooStub struct {
	mu     sync.Mutex
	series map[string]yahooSeries
	hits   map[string]int
	// before, when set, runs on every request before it is answered
	// (outside the lock): a test holds requests with it.
	before func(sym, interval string)
}

func (s *goldYahooStub) set(sym string, ys yahooSeries) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.series[sym] = ys
}

func (s *goldYahooStub) count(key string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.hits[key]
}

func (s *goldYahooStub) total(prefix string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for k, v := range s.hits {
		if strings.HasPrefix(k, prefix) {
			n += v
		}
	}
	return n
}

func yahooChartBody(t *testing.T, bars []types.OHLCVCandle) []byte {
	t.Helper()
	ts := make([]int64, len(bars))
	o, h, l, c := make([]float64, len(bars)), make([]float64, len(bars)), make([]float64, len(bars)), make([]float64, len(bars))
	for i, b := range bars {
		ts[i], o[i], h[i], l[i], c[i] = b.Time, b.Open, b.High, b.Low, b.Close
	}
	body, err := json.Marshal(map[string]any{"chart": map[string]any{"result": []any{map[string]any{
		"timestamp": ts,
		"indicators": map[string]any{"quote": []any{map[string]any{
			"open": o, "high": h, "low": l, "close": c, "volume": make([]any, len(bars)),
		}}},
	}}}})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func newGoldYahooStub(t *testing.T) *goldYahooStub {
	t.Helper()
	s := &goldYahooStub{series: map[string]yahooSeries{}, hits: map[string]int{}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sym, _ := url.PathUnescape(r.URL.Path[strings.LastIndex(r.URL.Path, "/")+1:])
		iv := r.URL.Query().Get("interval")
		s.mu.Lock()
		s.hits[sym+"|"+iv]++
		ys, ok := s.series[sym]
		before := s.before
		s.mu.Unlock()
		if before != nil {
			before(sym, iv)
		}
		switch {
		case !ok:
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"chart":{"result":null,"error":{"code":"Not Found","description":"No data found, symbol may be delisted"}}}`))
			return
		case ys.status != 0 && ys.status != http.StatusOK:
			w.WriteHeader(ys.status)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if iv == "1d" {
			_, _ = w.Write(yahooChartBody(t, ys.d1))
			return
		}
		_, _ = w.Write(yahooChartBody(t, ys.h1))
	}))
	t.Cleanup(srv.Close)
	orig := yahooChartBase
	yahooChartBase = srv.URL + "/v8/finance/chart/"
	t.Cleanup(func() { yahooChartBase = orig })
	return s
}

// goldRollMoment is the fixed reading moment of the pipeline cases: a Tuesday
// mid-session, far from any month end (no roll by the calendar either).
var goldRollMoment = time.Date(2026, 9, 15, 10, 30, 0, 0, time.UTC)

// fixedDaily is n closed daily bars ending with the 1d candle of 2026-09-14
// (04:00 UTC stamps, the GC=F grid), shaped by shape(i).
func fixedDaily(n int, shape func(i int) (o, h, l, c float64)) []types.OHLCVCandle {
	last := time.Date(2026, 9, 14, 4, 0, 0, 0, time.UTC).Unix()
	out := make([]types.OHLCVCandle, n)
	for i := range out {
		o, h, l, c := shape(i)
		out[i] = types.OHLCVCandle{Time: last - int64(n-1-i)*86400, Open: o, High: h, Low: l, Close: c}
	}
	return out
}

// fixedHourly is eight closed hourly bars, the last one opening at lastOpen,
// all at px (±1 for the high/low).
func fixedHourly(lastOpen time.Time, px float64) []types.OHLCVCandle {
	const n = 8
	out := make([]types.OHLCVCandle, n)
	for i := range out {
		out[i] = types.OHLCVCandle{Time: lastOpen.Unix() - int64(n-1-i)*3600, Open: px, High: px + 1, Low: px - 1, Close: px}
	}
	return out
}

// shiftBars is the same bars on another contract: every price moved by d.
func shiftBars(bars []types.OHLCVCandle, d float64) []types.OHLCVCandle {
	out := make([]types.OHLCVCandle, len(bars))
	for i, b := range bars {
		out[i] = types.OHLCVCandle{Time: b.Time, Open: b.Open + d, High: b.High + d, Low: b.Low + d, Close: b.Close + d, Volume: b.Volume}
	}
	return out
}

// goldPipelineCase is one card through GoldCard at a fixed clock.
type goldPipelineCase struct {
	daily []types.OHLCVCandle
	px    float64
	pxAt  time.Time // open stamp of the last closed 1h bar
	now   time.Time
}

func goldPipelineCases() map[string]goldPipelineCase {
	lastHour := time.Date(2026, 9, 15, 9, 0, 0, 0, time.UTC)
	return map[string]goldPipelineCase{
		"rising/inside":  {fixedDaily(260, risingDay), 5108, lastHour, goldRollMoment},
		"rising/above":   {fixedDaily(260, risingDay), 5200, lastHour, goldRollMoment},
		"rising/below":   {fixedDaily(260, risingDay), 5000, lastHour, goldRollMoment},
		"flat/inside":    {fixedDaily(260, flatDay), 2000, lastHour, goldRollMoment},
		"zigzag/inside":  {fixedDaily(260, zigzagDay), 4371.3, lastHour, goldRollMoment},
		"zigzag/above":   {fixedDaily(260, zigzagDay), 4520, lastHour, goldRollMoment},
		"zigzag/below":   {fixedDaily(260, zigzagDay), 4200, lastHour, goldRollMoment},
		"rising/stale":   {fixedDaily(260, risingDay), 5108, lastHour.Add(-9 * time.Hour), goldRollMoment},
		"rising/weekend": {fixedDaily(260, risingDay), 5108, lastHour, time.Date(2026, 9, 19, 12, 0, 0, 0, time.UTC)},
	}
}

// serveGoldPipelineCase scripts GC=F for the case. The near and next
// contracts of the date (GCZ26, GCG27) are served too: the near one carries
// GC=F's own bars (GC=F sits on it, no roll in progress), the next one the
// same bars 40.00 higher. A build that does not look at contracts never asks
// for them.
func serveGoldPipelineCase(s *goldYahooStub, pc goldPipelineCase) {
	h1 := fixedHourly(pc.pxAt, pc.px)
	s.set("GC=F", yahooSeries{d1: pc.daily, h1: h1})
	s.set("GCZ26.CMX", yahooSeries{d1: pc.daily, h1: h1})
	s.set("GCG27.CMX", yahooSeries{d1: shiftBars(pc.daily, 40), h1: shiftBars(h1, 40)})
}

// goldRollDumpCards is every card of the dump, by name.
func goldRollDumpCards(t *testing.T) map[string]Card {
	t.Helper()
	out := map[string]Card{}
	for k, c := range goldAllCards() {
		out["pure/"+k] = c
	}
	cases := goldPipelineCases()
	names := make([]string, 0, len(cases))
	for k := range cases {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, name := range names {
		pc := cases[name]
		s := newGoldYahooStub(t)
		serveGoldPipelineCase(s, pc)
		ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
		now := pc.now
		ag.now = func() time.Time { return now }
		out["pipeline/"+name] = ag.GoldCard(context.Background())
	}
	return out
}

// goldRollDump is the envelope of every card, as JSON values.
func goldRollDump(t *testing.T) map[string]json.RawMessage {
	t.Helper()
	out := map[string]json.RawMessage{}
	for k, c := range goldRollDumpCards(t) {
		b, err := json.Marshal(cardEnvelope(c))
		if err != nil {
			t.Fatal(err)
		}
		out[k] = b
	}
	return out
}

func TestGoldRollBaseGolden(t *testing.T) {
	path := filepath.Join("testdata", goldRollBasePath)
	got := goldRollDump(t)
	if os.Getenv("UPDATE_GOLD_ROLL_BASE") == "1" {
		b, err := json.MarshalIndent(got, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	t.Skip("base generator only; the comparison lives in gold_roll_test.go")
}
