package demobot

// trendchart.go — GET /agents/trend/chart: the chart that draws exactly what
// the Trend Agent reads (docs/demobot-http.md, "Trend chart").
//
// Nothing here restates the agent's rules. State, verdict, price, the pullback
// zone and the invalidation all come from the one trendReadOf read the card
// uses, through the same gating functions (pullbackZoneFor, invalidationFor,
// trendVerdict). The pivots are the same swingPointsIdx wing-3 points, and the
// labelled ones are the exact tail orderedPivots hands hhhlStructure. What this
// file adds is only the drawing data: candles, aligned EMA series, pivot times.

import (
	"fmt"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// trendChartBars is how many of the agent's closed bars the chart returns (the
// newest ones). The agent reads the whole window; the chart shows its tail.
const trendChartBars = 200

type chartCandle struct {
	Time  int64   `json:"time"` // unix seconds UTC, bar OPEN
	Open  float64 `json:"open"`
	High  float64 `json:"high"`
	Low   float64 `json:"low"`
	Close float64 `json:"close"`
}

type chartPoint struct {
	Time  int64   `json:"time"`
	Value float64 `json:"value"`
}

type chartZone struct {
	Low  float64 `json:"low"`
	High float64 `json:"high"`
}

type chartInvalidation struct {
	Level float64 `json:"level"`
	Side  string  `json:"side"` // above | below
}

type chartPivot struct {
	Time  int64   `json:"time"`
	Price float64 `json:"price"`
	Kind  string  `json:"kind"`  // high | low
	Label string  `json:"label"` // HH | HL | LH | LL | ""
}

// chartLive tells the browser which public kline stream continues the chart.
type chartLive struct {
	Provider string `json:"provider"`
	Symbol   string `json:"symbol"`
	Interval string `json:"interval"`
}

// trendChart is the /agents/trend/chart response body.
type trendChart struct {
	Agent        string             `json:"agent"`
	Asset        string             `json:"asset"`
	AssetKey     string             `json:"asset_key"`
	Timeframe    string             `json:"timeframe"`
	OK           bool               `json:"ok"`
	Reason       *string            `json:"reason"` // always null on a 200; degraded reads are 503s
	State        string             `json:"state"`
	Verdict      string             `json:"verdict"`
	Price        float64            `json:"price"`
	Candles      []chartCandle      `json:"candles"`
	EMA20        []chartPoint       `json:"ema20"`
	EMA50        []chartPoint       `json:"ema50"`
	EMA200       []chartPoint       `json:"ema200"`
	PullbackZone *chartZone         `json:"pullback_zone"`
	Invalidation *chartInvalidation `json:"invalidation"`
	Pivots       []chartPivot       `json:"pivots"`
	Structure    string             `json:"structure"`
	DataAsOf     string             `json:"data_as_of"`
	Source       string             `json:"source"`
	Live         *chartLive         `json:"live"`
	Disclaimer   string             `json:"disclaimer"`

	dataTime time.Time // Last-Modified; not serialized
}

// trendChartOf builds the chart from the agent's window (oldest first, CLOSED
// bars only — exactly what trendCandlesFor returns). Pure: no clock, no network.
// ok=false means the window is too short for the agent's indicator set; the
// caller serves that as insufficient_history, the same as the card.
func trendChartOf(spec assetSpec, assetKey string, candles []types.OHLCVCandle) (trendChart, bool) {
	r := trendReadOf(candles)
	if !r.OK {
		return trendChart{}, false
	}
	n := len(candles)
	start := n - trendChartBars
	if start < 0 {
		start = 0 // unreachable while trendMinBars > trendChartBars; kept for safety
	}
	closes := closesOf(candles)
	highs, lows := highsLowsOf(candles)

	out := trendChart{
		Agent:      "Trend Agent",
		Asset:      spec.Display,
		AssetKey:   assetKey,
		Timeframe:  spec.Interval,
		OK:         true,
		State:      r.State,
		Verdict:    trendVerdict(r, spec.Interval),
		Price:      r.Last,
		Candles:    make([]chartCandle, 0, n-start),
		EMA20:      emaChartSeries(candles, closes, 20, start),
		EMA50:      emaChartSeries(candles, closes, 50, start),
		EMA200:     emaChartSeries(candles, closes, 200, start),
		Pivots:     trendChartPivots(candles, highs, lows, r.Structure, start),
		Structure:  r.Structure,
		Source:     spec.Source,
		Disclaimer: disclaimerText,
		dataTime:   closeTimeOf(candles, spec.Interval),
	}
	out.DataAsOf = out.dataTime.UTC().Format(time.RFC3339)
	for _, c := range candles[start:] {
		out.Candles = append(out.Candles, chartCandle{c.Time, c.Open, c.High, c.Low, c.Close})
	}

	// Zone and invalidation exist only where the agent confirmed a trend: an
	// unconfirmed state has no trend to pull back within or to invalidate.
	// Neither the card nor the chart shows a level there: a line on a chart
	// reads as a claim whatever its caption says.
	if r.Confirmed() {
		if z := pullbackZoneFor(r.State, r.EMA20, r.EMA50); z != nil {
			out.PullbackZone = &chartZone{Low: math.Min(z.From, z.To), High: math.Max(z.From, z.To)}
		}
		// trendRead.ATR is the card's own ATR(14) — one definition for card and
		// chart, same >0 guard against a degenerate series.
		if r.ATR > 0 {
			level, side := invalidationFor(r.State, r.EMA50, r.EMA200, r.ATR)
			out.Invalidation = &chartInvalidation{Level: level, Side: side}
		}
	}
	if spec.Source == srcBinance {
		out.Live = &chartLive{Provider: srcBinance, Symbol: spec.Symbol, Interval: spec.Interval}
	}
	return out, true
}

// emaChartSeries computes EMA(period) over the FULL window (so the last point
// is exactly the value the agent read) and returns it aligned to the returned
// candles, only where defined: bar i qualifies once i+1 >= period — the same
// "shorter than the period is not an EMA" rule emaLast applies.
func emaChartSeries(candles []types.OHLCVCandle, closes []float64, period, start int) []chartPoint {
	s := emaSeries(closes, period)
	out := make([]chartPoint, 0, len(candles)-start)
	for i := start; i < len(candles); i++ {
		if i+1 < period {
			continue
		}
		out = append(out, chartPoint{Time: candles[i].Time, Value: s[i]})
	}
	return out
}

// trendChartPivots returns the agent's wing-3 swing points that fall inside the
// returned window, chronological. Labels go ONLY on pivots hhhlStructure
// actually compared, and only when it produced a reading: within the last six
// alternating pivots each high is compared with the previous high and each low
// with the previous low. The first high and first low of that tail are the
// baselines of those comparisons and carry "" — labelling them would need a
// pivot the agent never looked at. A tie (strictly neither higher nor lower,
// which is what made the reading "mixed") also carries "".
func trendChartPivots(candles []types.OHLCVCandle, highs, lows []float64, structure string, start int) []chartPivot {
	swingHighs, swingLows := swingPointsIdx(highs, lows, 3)
	pivots := orderedPivots(swingHighs, swingLows)

	type pivotKey struct {
		idx    int
		isHigh bool
	}
	labels := map[pivotKey]string{}
	if structure != "" && len(pivots) >= structNeedPivots {
		tail := pivots[len(pivots)-structNeedPivots:]
		var prevHigh, prevLow *structPivot
		for i := range tail {
			p := &tail[i]
			prev := &prevLow
			if p.isHigh {
				prev = &prevHigh
			}
			if *prev != nil {
				labels[pivotKey{p.idx, p.isHigh}] = pivotLabel(p.isHigh, p.price, (*prev).price)
			}
			*prev = p
		}
	}

	out := make([]chartPivot, 0, len(pivots))
	for _, p := range pivots {
		if p.idx < start {
			continue
		}
		kind := "low"
		if p.isHigh {
			kind = "high"
		}
		out = append(out, chartPivot{
			Time:  candles[p.idx].Time,
			Price: p.price,
			Kind:  kind,
			Label: labels[pivotKey{p.idx, p.isHigh}],
		})
	}
	return out
}

// pivotLabel words one strict comparison the way hhhlStructure makes it.
func pivotLabel(isHigh bool, price, prev float64) string {
	switch {
	case isHigh && price > prev:
		return "HH"
	case isHigh && price < prev:
		return "LH"
	case !isHigh && price > prev:
		return "HL"
	case !isHigh && price < prev:
		return "LL"
	default:
		return ""
	}
}

// handleTrendChart serves GET /agents/trend/chart?asset=. Degraded reads reuse
// writeCard with the same offline/insufficient cards TrendCard builds, so the
// 503 body and reason can never drift from /agents/trend's.
func (s *HTTPServer) handleTrendChart(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	if len(q["asset"]) > 1 {
		writeErr(w, http.StatusBadRequest, `duplicate parameter "asset" — pass it once`)
		return
	}
	for _, p := range []string{"assets", "tf"} {
		if _, has := q[p]; has {
			writeErr(w, http.StatusBadRequest, fmt.Sprintf(
				"/agents/trend/chart does not take a ?%s= parameter — the chart is the agent's own timeframe", p))
			return
		}
	}
	arg := strings.TrimSpace(q.Get("asset"))
	key := "btc" // resolveAsset's default
	if arg != "" {
		k, err := resolveAssetKey(arg)
		if err != nil {
			writeErr(w, http.StatusBadRequest, fmt.Sprintf(
				"unknown asset %q — allowed: %s (aliases: xau, gold, bitcoin)", arg, strings.Join(assetKeys(), ", ")))
			return
		}
		key = k
	}
	spec := assetTable[key]

	candles, complete, err := s.ag.candlesWindow(r.Context(), spec, trendKlineLimit) // the card's own window
	if err != nil {
		s.writeCard(w, r, assetOffline(spec, "Trend Agent", "Trend", keyTrend, howTexts[keyTrend]))
		return
	}
	chart, ok := trendChartOf(spec, key, candles)
	if !ok {
		c := insufficientCard(spec, "Trend Agent", "Trend", keyTrend, howTexts[keyTrend], "EMA200/ADX(14)")
		c.DataTime = closeTimeOf(candles, spec.Interval)
		s.writeCard(w, r, c)
		return
	}
	// Validator only for a full, contiguous Binance window (complete — see
	// candlesWindow): then its closed bars version the chart. A Yahoo
	// chart gets none — Yahoo can publish a bar late or revise a served bar
	// under the same timestamp, so its data time is not a version of the body
	// (see decorateFXAt). data_as_of in the body is the same either way.
	stamp := chart.dataTime
	if spec.Source == srcYahoo || !complete {
		stamp = time.Time{}
	}
	writeJSONAt(w, r, http.StatusOK, stamp, chart)
}
