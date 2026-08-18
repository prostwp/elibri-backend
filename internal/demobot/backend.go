package demobot

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// BackendClient is a thin typed reader over the PUBLIC (allowlisted, no-JWT)
// AlphaVizor REST endpoints. Field sets mirror docs/openapi.yaml — only the
// fields the bot renders are declared.
type BackendClient struct {
	base string
	hc   *http.Client
}

func NewBackendClient(base string) *BackendClient {
	return &BackendClient{
		base: base,
		hc:   &http.Client{Timeout: 10 * time.Second},
	}
}

func (c *BackendClient) getJSON(ctx context.Context, path string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+path, nil)
	if err != nil {
		return err
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	limited := io.LimitReader(resp.Body, 8<<20)
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, limited) // drain for connection reuse
		return fmt.Errorf("backend %s: HTTP %d", path, resp.StatusCode)
	}
	if err := json.NewDecoder(limited).Decode(out); err != nil {
		_, _ = io.Copy(io.Discard, limited)
		return err
	}
	_, _ = io.Copy(io.Discard, limited)
	return nil
}

// ── /api/v1/macro ────────────────────────────────────────────────────────────

type MacroLamp struct {
	Key      string   `json:"key"`
	Label    string   `json:"label"`
	Value    *float64 `json:"value"`
	OK       bool     `json:"ok"` // value present — false (with Value nil) on N/D; absent on older payloads, so readers also fall back to Value != nil
	DeltaPct *float64 `json:"delta_pct"`
	Status   string   `json:"status"` // tailwind | neutral | headwind | ""
}

type FearGreedCheck struct {
	Value int    `json:"value"`
	Label string `json:"label"`
	OK    bool   `json:"ok"`
}

type MacroResp struct {
	Regime        string          `json:"regime"` // risk_on | mixed | risk_off | unknown (unknown = zero lamps carry a value)
	Composite     *int            `json:"composite"`
	TradfinOpen   bool            `json:"tradfin_market_open"` // clock-based futures week (Sun 22:00 → Fri 21:00 UTC), not a data signal
	CapturedAt    string          `json:"captured_at"`
	Lamps         []MacroLamp     `json:"lamps"`
	FNG           *FearGreedCheck `json:"fng"`
	GeneratedIdea string          `json:"generated_idea"`
}

func (c *BackendClient) Macro(ctx context.Context) (*MacroResp, error) {
	var out MacroResp
	if err := c.getJSON(ctx, "/api/v1/macro", &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ── /api/v1/whale-flow ───────────────────────────────────────────────────────

type WhaleFlow struct {
	Asset         string  `json:"asset"`
	NetFlowUSD24h float64 `json:"net_flow_usd_24h"`
	Direction     string  `json:"direction"` // inflow | outflow | neutral
	InflowUSD24h  float64 `json:"inflow_usd_24h"`
	OutflowUSD24h float64 `json:"outflow_usd_24h"`
	TxCount24h    int     `json:"tx_count_24h"`
	Confidence    int     `json:"confidence"` // 0 = "too little data" sentinel
	Partial       bool    `json:"partial"`
	Source        string  `json:"source"`
	// Baseline pair — pointers so a payload without them (older snapshots)
	// is distinguishable from a genuine zero and no fake baseline renders.
	NetFlowPrev24h *float64 `json:"net_flow_prev_24h"`
	FlowPct        *float64 `json:"flow_pct"` // % change vs prior 24h, ±999 sentinel-capped
}

type WhaleTransfer struct {
	Chain        string    `json:"chain"`
	TxHash       string    `json:"tx_hash"`
	Timestamp    time.Time `json:"timestamp"`
	Asset        string    `json:"asset"`
	AmountNative float64   `json:"amount_native"`
	AmountUSD    float64   `json:"amount_usd"`
	Direction    string    `json:"direction"`
	Exchange     string    `json:"exchange"`
}

type WhaleResp struct {
	CapturedAt string          `json:"captured_at"`
	Flows      []WhaleFlow     `json:"flows"`
	Transfers  []WhaleTransfer `json:"transfers"`
}

func (c *BackendClient) WhaleFlow(ctx context.Context, limit int) (*WhaleResp, error) {
	var out WhaleResp
	if err := c.getJSON(ctx, fmt.Sprintf("/api/v1/whale-flow?limit=%d", limit), &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ── /api/v1/funding/liquidations ─────────────────────────────────────────────

type Liquidation struct {
	Symbol   string    `json:"symbol"`
	Side     string    `json:"side"` // long_liq | short_liq
	Qty      float64   `json:"qty"`
	Price    float64   `json:"price"`
	USDValue float64   `json:"usd_value"`
	TS       time.Time `json:"ts"`
}

type LiqZone struct {
	Symbol    string  `json:"symbol"`
	PriceBand string  `json:"price_band"`
	TotalUSD  float64 `json:"total_usd"`
	Count     int     `json:"count"`
	Side      string  `json:"side"`
}

type FundingResp struct {
	CapturedAt string        `json:"captured_at"`
	Feed       []Liquidation `json:"feed"`
	Zones      []LiqZone     `json:"zones"`
}

func (c *BackendClient) FundingLiquidations(ctx context.Context) (*FundingResp, error) {
	var out FundingResp
	if err := c.getJSON(ctx, "/api/v1/funding/liquidations", &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ── /api/v1/market/momentum ──────────────────────────────────────────────────

// MomentumItem — windows with too little history are omitted → pointers.
type MomentumItem struct {
	RS1D         *float64 `json:"rs_1d"`
	RS7D         *float64 `json:"rs_7d"`
	RS30D        *float64 `json:"rs_30d"`
	VolGrowthPct *float64 `json:"vol_growth_pct"`
}

type MomentumResp struct {
	Baseline string                  `json:"baseline"`
	Items    map[string]MomentumItem `json:"items"`
}

func (c *BackendClient) MomentumRS(ctx context.Context, symbols []string) (*MomentumResp, error) {
	q := url.Values{}
	q.Set("symbols", joinComma(symbols))
	var out MomentumResp
	if err := c.getJSON(ctx, "/api/v1/market/momentum?"+q.Encode(), &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ── /api/v1/market/fear-greed ────────────────────────────────────────────────

type FearGreedResp struct {
	Value  int    `json:"value"`
	Label  string `json:"label"`
	Source string `json:"source"`
}

func (c *BackendClient) FearGreed(ctx context.Context) (*FearGreedResp, error) {
	var out FearGreedResp
	if err := c.getJSON(ctx, "/api/v1/market/fear-greed", &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ── /api/v1/market/mood-read ─────────────────────────────────────────────────

type MoodResp struct {
	Read   string `json:"read"`
	Source string `json:"source"`
}

func (c *BackendClient) MoodRead(ctx context.Context) (*MoodResp, error) {
	var out MoodResp
	if err := c.getJSON(ctx, "/api/v1/market/mood-read", &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ── /api/v1/narratives ───────────────────────────────────────────────────────

type NarrativeSnapshot struct {
	Narrative      string `json:"narrative"`
	TrendScore     int    `json:"trend_score"`
	Stage          string `json:"stage"` // early | trending | mainstream | declining
	SentimentLabel string `json:"sentiment_label"`
	MentionCount   int    `json:"mention_count_24h"`
	Confidence     int    `json:"confidence"`
	// GeneratedIdea is the Haiku-written analytical paragraph the API
	// attaches to the TOP narrative only (omitempty on the wire).
	GeneratedIdea string `json:"generated_idea"`
}

type NarrativesResp struct {
	CapturedAt string              `json:"captured_at"`
	Narratives []NarrativeSnapshot `json:"narratives"`
}

func (c *BackendClient) Narratives(ctx context.Context) (*NarrativesResp, error) {
	var out NarrativesResp
	if err := c.getJSON(ctx, "/api/v1/narratives", &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ── helpers ──────────────────────────────────────────────────────────────────

func joinComma(parts []string) string {
	out := ""
	for i, p := range parts {
		if i > 0 {
			out += ","
		}
		out += p
	}
	return out
}

// parseWhen parses an RFC3339 stamp, falling back to now (UTC) — footer times
// must never be empty or fake-precise.
func parseWhen(s string) time.Time {
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t.UTC()
	}
	return time.Now().UTC()
}
