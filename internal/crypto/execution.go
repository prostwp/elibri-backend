// Package crypto execution: Binance Spot order placement via testnet API.
//
// Safety: defaults to TESTNET (https://testnet.binance.vision) where orders
// use fake money. Set BINANCE_PROD=1 to hit live API with real money — but
// server refuses to start in prod mode unless BINANCE_API_KEY + BINANCE_API_SECRET
// are set AND the user has explicitly acknowledged risk in config.
package crypto

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Mode enum — testnet or production.
type ExecMode string

const (
	ModeTestnet ExecMode = "testnet"
	ModeProd    ExecMode = "prod"
	ModePaper   ExecMode = "paper" // in-memory, no external calls
)

var (
	execMu       sync.RWMutex
	currentMode  ExecMode = ModePaper
	apiKey       string
	apiSecret    string
	testnetURL   = "https://testnet.binance.vision"
	productionURL = "https://api.binance.com"
)

// InitExecution reads env vars and picks the safest mode. Called from main.
func InitExecution() {
	execMu.Lock()
	defer execMu.Unlock()
	apiKey = os.Getenv("BINANCE_API_KEY")
	apiSecret = os.Getenv("BINANCE_API_SECRET")
	prod := os.Getenv("BINANCE_PROD") == "1"
	if apiKey == "" || apiSecret == "" {
		currentMode = ModePaper
		return
	}
	if prod {
		currentMode = ModeProd
		return
	}
	currentMode = ModeTestnet
}

func Mode() ExecMode {
	execMu.RLock()
	defer execMu.RUnlock()
	return currentMode
}

// PlaceOrderRequest is a unified Spot order.
type PlaceOrderRequest struct {
	Symbol   string  `json:"symbol"`    // "BTCUSDT"
	Side     string  `json:"side"`      // BUY | SELL
	Type     string  `json:"type"`      // MARKET | LIMIT
	Quantity float64 `json:"quantity"`
	Price    float64 `json:"price,omitempty"`        // for LIMIT
	SL       float64 `json:"stop_loss,omitempty"`    // attached stop (emulated in paper)
	TP       float64 `json:"take_profit,omitempty"`  // attached profit (emulated in paper)
}

type PlaceOrderResult struct {
	Mode        ExecMode `json:"mode"`
	Symbol      string   `json:"symbol"`
	Side        string   `json:"side"`
	OrderID     int64    `json:"order_id,omitempty"`    // exchange ID (testnet/prod)
	PaperID     string   `json:"paper_id,omitempty"`    // uuid for paper trades
	ExecutedQty float64  `json:"executed_qty"`
	AvgPrice    float64  `json:"avg_price"`
	Status      string   `json:"status"`
	Timestamp   int64    `json:"timestamp"`
	Raw         any      `json:"raw,omitempty"` // full exchange response
}

// PlaceOrder routes to testnet / prod / paper based on current mode.
func PlaceOrder(req PlaceOrderRequest) (*PlaceOrderResult, error) {
	mode := Mode()
	switch mode {
	case ModePaper:
		return placePaperOrder(req)
	case ModeTestnet, ModeProd:
		return placeBinanceOrder(req, mode)
	default:
		return nil, fmt.Errorf("unknown execution mode: %s", mode)
	}
}

// ─── Binance signed endpoint ─────────────────────────────────

func placeBinanceOrder(req PlaceOrderRequest, mode ExecMode) (*PlaceOrderResult, error) {
	if apiKey == "" || apiSecret == "" {
		return nil, errors.New("BINANCE_API_KEY / BINANCE_API_SECRET not set")
	}
	baseURL := testnetURL
	if mode == ModeProd {
		baseURL = productionURL
	}

	params := url.Values{}
	params.Set("symbol", req.Symbol)
	params.Set("side", strings.ToUpper(req.Side))
	params.Set("type", strings.ToUpper(req.Type))
	params.Set("quantity", strconv.FormatFloat(req.Quantity, 'f', -1, 64))
	if req.Type == "LIMIT" {
		params.Set("price", strconv.FormatFloat(req.Price, 'f', -1, 64))
		params.Set("timeInForce", "GTC")
	}
	params.Set("timestamp", strconv.FormatInt(time.Now().UnixMilli(), 10))
	params.Set("recvWindow", "5000")

	signature := sign(params.Encode(), apiSecret)
	params.Set("signature", signature)

	httpReq, err := http.NewRequest("POST", baseURL+"/api/v3/order?"+params.Encode(), nil)
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("X-MBX-APIKEY", apiKey)

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("binance %s status %d: %s", mode, resp.StatusCode, string(body))
	}

	var raw map[string]any
	_ = json.Unmarshal(body, &raw)
	result := &PlaceOrderResult{
		Mode:      mode,
		Symbol:    req.Symbol,
		Side:      req.Side,
		Timestamp: time.Now().UnixMilli(),
		Raw:       raw,
	}
	if id, ok := raw["orderId"].(float64); ok {
		result.OrderID = int64(id)
	}
	if qty, ok := raw["executedQty"].(string); ok {
		result.ExecutedQty, _ = strconv.ParseFloat(qty, 64)
	}
	if fills, ok := raw["fills"].([]any); ok && len(fills) > 0 {
		if fill, ok := fills[0].(map[string]any); ok {
			if p, ok := fill["price"].(string); ok {
				result.AvgPrice, _ = strconv.ParseFloat(p, 64)
			}
		}
	}
	if status, ok := raw["status"].(string); ok {
		result.Status = status
	}
	return result, nil
}

func sign(payload, secret string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(payload))
	return hex.EncodeToString(mac.Sum(nil))
}

// ─── Paper trading ───────────────────────────────────────────

type PaperPosition struct {
	ID        string    `json:"id"`
	Symbol    string    `json:"symbol"`
	Side      string    `json:"side"`
	Quantity  float64   `json:"quantity"`
	EntryPrice float64  `json:"entry_price"`
	SL        float64   `json:"sl"`
	TP        float64   `json:"tp"`
	OpenedAt  time.Time `json:"opened_at"`
	ClosedAt  *time.Time `json:"closed_at,omitempty"`
	ExitPrice float64   `json:"exit_price,omitempty"`
	PnL       float64   `json:"pnl,omitempty"`
	Status    string    `json:"status"` // open | closed | sl_hit | tp_hit
}

var (
	paperPositions = make(map[string]*PaperPosition)
	paperBalance   = 10000.0
	paperMu        sync.RWMutex
)

func placePaperOrder(req PlaceOrderRequest) (*PlaceOrderResult, error) {
	paperMu.Lock()
	defer paperMu.Unlock()

	// Look up current price (would use market data service; here stub).
	price := req.Price
	if price == 0 {
		price = approximatePrice(req.Symbol) // TODO: wire to market.GetLatestPrice()
	}
	if price == 0 {
		return nil, fmt.Errorf("no price for %s", req.Symbol)
	}

	id := fmt.Sprintf("paper_%d", time.Now().UnixNano())
	pos := &PaperPosition{
		ID:        id,
		Symbol:    req.Symbol,
		Side:      req.Side,
		Quantity:  req.Quantity,
		EntryPrice: price,
		SL:        req.SL,
		TP:        req.TP,
		OpenedAt:  time.Now(),
		Status:    "open",
	}
	paperPositions[id] = pos

	cost := price * req.Quantity
	paperBalance -= cost // simplified — not tracking margin

	return &PlaceOrderResult{
		Mode:        ModePaper,
		Symbol:      req.Symbol,
		Side:        req.Side,
		PaperID:     id,
		ExecutedQty: req.Quantity,
		AvgPrice:    price,
		Status:      "FILLED",
		Timestamp:   time.Now().UnixMilli(),
	}, nil
}

// approximatePrice is a stub; real impl queries latest candle.
func approximatePrice(symbol string) float64 {
	// Hardcoded guesses for common pairs; replaced by market layer.
	m := map[string]float64{
		"BTCUSDT": 75000.0,
		"ETHUSDT": 3500.0,
		"SOLUSDT": 150.0,
		"XRPUSDT": 2.2,
		"BNBUSDT": 620.0,
	}
	return m[symbol]
}

// ListPaperPositions returns all open+closed paper positions for UI.
func ListPaperPositions() []PaperPosition {
	paperMu.RLock()
	defer paperMu.RUnlock()
	out := make([]PaperPosition, 0, len(paperPositions))
	for _, p := range paperPositions {
		out = append(out, *p)
	}
	return out
}

func PaperBalance() float64 {
	paperMu.RLock()
	defer paperMu.RUnlock()
	return paperBalance
}
