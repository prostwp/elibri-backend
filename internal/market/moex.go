package market

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

const moexBaseURL = "https://iss.moex.com/iss"

var httpClient = &http.Client{Timeout: 10 * time.Second}

// FetchQuotes — batch quotes from MOEX ISS API
func FetchQuotes(tickers []string) (map[string]types.StockQuote, error) {
	symbols := strings.Join(tickers, ",")
	url := fmt.Sprintf(
		"%s/engines/stock/markets/shares/boards/TQBR/securities.json?iss.meta=off&iss.only=marketdata&marketdata.columns=SECID,LAST,CHANGE,HIGH,LOW,OPEN,ISSUECAPITALIZATION&securities=%s",
		moexBaseURL, symbols,
	)

	resp, err := httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("moex quotes: %w", err)
	}
	defer resp.Body.Close()

	var raw struct {
		Marketdata struct {
			Columns []string        `json:"columns"`
			Data    [][]interface{} `json:"data"`
		} `json:"marketdata"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("moex decode: %w", err)
	}

	result := make(map[string]types.StockQuote)
	for _, row := range raw.Marketdata.Data {
		if len(row) < 7 || row[0] == nil || row[1] == nil {
			continue
		}
		symbol := fmt.Sprintf("%v", row[0])
		price := toFloat(row[1])
		change := toFloat(row[2])
		high := toFloat(row[3])
		low := toFloat(row[4])
		open := toFloat(row[5])
		marketCap := toFloat(row[6])

		changePct := 0.0
		if open > 0 {
			changePct = ((price - open) / open) * 100
		}

		result[symbol] = types.StockQuote{
			Symbol:        symbol,
			Price:         price,
			Change:        change,
			ChangePercent: changePct,
			High:          high,
			Low:           low,
			Open:          open,
			PrevClose:     price - change,
			MarketCap:     marketCap,
			Timestamp:     time.Now().Unix(),
		}
	}

	return result, nil
}

// FetchCandles — historical daily candles from MOEX
func FetchCandles(ticker string, days int) ([]types.OHLCVCandle, error) {
	from := time.Now().AddDate(0, 0, -days).Format("2006-01-02")
	url := fmt.Sprintf(
		"%s/engines/stock/markets/shares/boards/TQBR/securities/%s/candles.json?iss.meta=off&iss.only=candles&candles.columns=begin,open,high,low,close,volume&interval=24&from=%s",
		moexBaseURL, ticker, from,
	)

	resp, err := httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("moex candles: %w", err)
	}
	defer resp.Body.Close()

	var raw struct {
		Candles struct {
			Data [][]interface{} `json:"data"`
		} `json:"candles"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("moex candles decode: %w", err)
	}

	candles := make([]types.OHLCVCandle, 0, len(raw.Candles.Data))
	for _, row := range raw.Candles.Data {
		if len(row) < 6 {
			continue
		}
		dateStr := fmt.Sprintf("%v", row[0])
		t, _ := time.Parse("2006-01-02 15:04:05", dateStr)

		candles = append(candles, types.OHLCVCandle{
			Time:   t.Unix(),
			Open:   toFloat(row[1]),
			High:   toFloat(row[2]),
			Low:    toFloat(row[3]),
			Close:  toFloat(row[4]),
			Volume: toFloat(row[5]),
		})
	}

	return candles, nil
}

func toFloat(v interface{}) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	case nil:
		return 0
	default:
		return 0
	}
}
