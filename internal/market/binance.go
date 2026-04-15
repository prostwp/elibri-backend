package market

import (
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

const binanceBaseURL = "https://api.binance.com/api/v3"

// BinanceTicker24h — single ticker from /ticker/24hr
type BinanceTicker24h struct {
	Symbol             string `json:"symbol"`
	PriceChange        string `json:"priceChange"`
	PriceChangePercent string `json:"priceChangePercent"`
	LastPrice          string `json:"lastPrice"`
	HighPrice          string `json:"highPrice"`
	LowPrice           string `json:"lowPrice"`
	OpenPrice          string `json:"openPrice"`
	Volume             string `json:"volume"`
	QuoteVolume        string `json:"quoteVolume"`
}

// FetchCryptoQuotes — fetch top crypto tickers from Binance
func FetchCryptoQuotes(symbols []string) (map[string]types.StockQuote, error) {
	url := fmt.Sprintf("%s/ticker/24hr", binanceBaseURL)

	resp, err := httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("binance ticker: %w", err)
	}
	defer resp.Body.Close()

	var tickers []BinanceTicker24h
	if err := json.NewDecoder(resp.Body).Decode(&tickers); err != nil {
		return nil, fmt.Errorf("binance decode: %w", err)
	}

	// Build lookup set
	wantSet := make(map[string]bool, len(symbols))
	for _, s := range symbols {
		wantSet[s] = true
	}

	result := make(map[string]types.StockQuote)
	for _, t := range tickers {
		if len(symbols) > 0 && !wantSet[t.Symbol] {
			continue
		}

		price, _ := strconv.ParseFloat(t.LastPrice, 64)
		change, _ := strconv.ParseFloat(t.PriceChange, 64)
		changePct, _ := strconv.ParseFloat(t.PriceChangePercent, 64)
		high, _ := strconv.ParseFloat(t.HighPrice, 64)
		low, _ := strconv.ParseFloat(t.LowPrice, 64)
		open, _ := strconv.ParseFloat(t.OpenPrice, 64)
		vol, _ := strconv.ParseFloat(t.QuoteVolume, 64)

		result[t.Symbol] = types.StockQuote{
			Symbol:        t.Symbol,
			Price:         price,
			Change:        change,
			ChangePercent: changePct,
			High:          high,
			Low:           low,
			Open:          open,
			PrevClose:     price - change,
			MarketCap:     vol, // using quote volume as proxy
			Timestamp:     time.Now().Unix(),
		}
	}

	return result, nil
}

// FetchCryptoCandles — historical klines from Binance
func FetchCryptoCandles(symbol string, interval string, limit int) ([]types.OHLCVCandle, error) {
	url := fmt.Sprintf(
		"%s/klines?symbol=%s&interval=%s&limit=%d",
		binanceBaseURL, symbol, interval, limit,
	)

	resp, err := httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("binance klines: %w", err)
	}
	defer resp.Body.Close()

	var raw [][]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("binance klines decode: %w", err)
	}

	candles := make([]types.OHLCVCandle, 0, len(raw))
	for _, row := range raw {
		if len(row) < 6 {
			continue
		}
		openTime := int64(toFloat(row[0]) / 1000) // ms → sec
		open, _ := strconv.ParseFloat(fmt.Sprintf("%v", row[1]), 64)
		high, _ := strconv.ParseFloat(fmt.Sprintf("%v", row[2]), 64)
		low, _ := strconv.ParseFloat(fmt.Sprintf("%v", row[3]), 64)
		close, _ := strconv.ParseFloat(fmt.Sprintf("%v", row[4]), 64)
		vol, _ := strconv.ParseFloat(fmt.Sprintf("%v", row[5]), 64)

		candles = append(candles, types.OHLCVCandle{
			Time:   openTime,
			Open:   open,
			High:   high,
			Low:    low,
			Close:  close,
			Volume: vol,
		})
	}

	return candles, nil
}

// FetchAllUSDTPairs — get all USDT trading pairs from Binance (for scanner)
func FetchAllUSDTPairs() ([]string, error) {
	url := fmt.Sprintf("%s/exchangeInfo", binanceBaseURL)

	resp, err := httpClient.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var info struct {
		Symbols []struct {
			Symbol     string `json:"symbol"`
			Status     string `json:"status"`
			QuoteAsset string `json:"quoteAsset"`
		} `json:"symbols"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, err
	}

	var pairs []string
	for _, s := range info.Symbols {
		if s.Status == "TRADING" && s.QuoteAsset == "USDT" {
			pairs = append(pairs, s.Symbol)
		}
	}

	return pairs, nil
}
