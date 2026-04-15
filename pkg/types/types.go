package types

import "time"

// OHLCVCandle mirrors frontend OHLCVCandle type
type OHLCVCandle struct {
	Time   int64   `json:"time"`
	Open   float64 `json:"open"`
	High   float64 `json:"high"`
	Low    float64 `json:"low"`
	Close  float64 `json:"close"`
	Volume float64 `json:"volume"`
}

type StockQuote struct {
	Symbol        string  `json:"symbol"`
	Price         float64 `json:"price"`
	Change        float64 `json:"change"`
	ChangePercent float64 `json:"changePercent"`
	High          float64 `json:"high"`
	Low           float64 `json:"low"`
	Open          float64 `json:"open"`
	PrevClose     float64 `json:"prevClose"`
	MarketCap     float64 `json:"marketCap"`
	Timestamp     int64   `json:"timestamp"`
}

type MLPrediction struct {
	Symbol       string  `json:"symbol"`
	Direction    string  `json:"direction"`    // "buy", "sell", "neutral"
	Confidence   float64 `json:"confidence"`   // 0-100
	PriceTarget  float64 `json:"priceTarget"`
	Timeframe    string  `json:"timeframe"`
	ModelVersion string  `json:"modelVersion"`
	PredictedAt  int64   `json:"predictedAt"`
}

type CryptoScanResult struct {
	Symbol      string  `json:"symbol"`
	Signal      string  `json:"signal"`
	Score       int     `json:"score"`
	Reason      string  `json:"reason"`
	Volume24h   float64 `json:"volume24h"`
	PriceChange float64 `json:"priceChange"`
	RSI         float64 `json:"rsi"`
	ScanType    string  `json:"scanType"`
}

type NewListing struct {
	Symbol     string    `json:"symbol"`
	BaseAsset  string    `json:"baseAsset"`
	ListedAt   time.Time `json:"listedAt"`
	DetectedAt time.Time `json:"detectedAt"`
}

type FundamentalData struct {
	Ticker       string  `json:"ticker"`
	Name         string  `json:"name"`
	Sector       string  `json:"sector"`
	ReportType   string  `json:"reportType"`
	PE           float64 `json:"pe"`
	PB           float64 `json:"pb"`
	PS           float64 `json:"ps"`
	EVEbitda     float64 `json:"evEbitda"`
	DivYield     float64 `json:"divYield"`
	ROE          float64 `json:"roe"`
	ROA          float64 `json:"roa"`
	NetMargin    float64 `json:"netMargin"`
	Revenue      float64 `json:"revenue"`
	NetIncome    float64 `json:"netIncome"`
	EBITDA       float64 `json:"ebitda"`
	FCF          float64 `json:"fcf"`
	NetDebt      float64 `json:"netDebt"`
	NetDebtEBITDA float64 `json:"netDebtEbitda"`
	FairValue    float64 `json:"fairValue"`
	CurrentPrice float64 `json:"currentPrice"`
	Upside       float64 `json:"upside"`
	MarketCap    float64 `json:"marketCap"`
}
