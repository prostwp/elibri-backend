package news

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Alpha Vantage reports time_published in US/Eastern.
var avLocation = func() *time.Location {
	loc, err := time.LoadLocation("America/New_York")
	if err != nil {
		return time.UTC
	}
	return loc
}()

const alphaVantageURL = "https://www.alphavantage.co/query"

type avFeed struct {
	Feed []avItem `json:"feed"`
}

type avItem struct {
	Title                string    `json:"title"`
	URL                  string    `json:"url"`
	TimePublished        string    `json:"time_published"`
	Summary              string    `json:"summary"`
	Source               string    `json:"source"`
	OverallSentimentScore interface{} `json:"overall_sentiment_score"`
	OverallSentimentLabel string    `json:"overall_sentiment_label"`
	Topics []struct {
		Topic             string      `json:"topic"`
		RelevanceScore    interface{} `json:"relevance_score"`
	} `json:"topics"`
}

// FetchAlphaVantage pulls NEWS_SENTIMENT endpoint. Returns items in last `hours` window.
// Requires ALPHA_VANTAGE_API_KEY; returns nil (no items, no error) if key absent.
func FetchAlphaVantage(ctx context.Context, apiKey string, hours int) ([]Item, error) {
	if apiKey == "" {
		return nil, nil
	}

	// Topics: cryptocurrencies (coverage for BTC/ETH/etc) + macro + financial_markets
	u := fmt.Sprintf("%s?function=NEWS_SENTIMENT&topics=blockchain,financial_markets,economy_macro&apikey=%s&limit=50",
		alphaVantageURL, apiKey)

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	client := &http.Client{Timeout: 10 * time.Second}
	res, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("alphavantage status %d", res.StatusCode)
	}

	var data avFeed
	if err := json.NewDecoder(res.Body).Decode(&data); err != nil {
		return nil, err
	}

	cutoff := time.Now().Add(-time.Duration(hours) * time.Hour)
	out := make([]Item, 0, len(data.Feed))
	for _, it := range data.Feed {
		// AV time format: 20250418T143021 (Eastern Time)
		ts, err := time.ParseInLocation("20060102T150405", it.TimePublished, avLocation)
		if err != nil || ts.Before(cutoff) {
			continue
		}
		sent := toFloat(it.OverallSentimentScore)

		cat := "macro"
		for _, t := range it.Topics {
			tl := strings.ToLower(t.Topic)
			if strings.Contains(tl, "blockchain") || strings.Contains(tl, "crypto") {
				cat = "crypto"
				break
			}
			if strings.Contains(tl, "financial") || strings.Contains(tl, "markets") {
				cat = "macro"
			}
			if strings.Contains(tl, "economy") {
				cat = "macro"
			}
		}

		out = append(out, Item{
			Source:      "alphavantage",
			Category:    cat,
			Headline:    it.Title,
			Summary:     truncate(it.Summary, 200),
			URL:         it.URL,
			PublishedAt: ts,
			Sentiment:   sent,
		})
	}
	return out, nil
}

func toFloat(v interface{}) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case string:
		f, _ := strconv.ParseFloat(x, 64)
		return f
	}
	return 0
}
