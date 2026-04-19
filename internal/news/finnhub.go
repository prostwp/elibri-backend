package news

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

const finnhubBase = "https://finnhub.io/api/v1"

type finnhubRaw struct {
	Category string `json:"category"`
	Datetime int64  `json:"datetime"`
	Headline string `json:"headline"`
	ID       int64  `json:"id"`
	Source   string `json:"source"`
	Summary  string `json:"summary"`
	URL      string `json:"url"`
	Related  string `json:"related"`
}

// fetchFinnhubCategory pulls news from a specific category.
func fetchFinnhubCategory(ctx context.Context, apiKey, category string, hours int, defaultCat string) ([]Item, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("finnhub api key missing")
	}

	url := fmt.Sprintf("%s/news?category=%s&token=%s", finnhubBase, category, apiKey)
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)

	client := &http.Client{Timeout: 8 * time.Second}
	res, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("finnhub status %d", res.StatusCode)
	}

	var raw []finnhubRaw
	if err := json.NewDecoder(res.Body).Decode(&raw); err != nil {
		return nil, err
	}

	cutoff := time.Now().Add(-time.Duration(hours) * time.Hour)
	out := make([]Item, 0, len(raw))
	for _, r := range raw {
		ts := time.Unix(r.Datetime, 0)
		if ts.Before(cutoff) {
			continue
		}
		headline := strings.TrimSpace(r.Headline)
		if headline == "" {
			continue
		}
		cat := classifyCategory(headline, r.Summary, r.Category)
		if cat == "general" && defaultCat != "" {
			cat = defaultCat
		}
		out = append(out, Item{
			Source:      "finnhub",
			Category:    cat,
			Headline:    headline,
			Summary:     r.Summary,
			URL:         r.URL,
			PublishedAt: ts,
			Sentiment:   scoreSentiment(headline + " " + r.Summary),
		})
	}
	return out, nil
}

// FetchFinnhubGeneral — macro + geopolitics
func FetchFinnhubGeneral(ctx context.Context, apiKey string, hours int) ([]Item, error) {
	return fetchFinnhubCategory(ctx, apiKey, "general", hours, "")
}

// FetchFinnhubCrypto — crypto-specific news from Finnhub
func FetchFinnhubCrypto(ctx context.Context, apiKey string, hours int) ([]Item, error) {
	return fetchFinnhubCategory(ctx, apiKey, "crypto", hours, "crypto")
}

// classifyCategory maps Finnhub's generic "general" into our 4 domains
// (macro, geopolitics, regulation, adoption).
func classifyCategory(headline, summary, _ string) string {
	t := strings.ToLower(headline + " " + summary)

	geo := []string{"war", "sanction", "tariff", "china", "russia", "iran", "israel", "ukraine", "trump", "biden", "election", "middle east", "putin", "geopolit"}
	reg := []string{"sec ", "regulation", "lawsuit", "ban", "court", "ftc", "doj", "investigation"}
	adopt := []string{"etf", "partnership", "adoption", "launch", "approve", "list", "integrate"}
	macro := []string{"fed", "fomc", "cpi", "gdp", "inflation", "rate", "powell", "ecb", "boj", "unemployment", "nonfarm", "nfp", "yield", "treasury"}

	for _, k := range geo {
		if strings.Contains(t, k) {
			return "geopolitics"
		}
	}
	for _, k := range reg {
		if strings.Contains(t, k) {
			return "regulation"
		}
	}
	for _, k := range adopt {
		if strings.Contains(t, k) {
			return "adoption"
		}
	}
	for _, k := range macro {
		if strings.Contains(t, k) {
			return "macro"
		}
	}
	return "general"
}
