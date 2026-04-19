package news

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Reddit hot-posts fetcher.
//
// ⚠ STATUS (2026): Reddit blocks unauthenticated requests from cloud/server
// IPs with HTTP 403 since July 2023. This fetcher will fail silently from
// most environments unless either:
//   1. Run from a residential IP (home machine, dev laptop) — may work
//   2. Reddit OAuth flow is added (TODO: register app at reddit.com/prefs/apps,
//      POST /api/v1/access_token with client_credentials grant, add Bearer token
//      to requests).
//
// Current behavior: if 403'd, returns empty slice (not error) so aggregator
// silently skips Reddit and continues with the other 5 sources.
//
// Subreddits mapped per coin for relevance. Posts with score ≥5000 and
// ≥200 comments → flagged as high-signal ("KOL" in UI).

const redditUserAgent = "ElibriFX/0.2 (https://github.com/prostwp/elibri-backend)"

// Debounce Reddit error logging: print once per status code per run.
var (
	redditLogOnce sync.Map
)

func logRedditOnce(status int, sub string) {
	key := fmt.Sprintf("%d", status)
	if _, loaded := redditLogOnce.LoadOrStore(key, true); loaded {
		return
	}
	if status == 403 {
		log.Printf("reddit: HTTP 403 (first seen on r/%s) — server IP blocked by Reddit since 2023. OAuth2 flow needed to enable social category.", sub)
	} else {
		log.Printf("reddit: HTTP %d on r/%s — skipping", status, sub)
	}
}

// coinSubreddits maps a coin ticker to the most active subs for its ecosystem.
// Covers top-20 we train on + general crypto subs as fallback.
var coinSubreddits = map[string][]string{
	"BTC":   {"Bitcoin", "CryptoCurrency"},
	"ETH":   {"ethereum", "ethtrader", "CryptoCurrency"},
	"SOL":   {"solana", "CryptoCurrency"},
	"XRP":   {"Ripple", "XRP", "CryptoCurrency"},
	"BNB":   {"binance", "CryptoCurrency"},
	"DOGE":  {"dogecoin", "CryptoCurrency"},
	"ADA":   {"cardano", "CryptoCurrency"},
	"AVAX":  {"Avax", "CryptoCurrency"},
	"DOT":   {"dot", "CryptoCurrency"},
	"MATIC": {"0xPolygon", "CryptoCurrency"},
}

// Known verified crypto-influencer Reddit handles (OPs whose posts move
// the needle). Hand-curated whitelist; backend flags these as "KOL".
var redditKOLs = map[string]bool{
	"vbuterin":      true, // Vitalik
	"nic__carter":   true,
	"aantonop":      true, // Andreas Antonopoulos
	"bitfinexed":    true,
	"cryptograffiti": true,
	"hasufl":        true,
	"AutoModerator": false, // explicit NO
}

type redditListing struct {
	Data struct {
		Children []struct {
			Data struct {
				ID          string  `json:"id"`
				Title       string  `json:"title"`
				SelfText    string  `json:"selftext"`
				Author      string  `json:"author"`
				Score       float64 `json:"score"`
				NumComments int     `json:"num_comments"`
				Created     float64 `json:"created_utc"`
				Permalink   string  `json:"permalink"`
				URL         string  `json:"url"`
				Subreddit   string  `json:"subreddit"`
				Stickied    bool    `json:"stickied"`
				LinkFlair   string  `json:"link_flair_text"`
			} `json:"data"`
		} `json:"children"`
	} `json:"data"`
}

// FetchReddit pulls hot posts from subs relevant to the coin.
// No API key required. Returns items tagged category="social".
func FetchReddit(ctx context.Context, symbol string, hours int) ([]Item, error) {
	coin := strings.ToUpper(coinSlugFromSymbol(symbol))
	subs, ok := coinSubreddits[coin]
	if !ok {
		subs = []string{"CryptoCurrency"} // fallback
	}

	cutoff := time.Now().Add(-time.Duration(hours) * time.Hour).Unix()
	seen := make(map[string]bool)
	out := make([]Item, 0, len(subs)*25)
	client := &http.Client{Timeout: 10 * time.Second}

	for _, sub := range subs {
		url := fmt.Sprintf("https://www.reddit.com/r/%s/hot.json?limit=25", sub)
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			continue
		}
		req.Header.Set("User-Agent", redditUserAgent)
		req.Header.Set("Accept", "application/json")

		res, err := client.Do(req)
		if err != nil || res == nil {
			continue
		}
		if res.StatusCode != http.StatusOK {
			logRedditOnce(res.StatusCode, sub)
			res.Body.Close()
			continue
		}
		var data redditListing
		decErr := json.NewDecoder(res.Body).Decode(&data)
		res.Body.Close()
		if decErr != nil {
			continue
		}

		for _, c := range data.Data.Children {
			p := c.Data
			if p.Stickied || int64(p.Created) < cutoff {
				continue
			}
			if seen[p.ID] {
				continue
			}
			seen[p.ID] = true

			// Skip low-signal posts (memes, <100 score).
			if p.Score < 100 {
				continue
			}

			fullText := p.Title + " " + p.SelfText
			sent := scoreSentiment(fullText)

			// "KOL" flag: whitelisted author OR very high engagement.
			isKOL := redditKOLs[p.Author] || (p.Score >= 5000 && p.NumComments >= 200)

			// Summary: author + engagement stats for UI.
			summary := fmt.Sprintf("u/%s · %s score · %d comments",
				p.Author, formatCount(p.Score), p.NumComments)

			headline := p.Title
			if len(headline) > 200 {
				headline = headline[:197] + "..."
			}

			out = append(out, Item{
				Source:       "reddit",
				Category:     "social",
				Headline:     headline,
				Summary:      summary,
				URL:          "https://www.reddit.com" + p.Permalink,
				PublishedAt:  time.Unix(int64(p.Created), 0),
				Sentiment:    sent,
				MentionsCoin: isKOL, // reused as "high-signal" flag in UI
			})
		}
		time.Sleep(300 * time.Millisecond) // respect Reddit rate limit
	}
	return out, nil
}
