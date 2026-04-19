package news

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// LunarCrush API v4 — crypto social intelligence.
// Free tier: 200 req/min, 10k/day. Aggregates tweets + reddit + telegram from
// 13M+ accounts including crypto KOL/verified influencers.
// Signup: https://lunarcrush.com/developers
// Key env var: LUNARCRUSH_API_KEY
const lunarCrushURL = "https://lunarcrush.com/api4"

// lcPost mirrors the "posts" array returned by /api4/public/coins/:symbol/posts/v1.
type lcPost struct {
	ID          string  `json:"id"`
	PostType    string  `json:"post_type"`        // "tweet" | "reddit-post" | "news"
	PostTitle   string  `json:"post_title"`
	PostLink    string  `json:"post_link"`
	PostImage   string  `json:"post_image,omitempty"`
	PostCreated int64   `json:"post_created"`     // unix seconds
	Sentiment   float64 `json:"post_sentiment"`   // 1..5 (1=very bear, 5=very bull)
	Creator     struct {
		DisplayName string  `json:"creator_display_name"`
		Name        string  `json:"creator_name"`   // @handle
		Followers   float64 `json:"creator_followers"`
		Avatar      string  `json:"creator_avatar"`
	} `json:"creator,inline"`
	CreatorName        string  `json:"creator_name"`
	CreatorDisplayName string  `json:"creator_display_name"`
	CreatorFollowers   float64 `json:"creator_followers"`
	Interactions       float64 `json:"interactions_24h"`
}

type lcResponse struct {
	Data []lcPost `json:"data"`
}

// FetchLunarCrush pulls top social posts for a coin symbol.
// Symbol should be canonical ticker (BTC, ETH, SOL — not BTCUSDT).
// Returns empty slice (not error) when apiKey absent, so aggregator gracefully
// falls back to other sources.
func FetchLunarCrush(ctx context.Context, apiKey, symbol string, hours int) ([]Item, error) {
	if apiKey == "" || symbol == "" {
		return nil, nil
	}
	coin := strings.ToUpper(coinSlugFromSymbol(symbol))
	if coin == "" {
		return nil, nil
	}

	// /posts/v1 endpoint returns recent top-ranked posts per coin.
	url := fmt.Sprintf("%s/public/coins/%s/posts/v1", lunarCrushURL, coin)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Accept", "application/json")

	client := &http.Client{Timeout: 12 * time.Second}
	res, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("lunarcrush status %d", res.StatusCode)
	}

	var data lcResponse
	if err := json.NewDecoder(res.Body).Decode(&data); err != nil {
		return nil, err
	}

	cutoff := time.Now().Add(-time.Duration(hours) * time.Hour).Unix()
	out := make([]Item, 0, len(data.Data))
	for _, p := range data.Data {
		if p.PostCreated < cutoff {
			continue
		}
		// LC sentiment is 1..5 — map to [-1, +1].
		// 1 = strong bear → -1.0, 3 = neutral → 0, 5 = strong bull → +1.0
		sent := (p.Sentiment - 3.0) / 2.0
		if sent > 1 {
			sent = 1
		}
		if sent < -1 {
			sent = -1
		}

		headline := p.PostTitle
		if headline == "" {
			headline = "(no title)"
		}
		// Truncate long threads.
		if len(headline) > 200 {
			headline = headline[:197] + "..."
		}

		// Resolve creator handle — different LC versions use different fields.
		handle := p.CreatorName
		if handle == "" {
			handle = p.Creator.Name
		}
		followers := p.CreatorFollowers
		if followers == 0 {
			followers = p.Creator.Followers
		}

		// Mark as insider/KOL if >10k followers AND verified by high interactions.
		isInsider := followers >= 10000 && p.Interactions >= 100

		summary := fmt.Sprintf("@%s · %sK followers · %s interactions",
			handle,
			formatCount(followers),
			formatCount(p.Interactions),
		)

		out = append(out, Item{
			Source:       "lunarcrush-" + p.PostType, // e.g. "lunarcrush-tweet"
			Category:     "social",
			Headline:     headline,
			Summary:      summary,
			URL:          p.PostLink,
			PublishedAt:  time.Unix(p.PostCreated, 0),
			Sentiment:    sent,
			MentionsCoin: isInsider, // reuse field as "priority/verified" flag
		})
	}
	return out, nil
}

func formatCount(n float64) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", n/1_000_000)
	case n >= 1_000:
		return fmt.Sprintf("%.1fK", n/1_000)
	default:
		return fmt.Sprintf("%.0f", n)
	}
}
