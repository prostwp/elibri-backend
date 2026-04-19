package news

import (
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// rssFeed / rssChannel / rssItem match the common RSS 2.0 schema.
type rssFeed struct {
	XMLName xml.Name   `xml:"rss"`
	Channel rssChannel `xml:"channel"`
}

type rssChannel struct {
	Items []rssItem `xml:"item"`
}

type rssItem struct {
	Title       string `xml:"title"`
	Link        string `xml:"link"`
	Description string `xml:"description"`
	PubDate     string `xml:"pubDate"`
	Categories  []string `xml:"category"`
}

// FetchRSS pulls an RSS feed and maps items into the unified Item shape.
// sourceName appears in Item.Source. categoryHint defaults to "crypto".
func FetchRSS(ctx context.Context, feedURL, sourceName, categoryHint string, hours int) ([]Item, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, feedURL, nil)
	if err != nil {
		return nil, err
	}
	// Some feeds reject default UA.
	req.Header.Set("User-Agent", "Mozilla/5.0 ElibriFX/0.2")

	client := &http.Client{Timeout: 10 * time.Second}
	res, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s rss status %d", sourceName, res.StatusCode)
	}

	body, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, err
	}

	var feed rssFeed
	if err := xml.Unmarshal(body, &feed); err != nil {
		return nil, err
	}

	cutoff := time.Now().Add(-time.Duration(hours) * time.Hour)
	out := make([]Item, 0, len(feed.Channel.Items))
	for _, it := range feed.Channel.Items {
		ts := parseRSSDate(it.PubDate)
		if ts.IsZero() || ts.Before(cutoff) {
			continue
		}
		headline := strings.TrimSpace(it.Title)
		if headline == "" {
			continue
		}
		desc := stripHTML(it.Description)
		out = append(out, Item{
			Source:      sourceName,
			Category:    classifyFromCategories(it.Categories, categoryHint, headline, desc),
			Headline:    headline,
			Summary:     truncate(desc, 200),
			URL:         it.Link,
			PublishedAt: ts,
			Sentiment:   scoreSentiment(headline + " " + desc),
		})
	}
	return out, nil
}

func parseRSSDate(s string) time.Time {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}
	}
	// RSS uses RFC 1123 (with TZ name) or RFC 1123Z (with numeric TZ).
	layouts := []string{
		time.RFC1123Z,
		time.RFC1123,
		"Mon, 2 Jan 2006 15:04:05 MST",
		"Mon, 2 Jan 2006 15:04:05 -0700",
		time.RFC3339,
	}
	for _, l := range layouts {
		if t, err := time.Parse(l, s); err == nil {
			return t
		}
	}
	return time.Time{}
}

// classifyFromCategories: prefer RSS <category> tags if they map to our buckets;
// else fall back to keyword classifier.
func classifyFromCategories(cats []string, hint, headline, summary string) string {
	for _, c := range cats {
		lc := strings.ToLower(c)
		if strings.Contains(lc, "regulation") || strings.Contains(lc, "policy") || strings.Contains(lc, "legal") {
			return "regulation"
		}
		if strings.Contains(lc, "market") || strings.Contains(lc, "macro") || strings.Contains(lc, "fed") {
			return "macro"
		}
		if strings.Contains(lc, "geo") || strings.Contains(lc, "politic") {
			return "geopolitics"
		}
	}
	mapped := classifyCategory(headline, summary, "")
	if mapped == "general" && hint != "" {
		return hint
	}
	return mapped
}

func stripHTML(s string) string {
	// Simple strip — RSS descriptions often have <p>, <img>.
	var b strings.Builder
	inTag := false
	for _, r := range s {
		if r == '<' {
			inTag = true
			continue
		}
		if r == '>' {
			inTag = false
			continue
		}
		if !inTag {
			b.WriteRune(r)
		}
	}
	return strings.TrimSpace(b.String())
}

func truncate(s string, n int) string {
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n]) + "…"
}
