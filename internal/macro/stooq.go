package macro

// stooq.go — stooq as a macro quoteSource.
//
// This is the DOCUMENTED source and stays first in the default order. The
// parsers themselves (ParseStooqCSV / ParseStooqDailyCSV) live in worker.go
// where they have always lived and where their tests reference them; this file
// only adds the URL construction + quoteSource plumbing that used to be inline
// in the Worker.
//
// ⚠️ Symbol ids are verbatim from the dnevnik — do NOT "fix" them. The
// intuitive caret forms (^vix / ^dxy / ^tnx / 10usy.b) return N/D; the working
// ones are vi.f / dx.f / 10yusy.b. Also: the multi-symbol batch (s=a,b,c)
// GARBLES when ^spx (a caret) is in the list, so we fetch ONE symbol per
// request.
//
// STATUS 2026-08-24: the quote endpoint answers HTTP 404 with an HTML "page
// does not exist" body and the daily endpoint answers with a JavaScript
// anti-bot challenge page. Both fail cleanly here (non-2xx → error;
// HTML → ParseStooqDailyCSV "unexpected header"), which is exactly what makes
// the per-symbol fallback to Yahoo kick in. stooq is kept first because it may
// recover, and because it is the source the scenario was specified against.

import (
	"context"
	"net/url"
	"time"
)

// stooqSource implements quoteSource against the public, keyless stooq CSV
// endpoints.
type stooqSource struct {
	base      string     // "" → defaultStooqBase
	dailyBase string     // "" → defaultStooqDailyBase
	get       httpGetter // supplied by the Worker
}

// Name implements quoteSource.
func (s *stooqSource) Name() string { return SourceStooq }

// FetchQuote GETs one stooq symbol and parses it into a Quote.
//
// A 200 with an N/D body is a SUCCESSFUL fetch returning Quote{OK:false} and a
// nil error — N/D is a valid "no data" answer, not a parse failure, and the row
// often still carries the last session's DATE (a real fact: "last time this
// symbol had data"). The Worker keeps that dated N/D quote as the fallback it
// stores if every other source also comes up empty, so the lamp's as_of keeps
// showing how stale the market is. Only a transport failure, a non-2xx or an
// unparseable body is an error.
func (s *stooqSource) FetchQuote(ctx context.Context, symbol string) (Quote, error) {
	base := s.base
	if base == "" {
		base = defaultStooqBase
	}
	u := base + "?s=" + url.QueryEscape(symbol) + "&f=sd2t2ohlcv&e=csv"

	body, err := s.get(ctx, u, maxQuoteBody, nil)
	if err != nil {
		return Quote{}, err
	}
	q, perr := ParseStooqCSV(body)
	if perr != nil {
		return Quote{}, perr
	}
	// stooq echoes the symbol uppercased (^SPX); keep the id we requested so the
	// rest of the pipeline keys on the canonical lowercase form.
	q.Symbol = symbol
	if q.OK {
		q.Source = SourceStooq
	}
	return q, nil
}

// FetchDaily GETs one symbol's ranged daily CSV and returns its closes
// (date-ascending). The dailyMaxAgeDays recency guard is applied by the Worker,
// shared with every other source.
func (s *stooqSource) FetchDaily(ctx context.Context, symbol string, now time.Time) ([]DailyClose, error) {
	base := s.dailyBase
	if base == "" {
		base = defaultStooqDailyBase
	}
	d2 := now.UTC()
	d1 := d2.AddDate(0, 0, -dailyFetchCalendarDays)
	u := base + "?s=" + url.QueryEscape(symbol) +
		"&d1=" + d1.Format("20060102") + "&d2=" + d2.Format("20060102") + "&i=d"

	body, err := s.get(ctx, u, maxDailyBody, nil)
	if err != nil {
		return nil, err
	}
	return ParseStooqDailyCSV(body)
}

// Compile-time assertion: stooqSource satisfies quoteSource.
var _ quoteSource = (*stooqSource)(nil)
