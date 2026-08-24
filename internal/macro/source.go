package macro

// source.go — the market-data provider abstraction behind the lamps.
//
// WHY THIS EXISTS (2026-08-24): stooq, the documented and until-now sole
// source, started answering the quote endpoint with a 404 HTML page and the
// daily endpoint with an anti-bot JavaScript challenge. Every lamp went null,
// regime went "unknown", and /agents/macro?asset=gold honestly reported "no
// tradfin data" — correct behaviour, zero customer value. A second provider
// (Yahoo's chart API, already proven in this repo for FX/gold candles) removes
// the single point of failure without touching the honesty rules: when BOTH
// providers come back empty the lamp is still not-ok, exactly as before.
//
// CONTRACT — every source speaks the CANONICAL macro symbol ids (the Sym*
// constants, which are the stooq ids and remain the Store's keys). A source is
// responsible for mapping them onto its own tickers; nothing outside this file
// and yahoo.go knows a Yahoo ticker exists. That keeps the Store, the handler,
// the correlations and every existing test keyed on one vocabulary.
//
// ORDER + ATTRIBUTION — sources are tried in MACRO_SOURCE_ORDER (default
// "stooq,yahoo"): stooq first because it is the documented source and may
// recover on its own; Yahoo picks up per symbol whenever stooq yields no
// usable row. Whichever provider produced a value is recorded on the Quote,
// on the daily history in the Store, and ships as the additive "source" field
// on each lamp and correlation — a value's provenance is never inferred, and
// never silently mixed.

import (
	"context"
	"strings"
	"time"
)

// Provider names. These are the exact strings that ship in the JSON "source"
// field and the only values MACRO_SOURCE_ORDER accepts.
const (
	SourceStooq = "stooq"
	SourceYahoo = "yahoo"

	// SourceMixed is emitted ONLY for a correlation whose two daily histories
	// came from different providers. It never appears on a lamp (a lamp has a
	// single value with a single provider) and is not accepted in
	// MACRO_SOURCE_ORDER.
	SourceMixed = "mixed"
)

// SourceOrderEnv is the operator override for the provider order. Comma
// separated, evaluated left to right, e.g.:
//
//	MACRO_SOURCE_ORDER=stooq,yahoo   (default — stooq first, Yahoo as fallback)
//	MACRO_SOURCE_ORDER=yahoo         (pin Yahoo; never touch stooq)
//	MACRO_SOURCE_ORDER=yahoo,stooq   (prefer Yahoo, keep stooq as the fallback)
//
// Unknown names are dropped with a log line; an empty or fully-unknown value
// falls back to defaultSourceOrder rather than leaving the worker with zero
// providers (a typo must not silently blind the lamps).
const SourceOrderEnv = "MACRO_SOURCE_ORDER"

// defaultSourceOrder is the shipped order: the documented source first, the
// fallback second.
var defaultSourceOrder = []string{SourceStooq, SourceYahoo}

// ParseSourceOrder parses a MACRO_SOURCE_ORDER value into a provider order.
// Entries are trimmed, lowercased and de-duplicated (first occurrence wins).
// Unknown entries are returned in `unknown` so the caller can log them.
// A value that yields no known provider returns defaultSourceOrder.
func ParseSourceOrder(v string) (order []string, unknown []string) {
	seen := make(map[string]bool, 2)
	for _, raw := range strings.Split(v, ",") {
		name := strings.ToLower(strings.TrimSpace(raw))
		if name == "" {
			continue
		}
		switch name {
		case SourceStooq, SourceYahoo:
			if !seen[name] {
				seen[name] = true
				order = append(order, name)
			}
		default:
			unknown = append(unknown, name)
		}
	}
	if len(order) == 0 {
		return append([]string(nil), defaultSourceOrder...), unknown
	}
	return order, unknown
}

// quoteSource is one market-data provider. Both methods speak canonical macro
// symbol ids and are expected to be best-effort: a transport failure, a
// non-200, an unparseable body or a body with no usable row is an error, and
// the caller moves on to the next provider in the order.
//
// Implementations MUST NOT invent data. "The provider answered but had nothing
// for this symbol" is an error (or an empty slice), never a zero-valued Quote
// dressed up as OK — the all-sources-failed path is what keeps the lamp not-ok.
type quoteSource interface {
	// Name is the provider name that ships in the JSON "source" field.
	Name() string

	// FetchQuote returns the latest snapshot for one canonical symbol.
	// The returned Quote carries Symbol, Price, Open (0 when unknown), AsOf
	// and Source; OK is false when the provider answered with an explicit
	// "no data" row that still carried a usable date (stooq's N/D case).
	FetchQuote(ctx context.Context, symbol string) (Quote, error)

	// FetchDaily returns the trailing daily closes for one canonical symbol,
	// date-ascending, keyed by the exchange-local session date. `now` is the
	// caller's clock (injectable for tests) and bounds the request window.
	FetchDaily(ctx context.Context, symbol string, now time.Time) ([]DailyClose, error)
}

// httpGetter is the HTTP dependency the sources share — supplied by the Worker
// so both providers reuse one client, one body cap policy and one context, and
// so tests can point a source at an httptest server without a live network.
type httpGetter func(ctx context.Context, u string, maxBody int64, headers map[string]string) ([]byte, error)

// CombineSources folds the provider names of a correlation's two legs into the
// single value that ships on Correlation.Source:
//
//	both known and equal     → that name
//	both known and different → SourceMixed (never silently one of the two)
//	either unknown           → "" (nothing to attribute)
//
// Exported because the correlation shape is assembled in internal/api.
func CombineSources(a, b string) string {
	if a == "" || b == "" {
		return ""
	}
	if a == b {
		return a
	}
	return SourceMixed
}
