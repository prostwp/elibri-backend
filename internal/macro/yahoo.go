package macro

// yahoo.go — the Yahoo Finance chart API as a macro quoteSource (fallback for
// stooq since 2026-08-24, when stooq began serving a 404 page for quotes and a
// JavaScript anti-bot challenge for daily history).
//
// ── WHY THIS DUPLICATES internal/demobot/yahoo.go ──────────────────────────
// internal/demobot already has a working, browser-UA Yahoo chart client. It is
// deliberately NOT imported here. internal/macro currently imports NOTHING
// from this repository — it is a leaf package holding the backend's macro
// worker, store and wire types. internal/demobot is the Telegram bot binary's
// package: it pulls in go-telegram/bot, the Anthropic client, the Binance
// client and the bot's whole HTTP surface. Importing it would hang all of that
// off the API server's macro worker, invert the dependency direction (a data
// package depending on a presentation binary) and make the two release
// cadences one. Duplicating ~80 lines of a JSON parser is the cheaper trade,
// and the two copies answer different questions anyway: demobot wants OHLCV
// candles at 1h/4h/1d for indicators, this wants a last close + a previous
// close + a date-keyed daily series. If a third consumer appears, the right
// move is a shared pkg/yahoo — not a cross-import between these two.
//
// ── SYMBOL MAPPING ─────────────────────────────────────────────────────────
// Callers speak canonical macro ids (the Sym* constants, which are the stooq
// ids). Only this file knows the Yahoo tickers. Every mapping below was
// verified live against the v8 chart API on 2026-08-24 (HTTP 200 + bars).
//
// ── BAR DATES ARE EXCHANGE-LOCAL ───────────────────────────────────────────
// Yahoo's daily-bar timestamp is the session-open instant in the exchange's
// own timezone, NOT UTC midnight — measured live: ^GSPC 13:30Z (09:30 New
// York), ^TNX 12:20Z (07:20 Chicago), ^VIX 07:00Z (02:00 Chicago), GC=F and
// DX-Y.NYB 04:00Z (00:00 New York), BTC-USD 00:00Z.
//
// ⚠️ meta.gmtoffset MUST NOT be used to date the bars. It is ONE scalar
// describing the exchange's offset AT REQUEST TIME, while the bar timestamps
// themselves shift with daylight saving. Measured over a 1y window (2026-08-24):
//
//	DX-Y.NYB / GC=F : 166 bars at 04:00Z (EDT) + 83 bars at 05:00Z (EST)
//	^GSPC           : 166 bars at 13:30Z (EDT) + 85 bars at 14:30Z (EST)
//	meta.gmtoffset  : -14400 for the whole payload
//
// Applying that single offset to bars from the OTHER regime shifts their date
// by a day. Measured against per-bar timezone truth: a request made during EST
// (offset -18000) mis-dates 166 of 251 DX-Y.NYB and GC=F bars — and since the
// store keeps only the last 30 rows, for roughly six weeks after the November
// transition the ENTIRE stored window would be off by one day against BTC's
// UTC days, silently misaligning the correlation join.
//
// So the date is derived by converting each bar's own instant into the
// exchange's timezone (meta.exchangeTimezoneName), which is correct in every
// regime by construction. When that zone is absent or the host has no tzdata,
// we fall back to the raw UTC date — measured to agree with per-bar timezone
// truth on 251/251 bars for all six symbols in BOTH regimes, because every one
// of their session instants sits far enough inside the UTC day. The fallback is
// therefore a verified-correct degradation for today's symbol set, not a guess;
// a future symbol whose session starts within a few hours of UTC midnight needs
// the tz path, which is why it is the primary.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"sort"
	"time"
)

const (
	// defaultYahooChartBase is the public, keyless v8 chart endpoint.
	defaultYahooChartBase = "https://query1.finance.yahoo.com/v8/finance/chart/"

	// yahooUA — Yahoo answers 429 to requests without a browser-like
	// User-Agent (verified live, same finding as internal/demobot/yahoo.go).
	yahooUA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

	// yahooInterval is daily for both uses: the lamps are session-level and
	// the correlation window is explicitly a daily-close window.
	yahooInterval = "1d"

	// yahooQuoteRange / yahooDailyRange size the two request shapes. The quote
	// only needs the last bar plus one previous close, but a month of slack
	// survives a long holiday stretch; the daily window needs enough trading
	// days to fill the store's 30-close history (3mo ≈ 63 trading days on a
	// 5-day symbol, ≈ 93 on 24/7 BTC).
	yahooQuoteRange = "1mo"
	yahooDailyRange = "3mo"

	// maxPrevBarGap bounds the previous-close delta fallback to an ADJACENT
	// session. 4 days covers a Friday→Monday weekend plus a Monday holiday;
	// anything wider is a market break, and a move measured across it is not a
	// session change.
	maxPrevBarGap = 4 * 24 * time.Hour

	// maxYahooBody caps the chart response read. A 3mo/1d payload is ~10 KiB;
	// 1 MiB is generous slack for an unexpected shape.
	maxYahooBody = 1 << 20
)

// yahooTickers maps canonical macro symbol ids → Yahoo tickers. Verified live
// 2026-08-24; every entry returned HTTP 200 with bars.
//
//	dxy   dx.f      → DX-Y.NYB  (ICE US Dollar Index)
//	rates 10yusy.b  → ^TNX      (CBOE 10Y Treasury yield, in %)
//	vix   vi.f      → ^VIX      (CBOE Volatility Index, spot)
//	spx   ^spx      → ^GSPC     (S&P 500)
//	gold  xauusd    → GC=F      (COMEX gold front future; XAUUSD=X is dead
//	                             upstream — same finding as the demobot FX card)
//	btc   btcusd    → BTC-USD   (for the btc_* correlations; not a lamp)
var yahooTickers = map[string]string{
	SymDXY:   "DX-Y.NYB",
	SymRates: "^TNX",
	SymVIX:   "^VIX",
	SymSPX:   "^GSPC",
	SymGold:  "GC=F",
	SymBTC:   "BTC-USD",
}

// YahooTicker returns the Yahoo ticker for a canonical macro symbol id and
// whether one is mapped. Exported for the symbol-mapping table test.
func YahooTicker(symbol string) (string, bool) {
	t, ok := yahooTickers[symbol]
	return t, ok
}

// yahooSource implements quoteSource against the v8 chart API.
type yahooSource struct {
	base string     // "" → defaultYahooChartBase (overridden in tests)
	get  httpGetter // supplied by the Worker
}

// Name implements quoteSource.
func (y *yahooSource) Name() string { return SourceYahoo }

// yahooBar is one parsed daily bar.
type yahooBar struct {
	// Start is the bar's opening instant (UTC). Used as the quote's AsOf: it
	// is the timestamp of the bar the value came from, so it can UNDERSTATE
	// freshness by at most one session but never overstates it.
	Start time.Time
	// Date is the exchange-local session date ("2006-01-02") — the key the
	// correlation window aligns on across symbols.
	Date string
	// Open is the bar's own open; 0 when Yahoo padded it null (a bar with a
	// good close and a null open is still usable history, so it is kept).
	Open  float64
	Close float64
}

// yahooChartResp mirrors the parts of /v8/finance/chart this package needs.
// OHLC arrays are pointer-typed because Yahoo pads non-trading slots inside
// the arrays with nulls (verified live: DX-Y.NYB carried 13 null bars in a 3mo
// window, ^VIX carried 2).
type yahooChartResp struct {
	Chart struct {
		Result []struct {
			Meta struct {
				Symbol string `json:"symbol"`
				// ExchangeTimezoneName is the IANA zone for the bars, e.g.
				// "America/New_York". Used per bar so daylight saving is handled
				// per bar; see the ⚠️ note in the file header for why the
				// sibling meta.gmtoffset scalar must NOT be used instead.
				ExchangeTimezoneName string `json:"exchangeTimezoneName"`
			} `json:"meta"`
			Timestamp  []int64 `json:"timestamp"`
			Indicators struct {
				Quote []struct {
					Open  []*float64 `json:"open"`
					Close []*float64 `json:"close"`
				} `json:"quote"`
			} `json:"indicators"`
		} `json:"result"`
		Error *struct {
			Code        string `json:"code"`
			Description string `json:"description"`
		} `json:"error"`
	} `json:"chart"`
}

// ParseYahooChart converts a v8 chart payload into date-ascending daily bars.
//
// Honesty rules (mirroring the stooq parsers, plus the review's non-finite
// guards):
//   - a bar with a null or NON-FINITE close is DROPPED — the close is the
//     value we serve, and a padded slot is not data;
//   - a bar with a good close but a null/non-finite open is KEPT with Open 0.
//     Dropping it would throw away a real daily close over a missing delta
//     baseline; the caller degrades to the previous close, or to no delta;
//   - the session date comes from each bar's own instant converted into
//     meta.exchangeTimezoneName, NOT from meta.gmtoffset (see the file header);
//   - an empty/erroring/unparseable payload, or one whose bars are all padded,
//     is an ERROR — never an empty-but-successful read, so the caller falls
//     through to the next provider instead of recording "yahoo had nothing".
//
// Note on non-finite values: encoding/json cannot actually deliver NaN/±Inf
// (bare NaN is rejected and an overflowing literal fails the whole decode), so
// the isFinite guards are unreachable via the wire today. They are kept as the
// same cheap belt the stooq parsers wear — a future decode path (a hand-rolled
// scanner, a different codec) must not be able to slip a non-serialisable
// float into a lamp.
func ParseYahooChart(data []byte) ([]yahooBar, error) {
	var resp yahooChartResp
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, fmt.Errorf("yahoo chart: decode: %w", err)
	}
	if resp.Chart.Error != nil {
		return nil, fmt.Errorf("yahoo chart: %s — %s", resp.Chart.Error.Code, resp.Chart.Error.Description)
	}
	if len(resp.Chart.Result) == 0 {
		return nil, errors.New("yahoo chart: empty result")
	}
	r := resp.Chart.Result[0]
	if len(r.Indicators.Quote) == 0 {
		return nil, errors.New("yahoo chart: no quote block")
	}
	q := r.Indicators.Quote[0]

	// Yahoo returns parallel arrays; trust the SHORTEST so a truncated payload
	// can never index out of range or pair a close with the wrong timestamp.
	n := len(r.Timestamp)
	if len(q.Close) < n {
		n = len(q.Close)
	}
	loc := exchangeLocation(r.Meta.ExchangeTimezoneName)

	bars := make([]yahooBar, 0, n)
	for i := 0; i < n; i++ {
		if q.Close[i] == nil || !isFinite(*q.Close[i]) {
			continue // null-padded / non-finite slot — never invent a bar
		}
		start := time.Unix(r.Timestamp[i], 0).UTC()
		bar := yahooBar{
			Start: start,
			// Per-bar conversion: each instant carries its own DST state.
			Date:  start.In(loc).Format("2006-01-02"),
			Close: *q.Close[i],
		}
		if i < len(q.Open) && q.Open[i] != nil && isFinite(*q.Open[i]) {
			bar.Open = *q.Open[i]
		}
		bars = append(bars, bar)
	}
	if len(bars) == 0 {
		return nil, errors.New("yahoo chart: no usable bars (all null-padded)")
	}
	// Yahoo returns ascending order; sorting makes that an invariant we rely on
	// rather than an assumption (the last element must be the newest bar).
	// STABLE so that two bars sharing an instant keep their payload order —
	// FetchDaily's same-date de-dupe is "last wins" and must be deterministic.
	sort.SliceStable(bars, func(i, j int) bool { return bars[i].Start.Before(bars[j].Start) })
	return bars, nil
}

// exchangeLocation resolves the payload's IANA zone, falling back to UTC.
//
// time.LoadLocation needs the host tzdata (present on the Ubuntu deploy target
// and on macOS). If it is ever missing the fallback is UTC, which is measured
// to produce identical dates to the tz path for all six symbols in both DST
// regimes — a verified-correct degradation, never a wrong date today. It is
// still worth knowing about, so the caller logs nothing but the header
// documents it; a symbol whose session starts near UTC midnight would need
// tzdata present.
func exchangeLocation(name string) *time.Location {
	if name == "" {
		return time.UTC
	}
	loc, err := time.LoadLocation(name)
	if err != nil || loc == nil {
		return time.UTC
	}
	return loc
}

// FetchQuote implements quoteSource: the newest usable bar's close, with the
// delta baseline in Quote.Open and the bar's start instant as AsOf.
//
// ⚠️ DELTA BASELINE — deliberate deviation, read before "fixing" it. The lamp
// delta is defined repo-wide as the SESSION change (Close − Open) and the
// LampStatus thresholds (strongPct = 0.5) are tuned on that definition. So the
// baseline here is the bar's OWN open, exactly like stooq's quote row — NOT
// the previous close. The previous bar's close is used only as a FALLBACK when
// Yahoo padded the open null (verified to happen). Using previous-close as the
// primary would silently redefine delta_pct to a close-to-close change for
// whichever lamps happened to fall back to Yahoo, and the composite would then
// mix two different measurements — the exact thing the source field exists to
// prevent. With no open and no previous close, Open stays 0 and the handler
// emits delta_pct:null (VIX is unaffected: it is level-based).
func (y *yahooSource) FetchQuote(ctx context.Context, symbol string) (Quote, error) {
	bars, err := y.bars(ctx, symbol, yahooQuoteRange)
	if err != nil {
		return Quote{}, err
	}
	last := bars[len(bars)-1]
	q := Quote{
		Symbol: symbol,
		Price:  last.Close,
		AsOf:   last.Start,
		OK:     true,
		Source: SourceYahoo,
	}
	switch {
	case last.Open > 0:
		q.Open = last.Open
	case len(bars) >= 2 && last.Start.Sub(bars[len(bars)-2].Start) <= maxPrevBarGap:
		// Fallback ONLY across an adjacent session. Without the bound, a bar
		// sitting on the far side of a holiday gap would turn a multi-day move
		// into something the wire labels a single session's change — and
		// strongPct (0.5%) would judge it as one. Beyond the gap we leave
		// Open 0 and the lamp honestly reports no direction.
		q.Open = bars[len(bars)-2].Close
	}
	return q, nil
}

// FetchDaily implements quoteSource: the trailing daily closes, date-ascending,
// keyed by exchange-local session date.
//
// `now` is unused — the request window is a fixed trailing range rather than an
// explicit d1/d2 pair (Yahoo has no ranged-date parameter on this endpoint).
// The Worker applies the shared dailyMaxAgeDays recency guard to whatever comes
// back, so a source that ignored the window cannot smuggle ancient rows in as
// "the last 30 days".
func (y *yahooSource) FetchDaily(ctx context.Context, symbol string, _ time.Time) ([]DailyClose, error) {
	bars, err := y.bars(ctx, symbol, yahooDailyRange)
	if err != nil {
		return nil, err
	}
	out := make([]DailyClose, 0, len(bars))
	for _, b := range bars {
		// De-duplicate on the session date, last bar wins. Yahoo does not
		// normally repeat a date at interval=1d, but the correlation pairs on
		// this key and a duplicate would weight one day twice.
		if len(out) > 0 && out[len(out)-1].Date == b.Date {
			out[len(out)-1].Close = b.Close
			continue
		}
		out = append(out, DailyClose{Date: b.Date, Close: b.Close})
	}
	return out, nil
}

// bars performs one chart GET for a canonical symbol and parses it.
func (y *yahooSource) bars(ctx context.Context, symbol, rng string) ([]yahooBar, error) {
	ticker, ok := YahooTicker(symbol)
	if !ok {
		return nil, fmt.Errorf("yahoo: no ticker mapped for %q", symbol)
	}
	base := y.base
	if base == "" {
		base = defaultYahooChartBase
	}
	u := base + url.PathEscape(ticker) + "?interval=" + yahooInterval + "&range=" + rng

	// Bars are accepted from a 2xx only — the shared httpGetter errors on any
	// other status, so a 429/500 body that happens to parse can never be
	// treated as live data.
	body, err := y.get(ctx, u, maxYahooBody, map[string]string{"User-Agent": yahooUA})
	if err != nil {
		return nil, fmt.Errorf("yahoo %s (%s): %w", symbol, ticker, err)
	}
	bars, err := ParseYahooChart(body)
	if err != nil {
		return nil, fmt.Errorf("yahoo %s (%s): %w", symbol, ticker, err)
	}
	return bars, nil
}

// Compile-time assertion: yahooSource satisfies quoteSource.
var _ quoteSource = (*yahooSource)(nil)
