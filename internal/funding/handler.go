package funding

// handler.go — the read-side interface the REST handler depends on.
//
// Why an interface (rather than handing the api handler a *Store directly):
// the api package needs a unit test that doesn't stand up a live WS. By
// programming the handler against LiquidationReader, the test in
// internal/api/funding_handlers_test.go can pass a fake reader and exercise the
// limit-clamping / empty-store paths offline — exactly the pattern
// whale.SnapshotReader uses.
//
// Real callers do:  api.SetFundingStore(funding.NewStore())  (a process
// singleton wired once at boot; the worker writes to it, the handler reads it).
// Tests do:         the handler reads a fakeReader implementing LiquidationReader.
//
// The interface is a strict read-only subset of *Store — the write-side method
// (Add) never leaks into the HTTP path.

// LiquidationReader is the read surface the funding handler depends on.
// Implementations: *Store (production) and any test fake.
type LiquidationReader interface {
	// Recent returns up to `limit` liquidations, newest-first.
	Recent(limit int) []Liquidation
	// Zones returns the aggregated price-magnet bands (top-N by total USD).
	Zones() []Zone
}

// Compile-time assertion: *Store satisfies LiquidationReader. Breaking the
// contract fails the build here rather than at the call site.
var _ LiquidationReader = (*Store)(nil)
