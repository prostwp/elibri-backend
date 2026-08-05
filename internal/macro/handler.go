package macro

// handler.go — the read-side interface the REST handler depends on.
//
// Why an interface (rather than handing the api handler a *Store directly): the
// api package needs a unit test that doesn't run the stooq worker. By
// programming the handler against SnapshotReader, the test in
// internal/api/macro_handlers_test.go can pass a fake reader and exercise the
// nil-store / populated / weekend paths offline — the same pattern
// funding.LiquidationReader / whale.SnapshotReader use.
//
// Real callers do:  api.SetMacroStore(macro.NewStore())  (a process singleton
// wired once at boot; the worker writes to it, the handler reads it).
// Tests do:         the handler reads a fake implementing SnapshotReader.
//
// The interface is a strict read-only subset of *Store — the write-side methods
// (SetQuote/AddPoint/SetFnG) never leak into the HTTP path.

// SnapshotReader is the read surface the macro handler depends on.
// Implementations: *Store (production) and any test fake.
type SnapshotReader interface {
	// Latest returns the latest-per-symbol quote map (a copy).
	Latest() map[string]Quote
	// Correlation returns Pearson(symA,symB) over the rolling window, or nil
	// when there are too few overlapping points.
	Correlation(symA, symB string) *float64
	// PointCount reports the rolling-window size (for the window description).
	PointCount() int
	// FnG returns the last valid Fear & Greed read and whether one was ever set.
	FnG() (FnG, bool)
}

// Compile-time assertion: *Store satisfies SnapshotReader. Breaking the
// contract fails the build here rather than at the call site.
var _ SnapshotReader = (*Store)(nil)
