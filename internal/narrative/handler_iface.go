package narrative

// handler_iface.go — read-side interface used by the REST handler.
//
// Why an interface (rather than handing the handler a *Store directly):
// the api package needs a unit test that doesn't stand up Postgres. By
// programming the handler against SnapshotReader, the test in
// internal/api/narrative_handlers_test.go can pass a fake reader and
// exercise the limit / min_trend filtering paths offline.
//
// Real callers do:    handleNarrativesList(narrative.NewStore(pool))
// Tests do:           handleNarrativesList(&fakeReader{...})
//
// The interface is intentionally a one-method strict subset of *Store —
// no leakage of write-side methods (InsertMention, UpsertSnapshot) into
// the read-only HTTP path.

import "context"

// SnapshotReader is the read-side surface narrative handlers depend on.
// Implementations: *Store (production) and any test fake.
type SnapshotReader interface {
	LatestSnapshots(ctx context.Context) ([]Snapshot, error)
}

// Compile-time assertion that *Store satisfies SnapshotReader. If a future
// signature change to Store.LatestSnapshots breaks the contract, the build
// fails here rather than at the call-site in the api package.
var _ SnapshotReader = (*Store)(nil)
