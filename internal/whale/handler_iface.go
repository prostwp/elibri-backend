package whale

// handler_iface.go — read-side interface used by the REST handler.
//
// Why an interface (rather than handing the handler a *Store directly): the
// api package needs a unit test that doesn't stand up Postgres. By programming
// the handler against SnapshotReader, the test in
// internal/api/whale_handlers_test.go can pass a fake reader and exercise the
// limit-clamping / empty-store paths offline.
//
// Real callers do:   handleWhaleFlowList(whale.NewStore(pool))
// Tests do:          handleWhaleFlowList(&fakeReader{...})
//
// The interface is a strict read-only subset of *Store — no write-side methods
// (InsertTransfer, UpsertSnapshot) leak into the HTTP path.

import "context"

// SnapshotReader is the read-side surface the whale-flow handler depends on.
// Implementations: *Store (production) and any test fake.
type SnapshotReader interface {
	LatestSnapshots(ctx context.Context) ([]Snapshot, error)
	RecentTransfers(ctx context.Context, limit int) ([]WhaleTransfer, error)
}

// Compile-time assertion: *Store satisfies SnapshotReader. Breaking the
// contract fails the build here rather than at the call-site.
var _ SnapshotReader = (*Store)(nil)
