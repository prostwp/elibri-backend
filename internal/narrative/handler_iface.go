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

import (
	"context"
	"time"
)

// SnapshotReader is the read-side surface narrative handlers depend on.
// Implementations: *Store (production) and any test fake.
type SnapshotReader interface {
	LatestSnapshots(ctx context.Context) ([]Snapshot, error)
}

// MentionReader is the read-side surface the per-narrative click-target
// endpoint depends on. Kept separate from SnapshotReader so the existing
// fakeReader in narrative_handlers_test.go doesn't have to grow a new
// method just to satisfy the snapshot-list handler.
type MentionReader interface {
	MentionsByNarrative(ctx context.Context, narrative string, since time.Time, limit int, minImportance int) ([]MentionRef, error)
}

// Compile-time assertions: *Store satisfies both interfaces. Breaking
// either contract fails the build here rather than at the call-site.
var (
	_ SnapshotReader = (*Store)(nil)
	_ MentionReader  = (*Store)(nil)
)
