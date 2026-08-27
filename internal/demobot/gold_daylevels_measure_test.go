package demobot

// gold_daylevels_measure_test.go — MEASUREMENT harness for the gold agent's
// "levels of the day" rule (gold spec, section 4.2). It exists to keep one
// magic constant honest: how deep does inside-day nesting actually run on
// gold, and therefore where does the unwind cap belong?
//
//	MEASURE_GOLD_DAYLEVELS=1 go test ./internal/demobot/ \
//	    -run TestMeasureGoldDayLevels -v
//
// Env-guarded: it hits Yahoo, so the normal suite stays offline.
//
// Definition used throughout: day D is INSIDE day D-1 when
// high(D) <= high(D-1) AND low(D) >= low(D-1). Nesting depth at D is the
// number of consecutive steps back that stay inside, i.e. how far the unwind
// in the spec would have to walk before it finds a range that actually bounds
// price.

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// insideOf reports whether bar a sits entirely inside bar b.
func insideOf(a, b types.OHLCVCandle) bool {
	return a.High <= b.High && a.Low >= b.Low
}

// nestDepth counts consecutive inside-relationships ending at index i:
// 0 = bar i is not inside bar i-1, 1 = inside one, 2 = inside two, …
func nestDepth(bars []types.OHLCVCandle, i int) int {
	d := 0
	for j := i; j > 0 && insideOf(bars[j], bars[j-1]); j-- {
		d++
	}
	return d
}

func TestMeasureGoldDayLevels(t *testing.T) {
	if os.Getenv("MEASURE_GOLD_DAYLEVELS") == "" {
		t.Skip("set MEASURE_GOLD_DAYLEVELS=1 (hits Yahoo)")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	bars, err := fetchYahooChart(ctx, "GC=F", "1d", "5y")
	if err != nil {
		t.Skipf("gold fetch failed: %v", err)
	}
	bars = dropUnclosedBars(bars, "1d", time.Now())
	if len(bars) < 2 {
		t.Skip("not enough bars")
	}

	fmt.Printf("\nGC=F daily · %d closed bars · %s .. %s\n", len(bars),
		time.Unix(bars[0].Time, 0).UTC().Format("2006-01-02"),
		time.Unix(bars[len(bars)-1].Time, 0).UTC().Format("2006-01-02"))

	depths := map[int]int{}
	maxDepth, maxAt := 0, 0
	n := 0
	for i := 1; i < len(bars); i++ {
		n++
		d := nestDepth(bars, i)
		depths[d]++
		if d > maxDepth {
			maxDepth, maxAt = d, i
		}
	}

	fmt.Println("\n=== how often the last closed day is an inside day ===")
	inside := n - depths[0]
	fmt.Printf("  not inside          %5d  %5.1f%%\n", depths[0], 100*float64(depths[0])/float64(n))
	fmt.Printf("  inside (any depth)  %5d  %5.1f%%\n", inside, 100*float64(inside)/float64(n))

	fmt.Println("\n=== nesting depth distribution ===")
	cum := 0
	for d := 0; d <= maxDepth; d++ {
		c := depths[d]
		if d > 0 {
			cum += c
		}
		line := fmt.Sprintf("  depth %d  %5d  %5.1f%%", d, c, 100*float64(c)/float64(n))
		if d > 0 {
			line += fmt.Sprintf("   (cumulative inside: %5.1f%%)", 100*float64(cum)/float64(n))
		}
		fmt.Println(line)
	}
	fmt.Printf("\n  deepest nest observed: %d, ending %s\n", maxDepth,
		time.Unix(bars[maxAt].Time, 0).UTC().Format("2006-01-02"))

	fmt.Println("\n=== what an unwind cap of K would cost ===")
	fmt.Println("  a cap of K resolves every case with depth <= K; deeper cases")
	fmt.Println("  fall through to \"range undefined\".")
	for _, k := range []int{1, 2, 3, 4, 5} {
		unresolved := 0
		for d := k + 1; d <= maxDepth; d++ {
			unresolved += depths[d]
		}
		fmt.Printf("  cap %d → unresolved on %d days (%.2f%% of all days)\n",
			k, unresolved, 100*float64(unresolved)/float64(n))
	}

	// The rule only fires when a direction would otherwise be issued, so the
	// share that matters is inside-days among days the regime confirms. That
	// intersection is reported once the structure gate question is settled;
	// printing it now would quote a number produced by a gate under review.
	fmt.Println()
}

// TestPreviewGoldCardLive renders the gold agent against the REAL sources and
// prints the card. It is a review aid, not an assertion suite: process stage 6
// asks a human to read the actual output, and edge states (weekend, compressed
// range, dead macro) only show up on live data.
//
//	MEASURE_GOLD_LIVE=1 go test ./internal/demobot/ \
//	    -run TestPreviewGoldCardLive -v
func TestPreviewGoldCardLive(t *testing.T) {
	if os.Getenv("MEASURE_GOLD_LIVE") == "" {
		t.Skip("set MEASURE_GOLD_LIVE=1 (hits Yahoo and the backend)")
	}
	backend := os.Getenv("DEMOBOT_BACKEND")
	if backend == "" {
		backend = "http://127.0.0.1:8080"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	ag := NewAgents(NewBackendClient(backend))
	c := ag.GoldCard(ctx)

	fmt.Printf("\n── live gold card · %s UTC ──\n\n", time.Now().UTC().Format("2006-01-02 15:04"))
	fmt.Printf("%s %s · %s\n", c.Emoji, c.Agent, c.Asset)
	fmt.Printf("%s\n", c.Verdict)
	for _, f := range c.Facts {
		fmt.Printf("  • %s\n", f)
	}
	fmt.Printf("\nstate=%q offline=%v status=%v\n", c.State, c.Offline, c.Status)
	fmt.Printf("data as of %s · %s\n", c.DataTime.UTC().Format(time.RFC3339), c.SourceNote)
	if c.Levels != nil {
		fmt.Printf("levels: %+v\n", c.Levels)
	}

	// The one thing worth asserting even here: no card may ship a NaN or an
	// infinity into a reader's screen.
	for _, f := range c.Facts {
		for _, bad := range []string{"NaN", "Inf", "+Inf", "-Inf"} {
			if strings.Contains(f, bad) {
				t.Errorf("non-finite value reached a fact: %q", f)
			}
		}
	}
	fmt.Println()
}
