package market

import (
	"math"
	"testing"

	"github.com/prostwp/elibri-backend/pkg/types"
)

// makeFlatCandles builds n daily candles with every close=100 and volume=10.
// Tests mutate the specific anchor indices they care about, leaving the rest
// flat so any computed change comes only from the mutated points.
func makeFlatCandles(n int) []types.OHLCVCandle {
	c := make([]types.OHLCVCandle, n)
	for i := range c {
		c[i] = types.OHLCVCandle{Time: int64(i), Open: 100, High: 100, Low: 100, Close: 100, Volume: 10}
	}
	return c
}

func approxEqual(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestPctChangeOverDays(t *testing.T) {
	tests := []struct {
		name   string
		closes []float64
		days   int
		want   float64
		wantOK bool
	}{
		{"up 1d", []float64{100, 110}, 1, 10, true},
		{"down 1d", []float64{100, 90}, 1, -10, true},
		{"3d window picks oldest", []float64{100, 105, 110, 120}, 3, 20, true},
		{"insufficient history", []float64{100, 110}, 2, 0, false},
		{"zero base price", []float64{0, 110}, 1, 0, false},
		{"days below one", []float64{100, 110}, 0, 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			candles := make([]types.OHLCVCandle, len(tt.closes))
			for i, c := range tt.closes {
				candles[i] = types.OHLCVCandle{Close: c}
			}
			got, ok := pctChangeOverDays(candles, tt.days)
			if ok != tt.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tt.wantOK)
			}
			if ok && !approxEqual(got, tt.want) {
				t.Errorf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRelativeStrength(t *testing.T) {
	tests := []struct {
		name     string
		sym, btc []float64
		days     int
		want     float64
		wantOK   bool
	}{
		{"coin outperforms btc", []float64{100, 110}, []float64{100, 104}, 1, 6, true},
		{"coin fades vs btc", []float64{100, 95}, []float64{100, 102}, 1, -7, true},
		{"flat tie", []float64{100, 100}, []float64{100, 100}, 1, 0, true},
		{"btc history too short", []float64{100, 110}, []float64{100}, 1, 0, false},
		{"sym history too short", []float64{100}, []float64{100, 104}, 1, 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			toCandles := func(cs []float64) []types.OHLCVCandle {
				out := make([]types.OHLCVCandle, len(cs))
				for i, c := range cs {
					out[i] = types.OHLCVCandle{Close: c}
				}
				return out
			}
			got, ok := relativeStrength(toCandles(tt.sym), toCandles(tt.btc), tt.days)
			if ok != tt.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tt.wantOK)
			}
			if ok && !approxEqual(got, tt.want) {
				t.Errorf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestVolumeGrowth(t *testing.T) {
	// Anchoring check: the LAST candle is the in-progress day and must be
	// ignored. We park an absurd volume there; if the code wrongly used it,
	// the result would blow up instead of matching the expected value.
	t.Run("growth ignores in-progress last day", func(t *testing.T) {
		c := makeFlatCandles(9) // window = idx 0..6 (vol 10), last closed = idx 7
		c[7].Volume = 20
		c[8].Volume = 99999
		got, ok := volumeGrowth(c)
		if !ok {
			t.Fatal("ok = false, want true")
		}
		if !approxEqual(got, 100) {
			t.Errorf("got %v, want 100", got)
		}
	})
	t.Run("shrink is negative", func(t *testing.T) {
		c := makeFlatCandles(9)
		c[7].Volume = 5
		got, ok := volumeGrowth(c)
		if !ok || !approxEqual(got, -50) {
			t.Errorf("got %v ok=%v, want -50 ok=true", got, ok)
		}
	})
	t.Run("insufficient history", func(t *testing.T) {
		if _, ok := volumeGrowth(makeFlatCandles(8)); ok {
			t.Error("ok = true, want false for <9 candles")
		}
	})
	t.Run("zero trailing average", func(t *testing.T) {
		c := makeFlatCandles(9)
		for i := 0; i < 7; i++ {
			c[i].Volume = 0
		}
		c[7].Volume = 5
		if _, ok := volumeGrowth(c); ok {
			t.Error("ok = true, want false when trailing average is zero")
		}
	})
}

func TestComputeMomentum(t *testing.T) {
	t.Run("full history populates all fields", func(t *testing.T) {
		sym := makeFlatCandles(31)
		sym[30].Close = 130 // +30% vs the flat-100 base at every window
		sym[29].Volume = 20 // last closed day doubles the trailing-7 avg
		btc := makeFlatCandles(31)
		btc[30].Close = 110 // +10% vs the flat-100 base at every window

		m := computeMomentum(sym, btc)

		for _, f := range []struct {
			name string
			got  *float64
			want float64
		}{
			{"RS1d", m.RS1d, 20},
			{"RS7d", m.RS7d, 20},
			{"RS30d", m.RS30d, 20},
			{"VolGrowthPct", m.VolGrowthPct, 100},
		} {
			if f.got == nil {
				t.Errorf("%s = nil, want %v", f.name, f.want)
			} else if !approxEqual(*f.got, f.want) {
				t.Errorf("%s = %v, want %v", f.name, *f.got, f.want)
			}
		}
	})

	t.Run("short history omits unavailable windows", func(t *testing.T) {
		sym := makeFlatCandles(5)
		btc := makeFlatCandles(5)
		m := computeMomentum(sym, btc)

		if m.RS1d == nil {
			t.Error("RS1d = nil, want present (5 candles is enough for 1d)")
		}
		if m.RS7d != nil {
			t.Error("RS7d should be nil with only 5 candles")
		}
		if m.RS30d != nil {
			t.Error("RS30d should be nil with only 5 candles")
		}
		if m.VolGrowthPct != nil {
			t.Error("VolGrowthPct should be nil with only 5 candles")
		}
	})
}
