package funding

// worker_test.go — offline tests for the !forceOrder@arr parser (ParseForceOrder).
//
// The WS read loop itself (dial/backoff/keepalive) is integration-level and not
// unit-tested here — it is best-effort glue around the gorilla dialer. The
// load-bearing, deterministic logic is the event→Liquidation mapping, which is
// fully covered below.

import (
	"testing"
	"time"
)

// TestParseForceOrder_SellIsLongLiq verifies the canonical mapping from the
// spec example: S=SELL ⇒ long_liq, qty=z, price=ap, usd=price×qty, ts ISO.
func TestParseForceOrder_SellIsLongLiq(t *testing.T) {
	// Matches the confirmed wire shape (spec §"Confirmed source"). T below is
	// 2026-05-30T00:48:51Z in ms epoch.
	tsMs := time.Date(2026, 5, 30, 0, 48, 51, 0, time.UTC).UnixMilli()
	raw := []byte(`{"e":"forceOrder","E":1747000000000,"o":{"s":"BTCUSDT","S":"SELL","o":"LIMIT","f":"IOC","q":"0.014","p":"9910","ap":"9910","X":"FILLED","l":"0.014","z":"0.014","T":` +
		itoa(tsMs) + `}}`)

	liq, ok, err := ParseForceOrder(raw)
	if err != nil {
		t.Fatalf("ParseForceOrder returned error: %v", err)
	}
	if !ok {
		t.Fatalf("ParseForceOrder ok=false, want true for a forceOrder event")
	}
	if liq.Symbol != "BTCUSDT" {
		t.Errorf("Symbol = %q, want BTCUSDT", liq.Symbol)
	}
	if liq.Side != SideLongLiq {
		t.Errorf("Side = %q, want %q (SELL ⇒ a long was liquidated)", liq.Side, SideLongLiq)
	}
	if liq.Qty != 0.014 {
		t.Errorf("Qty = %v, want 0.014 (from z)", liq.Qty)
	}
	if liq.Price != 9910.0 {
		t.Errorf("Price = %v, want 9910 (from ap)", liq.Price)
	}
	wantUSD := 9910.0 * 0.014
	if liq.USDValue != wantUSD {
		t.Errorf("USDValue = %v, want %v (price×qty)", liq.USDValue, wantUSD)
	}
	wantTS := "2026-05-30T00:48:51Z"
	if liq.TS != wantTS {
		t.Errorf("TS = %q, want %q (o.T ms → UTC RFC3339)", liq.TS, wantTS)
	}
}

// TestParseForceOrder_BuyIsShortLiq verifies the inverse mapping: S=BUY ⇒
// short_liq.
func TestParseForceOrder_BuyIsShortLiq(t *testing.T) {
	raw := []byte(`{"e":"forceOrder","E":1747000000000,"o":{"s":"SOLUSDT","S":"BUY","ap":"150.5","z":"10","T":1747000000000}}`)
	liq, ok, err := ParseForceOrder(raw)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if !ok {
		t.Fatalf("ok=false, want true")
	}
	if liq.Side != SideShortLiq {
		t.Errorf("Side = %q, want %q (BUY ⇒ a short was liquidated)", liq.Side, SideShortLiq)
	}
	if liq.USDValue != 150.5*10 {
		t.Errorf("USDValue = %v, want %v", liq.USDValue, 150.5*10)
	}
}

// TestParseForceOrder_PriceFallbackToP verifies that an absent ap falls back to
// p, and an absent z falls back to l.
func TestParseForceOrder_PriceFallbackToP(t *testing.T) {
	raw := []byte(`{"e":"forceOrder","o":{"s":"ETHUSDT","S":"SELL","p":"3000","l":"2","T":1747000000000}}`)
	liq, ok, err := ParseForceOrder(raw)
	if err != nil || !ok {
		t.Fatalf("parse failed: ok=%v err=%v", ok, err)
	}
	if liq.Price != 3000.0 {
		t.Errorf("Price = %v, want 3000 (fallback ap→p)", liq.Price)
	}
	if liq.Qty != 2.0 {
		t.Errorf("Qty = %v, want 2 (fallback z→l)", liq.Qty)
	}
	if liq.USDValue != 6000.0 {
		t.Errorf("USDValue = %v, want 6000", liq.USDValue)
	}
}

// TestParseForceOrder_NonForceOrderSkipped verifies a non-forceOrder JSON frame
// (e.g. a subscription ack or a different event) parses cleanly but ok=false.
func TestParseForceOrder_NonForceOrderSkipped(t *testing.T) {
	cases := [][]byte{
		[]byte(`{"e":"aggTrade","s":"BTCUSDT"}`),    // wrong event type
		[]byte(`{"result":null,"id":1}`),            // subscription ack
		[]byte(`{"e":"forceOrder","o":{"S":"BUY"}}`), // forceOrder but no symbol
	}
	for i, raw := range cases {
		liq, ok, err := ParseForceOrder(raw)
		if err != nil {
			t.Errorf("case %d: unexpected error: %v", i, err)
		}
		if ok {
			t.Errorf("case %d: ok=true, want false (got %+v)", i, liq)
		}
	}
}

// TestParseForceOrder_MalformedJSON verifies invalid JSON returns an error
// (caller logs + skips, never crashes).
func TestParseForceOrder_MalformedJSON(t *testing.T) {
	_, ok, err := ParseForceOrder([]byte(`{not json`))
	if err == nil {
		t.Errorf("err = nil, want a JSON error for malformed input")
	}
	if ok {
		t.Errorf("ok = true, want false for malformed input")
	}
}

// itoa is a tiny int64→string helper kept local to the test so the test
// strings can interpolate a computed epoch without importing strconv at the
// top (keeps the parse test self-contained and readable).
func itoa(n int64) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
