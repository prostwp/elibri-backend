package funding

// worker.go — long-lived WebSocket consumer of Binance Futures'
// all-market force-order stream (`wss://fstream.binance.com/ws/!forceOrder@arr`).
//
// Lifecycle contract (mirrors whale.Worker.Run / narrative.Worker.Run):
//   Run(ctx) blocks until ctx is cancelled, then returns ctx.Err() so the
//   caller can distinguish "shutdown requested" from "fatal worker bug".
//
// Unlike whale (a time.Ticker poll), there is NO tick — events arrive push-
// style. The worker's job is to keep ONE socket alive forever:
//   dial → read loop → parse → store.Add → (on any error) backoff → redial.
//
// Best-effort by design: every dial/read error is logged, never fatal. If the
// WS is permanently unreachable the Store simply stays empty → the endpoint
// returns an empty feed → the frontend's liquidation section degrades to its
// own empty/error card while the other four features keep working.
//
// Two robustness mechanisms are mandatory (spec §9 gotchas):
//   1. Exponential backoff on reconnect (1s→2s→…cap 30s), reset on a clean
//      read — a flapping socket without backoff hammers Binance and risks an
//      IP ban.
//   2. Read deadline + pong handler keepalive — Binance sends ping frames and
//      drops idle/half-open sockets; gorilla auto-replies to pings, and we bump
//      the read deadline on every pong (and every message) so a silent
//      half-open socket trips the deadline and forces a reconnect rather than
//      blocking ReadMessage forever.

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"net"
	"strconv"
	"time"

	"github.com/gorilla/websocket"
)

const (
	// forceOrderWSURL is the public all-market liquidation stream. Single
	// combined stream, no auth, no key. curl/wscat-verified 2026-05-30.
	forceOrderWSURL = "wss://fstream.binance.com/ws/!forceOrder@arr"

	// backoffInitial / backoffMax bound the reconnect delay. Reset to initial
	// after a clean read so a brief blip recovers fast but a sustained outage
	// settles at a polite 30s cadence.
	backoffInitial = 1 * time.Second
	backoffMax     = 30 * time.Second

	// readDeadline is how long a single ReadMessage may block before the socket
	// is considered dead. Binance pushes liquidations irregularly (a quiet
	// market can go minutes between events), so this must comfortably exceed
	// the keepalive interval — we rely on PING/PONG, not data, to hold it open.
	// 70s ≈ 2× the server's ~30s ping cadence with slack.
	readDeadline = 70 * time.Second

	// clientPingInterval — how often WE proactively send a ping. Belt-and-
	// suspenders alongside Binance's server pings: if the server's pings stall
	// we still exercise the socket and detect a half-open connection via the
	// write error or the next missed deadline.
	clientPingInterval = 3 * time.Minute

	// dialTimeout caps the TCP+TLS+WS handshake.
	dialTimeout = 15 * time.Second

	// writeTimeout caps a single control-frame (ping) write.
	writeTimeout = 10 * time.Second
)

// Worker drives the persistent WS read loop. Store is required; Logger and URL
// are optional (URL defaults to the public Binance stream, Logger to
// log.Default()). The zero value is NOT usable — Store must be set.
type Worker struct {
	Store *Store
	// Logger optional; nil → log.Default().
	Logger *log.Logger
	// URL optional; "" → forceOrderWSURL. Overridable so a test can point the
	// worker at a local httptest WS server.
	URL string
}

// Run blocks until ctx is cancelled, maintaining a single live WS connection
// with reconnect + backoff + keepalive. Returns ctx.Err() on cancellation.
//
// The outer loop owns reconnect/backoff; runOnce owns one connection's
// lifetime (dial + read until error or cancel). runOnce returning nil means a
// clean ctx-cancel shutdown; a non-nil error means the socket died and we back
// off then redial.
func (w *Worker) Run(ctx context.Context) error {
	if w.Store == nil {
		return errors.New("funding.Worker.Run: Store is nil")
	}
	logger := w.logger()
	url := w.URL
	if url == "" {
		url = forceOrderWSURL
	}

	backoff := backoffInitial
	for {
		// Bail immediately if already cancelled (so a cancelled ctx never even
		// attempts a dial).
		if err := ctx.Err(); err != nil {
			return err
		}

		cleanRead, err := w.runOnce(ctx, url, logger)
		if ctx.Err() != nil {
			// Shutdown requested — return the ctx error regardless of how
			// runOnce ended.
			return ctx.Err()
		}
		if err != nil {
			logger.Printf("funding: WS connection error (will reconnect): %v", err)
		}
		// Reset backoff if this connection managed at least one clean read —
		// the endpoint is healthy, the drop was transient.
		if cleanRead {
			backoff = backoffInitial
		}

		// Backoff sleep, but stay cancellable.
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(backoff):
		}
		// Exponential growth with cap, applied AFTER the sleep so the next
		// failure waits longer.
		backoff *= 2
		if backoff > backoffMax {
			backoff = backoffMax
		}
	}
}

// runOnce dials the URL and reads until the connection errors or ctx is
// cancelled. Returns (cleanRead, err): cleanRead is true if at least one
// message was read successfully (used by the caller to reset backoff); err is
// the failure that ended the connection (nil on a clean ctx-cancel close).
func (w *Worker) runOnce(ctx context.Context, url string, logger *log.Logger) (cleanRead bool, err error) {
	dialer := websocket.Dialer{
		HandshakeTimeout: dialTimeout,
	}
	dialCtx, dialCancel := context.WithTimeout(ctx, dialTimeout)
	defer dialCancel()

	conn, resp, derr := dialer.DialContext(dialCtx, url, nil)
	if derr != nil {
		if resp != nil {
			_ = resp.Body.Close()
		}
		return false, derr
	}
	defer conn.Close()
	logger.Printf("funding: connected to %s", url)

	// Keepalive: set an initial read deadline and bump it on every pong AND on
	// every successful message. gorilla auto-sends a pong reply to server pings
	// inside ReadMessage, and invokes this PongHandler when WE receive a pong
	// to our client pings.
	_ = conn.SetReadDeadline(time.Now().Add(readDeadline))
	conn.SetPongHandler(func(string) error {
		return conn.SetReadDeadline(time.Now().Add(readDeadline))
	})
	// Also bump the deadline when the server pings us (default handler sends
	// the pong; we additionally extend the deadline so a ping-only quiet period
	// keeps the socket alive).
	conn.SetPingHandler(func(appData string) error {
		_ = conn.SetReadDeadline(time.Now().Add(readDeadline))
		// Mirror gorilla's default ping handler: reply with a pong, but never let
		// a transient write hiccup kill an otherwise-healthy socket. Swallow both
		// close-in-flight (ErrCloseSent) and a write-deadline timeout — if the
		// peer is genuinely dead, the read deadline + client ping detect it and
		// trigger a reconnect. Propagating a timeout here would abort a fine
		// connection on a momentarily full write buffer.
		err := conn.WriteControl(websocket.PongMessage, []byte(appData), time.Now().Add(writeTimeout))
		if errors.Is(err, websocket.ErrCloseSent) {
			return nil
		}
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			return nil
		}
		return err
	})

	// Proactive client-ping ticker — runs in a child goroutine so the read
	// loop below stays a tight ReadMessage call. The ticker goroutine exits
	// when readDone is closed (read loop ended) or ctx is cancelled.
	readDone := make(chan struct{})
	defer close(readDone)
	go w.pingLoop(ctx, conn, readDone)

	// A goroutine to force the blocking ReadMessage to unblock on ctx-cancel:
	// closing the connection makes ReadMessage return an error promptly. Guarded
	// by readDone so we don't close twice / race the defer conn.Close().
	go func() {
		select {
		case <-ctx.Done():
			_ = conn.Close()
		case <-readDone:
		}
	}()

	for {
		_, msg, rerr := conn.ReadMessage()
		if rerr != nil {
			// ctx-cancel path: report no error so the caller returns ctx.Err().
			if ctx.Err() != nil {
				return cleanRead, nil
			}
			return cleanRead, rerr
		}
		cleanRead = true
		_ = conn.SetReadDeadline(time.Now().Add(readDeadline))

		liq, ok, perr := ParseForceOrder(msg)
		if perr != nil {
			// A malformed / non-forceOrder frame (e.g. a subscription ack) is
			// logged at most as a debug curiosity, not an error — the combined
			// stream is clean, but we never crash on an unexpected frame.
			logger.Printf("funding: skipped unparseable WS frame: %v", perr)
			continue
		}
		if !ok {
			// Recognized JSON but not a forceOrder event (different "e"); skip.
			continue
		}
		w.Store.Add(liq)
	}
}

// pingLoop sends a periodic client ping until ctx is cancelled or the read loop
// signals done. A write error is logged and ends the loop; the read loop's
// deadline will then trip and trigger a reconnect.
func (w *Worker) pingLoop(ctx context.Context, conn *websocket.Conn, done <-chan struct{}) {
	ticker := time.NewTicker(clientPingInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-done:
			return
		case <-ticker.C:
			if err := conn.WriteControl(websocket.PingMessage, nil, time.Now().Add(writeTimeout)); err != nil {
				w.logger().Printf("funding: client ping failed (socket likely dead): %v", err)
				return
			}
		}
	}
}

// logger returns the Worker's logger or log.Default() when nil.
func (w *Worker) logger() *log.Logger {
	if w.Logger != nil {
		return w.Logger
	}
	return log.Default()
}

// --- parsing ---

// rawForceOrder is the wire shape of a single !forceOrder@arr event. Only the
// fields we map are declared; Binance may add more.
//
//	{"e":"forceOrder","E":1747000000000,"o":{
//	   "s":"BTCUSDT","S":"SELL","o":"LIMIT","f":"IOC",
//	   "q":"0.014","p":"9910","ap":"9910","X":"FILLED",
//	   "l":"0.014","z":"0.014","T":1747000000000}}
type rawForceOrder struct {
	EventType string `json:"e"`
	// EventTimeMs is declared (though unused for mapping) PURELY so encoding/json
	// has an EXACT match for the numeric "E" field. Without it, json's
	// case-insensitive fallback maps "E" (a number) onto the "e"-tagged
	// EventType (a string) and the whole frame fails to unmarshal. This bit in
	// the parse test — keep this field even though we read o.T, not E.
	EventTimeMs int64 `json:"E"`
	Order       struct {
		Symbol   string `json:"s"`
		Side     string `json:"S"`  // "SELL" (long liq) | "BUY" (short liq)
		AvgPrice string `json:"ap"` // average fill price
		Price    string `json:"p"`  // order price (fallback for ap)
		FilledZ  string `json:"z"`  // filled accumulated qty
		FilledL  string `json:"l"`  // last filled qty (fallback for z)
		TradeMs  int64  `json:"T"`  // transaction time (ms epoch)
	} `json:"o"`
}

// ParseForceOrder normalizes one raw !forceOrder@arr JSON frame into a
// Liquidation. Exported so worker_test.go can exercise the mapping directly.
//
// Returns (liq, ok, err):
//   - err != nil: the bytes weren't valid JSON for this shape.
//   - ok == false: valid JSON but not a forceOrder event ("e" != "forceOrder"),
//     or missing a symbol — caller skips it silently.
//   - ok == true: a fully-mapped Liquidation.
//
// Mapping (spec §5):
//
//	symbol    = o.s
//	side      = SELL → long_liq, BUY → short_liq
//	qty       = o.z (filled), fallback o.l
//	price     = o.ap (avg), fallback o.p
//	usd_value = price × qty   (0 when price is 0/absent)
//	ts        = o.T (ms epoch) → UTC RFC3339 string
func ParseForceOrder(data []byte) (Liquidation, bool, error) {
	var raw rawForceOrder
	if err := json.Unmarshal(data, &raw); err != nil {
		return Liquidation{}, false, err
	}
	if raw.EventType != "forceOrder" {
		return Liquidation{}, false, nil
	}
	o := raw.Order
	if o.Symbol == "" {
		// A forceOrder with no symbol is unusable; treat as "skip", not error.
		return Liquidation{}, false, nil
	}

	side := mapSide(o.Side)

	// qty: prefer accumulated filled (z); fall back to last filled (l).
	qty := parseFloat(o.FilledZ)
	if qty == 0 {
		qty = parseFloat(o.FilledL)
	}

	// price: prefer average fill (ap); fall back to order price (p).
	price := parseFloat(o.AvgPrice)
	if price == 0 {
		price = parseFloat(o.Price)
	}

	ts := ""
	if o.TradeMs > 0 {
		ts = time.UnixMilli(o.TradeMs).UTC().Format(time.RFC3339)
	} else {
		// No trade time → stamp receipt time so the entry is still datable for
		// the ring's age-pruning. Rare; Binance always sends T.
		ts = time.Now().UTC().Format(time.RFC3339)
	}

	return Liquidation{
		Symbol:   o.Symbol,
		Side:     side,
		Qty:      qty,
		Price:    price,
		USDValue: price * qty,
		TS:       ts,
	}, true, nil
}

// mapSide converts the forceOrder taker side to our liquidation side.
//
//	S=="SELL" → a long was force-closed → long_liq (▼)
//	S=="BUY"  → a short was force-closed → short_liq (▲)
//
// Any other value (shouldn't occur) defaults to long_liq so the event is still
// counted rather than dropped — but it's a single deterministic default, not a
// guess that fabricates direction.
func mapSide(s string) string {
	if s == "BUY" {
		return SideShortLiq
	}
	return SideLongLiq
}

// parseFloat parses a Binance numeric string ("0.014", "9910"); returns 0 on
// empty/invalid input (the caller's fallbacks + the "0 → —" UI rule handle it).
func parseFloat(s string) float64 {
	if s == "" {
		return 0
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0
	}
	return f
}
