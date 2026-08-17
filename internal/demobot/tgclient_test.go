package demobot

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"context"
)

// fakeTG is a scripted Telegram Bot API server. It records every request
// body per method and serves queued responses.
type fakeTG struct {
	mu       sync.Mutex
	requests map[string][]map[string]any // method → decoded request bodies
	handler  func(method string, body map[string]any, callNum int) (status int, resp string)
}

func newFakeTG(handler func(method string, body map[string]any, callNum int) (int, string)) (*fakeTG, *httptest.Server) {
	f := &fakeTG{requests: map[string][]map[string]any{}, handler: handler}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		parts := strings.Split(r.URL.Path, "/")
		method := parts[len(parts)-1]
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		f.mu.Lock()
		f.requests[method] = append(f.requests[method], body)
		n := len(f.requests[method])
		f.mu.Unlock()
		status, resp := f.handler(method, body, n)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(resp))
	}))
	return f, srv
}

func (f *fakeTG) count(method string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.requests[method])
}

func (f *fakeTG) request(method string, i int) map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	if i >= len(f.requests[method]) {
		return nil
	}
	return f.requests[method][i]
}

func okEnvelope(result string) string { return `{"ok":true,"result":` + result + `}` }

const emptyUpdates = `{"ok":true,"result":[]}`

// testBot builds a Bot wired to the fake server. Agents is nil-safe here
// because the backlog test never reaches an agent (that is the point).
func testBot(srvURL string) *Bot {
	tg := newTGClientWithBase("TESTTOKEN", srvURL+"/botTESTTOKEN")
	// Fast pacer so tests don't sleep on real send gaps.
	tg.pace = newPacer(time.Millisecond, 5*time.Millisecond)
	return newBotWithClient(tg, NewAgents(NewBackendClient("http://127.0.0.1:1")))
}

// ── Item 1: restart must not replay the backlog ──────────────────────────────

func TestFastForwardSkipsBacklog(t *testing.T) {
	backlog := `{"ok":true,"result":[{"update_id":100,"message":{"message_id":7,"chat":{"id":42},"text":"/help"}}]}`
	f, srv := newFakeTG(func(method string, body map[string]any, n int) (int, string) {
		switch method {
		case "getMe":
			return 200, okEnvelope(`{"username":"testbot"}`)
		case "getUpdates":
			if n == 1 {
				return 200, backlog // the stale backlog: offset=-1 probe sees the LAST old update
			}
			return 200, emptyUpdates
		default:
			return 200, okEnvelope(`{"message_id":1}`)
		}
	})
	defer srv.Close()

	bot := testBot(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 400*time.Millisecond)
	defer cancel()
	if err := bot.Run(ctx); err != nil {
		t.Fatalf("Run: %v", err)
	}

	// The old /help update must never have produced a reply.
	if got := f.count("sendMessage"); got != 0 {
		t.Fatalf("backlog was processed: %d sendMessage calls, want 0", got)
	}
	// First poll is the fast-forward probe: offset=-1, timeout=0, limit=1.
	probe := f.request("getUpdates", 0)
	if probe == nil {
		t.Fatal("no getUpdates recorded")
	}
	if off, _ := probe["offset"].(float64); off != -1 {
		t.Errorf("probe offset: got %v, want -1", probe["offset"])
	}
	if to, _ := probe["timeout"].(float64); to != 0 {
		t.Errorf("probe timeout: got %v, want 0", probe["timeout"])
	}
	if lim, _ := probe["limit"].(float64); lim != 1 {
		t.Errorf("probe limit: got %v, want 1", probe["limit"])
	}
	// Second poll must start AFTER the backlog: offset = 100+1.
	next := f.request("getUpdates", 1)
	if next == nil {
		t.Fatal("no second getUpdates recorded")
	}
	if off, _ := next["offset"].(float64); off != 101 {
		t.Errorf("post-fast-forward offset: got %v, want 101", next["offset"])
	}
}

// ── Item 2: 401/409 are fatal, not retried forever ───────────────────────────

func TestRunExitsOnUnauthorized(t *testing.T) {
	_, srv := newFakeTG(func(method string, body map[string]any, n int) (int, string) {
		if method == "getMe" {
			return 401, `{"ok":false,"error_code":401,"description":"Unauthorized"}`
		}
		return 401, `{"ok":false,"error_code":401,"description":"Unauthorized"}`
	})
	defer srv.Close()

	bot := testBot(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	start := time.Now()
	err := bot.Run(ctx)
	if err == nil {
		t.Fatal("Run must return a terminal error on 401")
	}
	if ctx.Err() != nil {
		t.Fatal("Run only stopped because the test deadline expired — 401 was retried, not fatal")
	}
	if !strings.Contains(err.Error(), "Unauthorized") {
		t.Errorf("error should carry the Telegram description, got: %v", err)
	}
	if time.Since(start) > 2*time.Second {
		t.Errorf("fatal 401 took %v — looks like it slept through retries", time.Since(start))
	}
}

func TestRunExitsOnConflict409(t *testing.T) {
	_, srv := newFakeTG(func(method string, body map[string]any, n int) (int, string) {
		if method == "getMe" {
			return 200, okEnvelope(`{"username":"testbot"}`)
		}
		return 409, `{"ok":false,"error_code":409,"description":"Conflict: terminated by other getUpdates request"}`
	})
	defer srv.Close()

	bot := testBot(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	err := bot.Run(ctx)
	if err == nil || ctx.Err() != nil {
		t.Fatalf("409 must be terminal, got err=%v ctxErr=%v", err, ctx.Err())
	}
}

// ── Item 3: 429 respected, one retry after retry_after ───────────────────────

func TestSendRetriesAfter429(t *testing.T) {
	f, srv := newFakeTG(func(method string, body map[string]any, n int) (int, string) {
		if method == "sendMessage" && n == 1 {
			return 429, `{"ok":false,"error_code":429,"description":"Too Many Requests","parameters":{"retry_after":1}}`
		}
		return 200, okEnvelope(`{"message_id":5}`)
	})
	defer srv.Close()

	tg := newTGClientWithBase("TESTTOKEN", srv.URL+"/botTESTTOKEN")
	tg.pace = newPacer(time.Millisecond, time.Millisecond)
	start := time.Now()
	id, err := tg.SendMessage(context.Background(), 42, "hello", nil)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("send after 429 retry: %v", err)
	}
	if id != 5 {
		t.Errorf("message id: got %d, want 5", id)
	}
	if elapsed < time.Second {
		t.Errorf("retry fired after %v — must wait retry_after (1s)", elapsed)
	}
	if got := f.count("sendMessage"); got != 2 {
		t.Errorf("attempts: got %d, want 2", got)
	}
}

func TestPacerSpacing(t *testing.T) {
	p := newPacer(20*time.Millisecond, 100*time.Millisecond)
	ctx := context.Background()

	// Same chat twice → per-chat gap governs.
	start := time.Now()
	p.wait(ctx, 1)
	p.wait(ctx, 1)
	if elapsed := time.Since(start); elapsed < 100*time.Millisecond {
		t.Errorf("same-chat spacing: %v, want >=100ms", elapsed)
	}

	// Different chats → only the global gap governs.
	start = time.Now()
	p.wait(ctx, 2)
	p.wait(ctx, 3)
	if elapsed := time.Since(start); elapsed < 20*time.Millisecond {
		t.Errorf("cross-chat spacing: %v, want >=20ms", elapsed)
	}
}

// ── Item 5: the token never appears in errors/logs ───────────────────────────

func TestErrorsRedactToken(t *testing.T) {
	const token = "123456:SECRETTOKENVALUE"
	// Port 1 refuses connections → transport error embedding the full URL.
	tg := newTGClientWithBase(token, "http://127.0.0.1:1/bot"+token)
	tg.pace = newPacer(time.Millisecond, time.Millisecond)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	_, err := tg.GetMe(ctx)
	if err == nil {
		t.Fatal("expected transport error")
	}
	msg := err.Error()
	if strings.Contains(msg, "SECRETTOKENVALUE") {
		t.Fatalf("token leaked into error: %s", msg)
	}
	if !strings.Contains(msg, "<token>") {
		t.Errorf("redaction marker missing from: %s", msg)
	}

	_, err = tg.SendMessage(ctx, 1, "x", nil)
	if err == nil || strings.Contains(err.Error(), "SECRETTOKENVALUE") {
		t.Fatalf("sendMessage leak: %v", err)
	}
}

// Sanity: the redactor must preserve the ErrNotModified sentinel.
func TestRedactPreservesSentinels(t *testing.T) {
	tg := newTGClientWithBase("TOK", "http://example.invalid/botTOK")
	if got := tg.redact(ErrNotModified); !errors.Is(got, ErrNotModified) {
		t.Fatal("redact must pass ErrNotModified through")
	}
	plain := fmt.Errorf("dial tcp: lookup api.telegram.org/botTOK failed")
	if got := tg.redact(plain); strings.Contains(got.Error(), "botTOK") {
		t.Fatalf("redact missed embedded token: %v", got)
	}
}
