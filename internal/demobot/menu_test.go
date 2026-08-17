package demobot

import (
	"context"
	"strings"
	"testing"
	"time"
)

// stubExternalBases points every external data source at an instantly
// refused port so agent builders produce deterministic offline cards
// without touching the network.
func stubExternalBases(t *testing.T) {
	t.Helper()
	origYahoo, origBinance, origPremium := yahooChartBase, binanceKlinesBase, premiumIndexURL
	yahooChartBase = "http://127.0.0.1:1/v8/finance/chart/"
	binanceKlinesBase = "http://127.0.0.1:1/api/v3/klines"
	premiumIndexURL = "http://127.0.0.1:1/fapi/v1/premiumIndex?symbol="
	t.Cleanup(func() {
		yahooChartBase, binanceKlinesBase, premiumIndexURL = origYahoo, origBinance, origPremium
	})
}

// stripFooter removes the timestamped footer so two renders made
// milliseconds apart compare equal.
func stripFooter(s string) string {
	if i := strings.LastIndex(s, "\n<i>Analytics"); i >= 0 {
		return s[:i]
	}
	return s
}

// ── Menu card golden ─────────────────────────────────────────────────────────

func TestMenuCardGolden(t *testing.T) {
	wantText := "<b>AlphaVizor Demo Bot</b> 🛰\n" +
		"Pick an agent — no typing needed:\n" +
		"\n<i>Analytics, not financial advice.</i>"
	if got := menuText; got != wantText {
		t.Fatalf("menu text mismatch:\ngot:\n%s\nwant:\n%s", got, wantText)
	}

	kb := menuKeyboard()
	wantGrid := [][]struct{ label, data string }{
		{{"📊 Digest", "cmd|digest"}, {"🔥 Top", "cmd|top"}},
		{{"💱 FX", "cmd|fx"}, {"🌍 Macro", "cmd|macro"}},
		{{"🐋 Whale", "cmd|whale"}, {"💸 Funding", "cmd|funding"}},
		{{"⚡ Momentum", "cmd|momentum"}, {"📈 Trend", "cmd|trend"}},
		{{"🎯 Levels", "cmd|sr"}, {"🌪 Volatility", "cmd|vol"}},
		{{"🧮 Risk", "cmd|risk"}, {"ℹ️ Help", "cmd|help"}},
	}
	if len(kb.InlineKeyboard) != len(wantGrid) {
		t.Fatalf("grid rows: got %d, want %d", len(kb.InlineKeyboard), len(wantGrid))
	}
	for r, wantRow := range wantGrid {
		if len(kb.InlineKeyboard[r]) != 2 {
			t.Fatalf("row %d: got %d buttons, want 2", r, len(kb.InlineKeyboard[r]))
		}
		for c, want := range wantRow {
			b := kb.InlineKeyboard[r][c]
			if b.Text != want.label || b.CallbackData != want.data {
				t.Errorf("button [%d][%d]: got %q/%q, want %q/%q", r, c, b.Text, b.CallbackData, want.label, want.data)
			}
		}
	}
}

// /start and /menu serve the menu card; /help appends the same grid.
func TestMenuRouting(t *testing.T) {
	bot := newBotWithClient(newTGClientWithBase("T", "http://127.0.0.1:1/botT"), NewAgents(NewBackendClient("http://127.0.0.1:1")))
	ctx := context.Background()
	for _, cmd := range []string{"start", "menu"} {
		text, kb := bot.buildReply(ctx, cmd, nil)
		if text != menuText {
			t.Errorf("/%s must serve the menu card", cmd)
		}
		if kb == nil || len(kb.InlineKeyboard) != 6 {
			t.Errorf("/%s must carry the 6-row grid", cmd)
		}
	}
	text, kb := bot.buildReply(ctx, "help", nil)
	if text != helpText {
		t.Error("/help keeps its text")
	}
	if kb == nil || len(kb.InlineKeyboard) != 6 {
		t.Error("/help must carry the same grid appended")
	}
}

// ── cmd| routing: single dispatch path over all 12 grid commands ─────────────

func TestCmdCallbackRoutingTable(t *testing.T) {
	stubExternalBases(t)
	f, srv := newFakeTG(func(method string, body map[string]any, n int) (int, string) {
		switch method {
		case "sendMessage":
			return 200, okEnvelope(`{"message_id":1}`)
		default:
			return 200, okEnvelope(`true`)
		}
	})
	defer srv.Close()
	bot := testBot(srv.URL)
	ctx := context.Background()

	menuCmds := []string{"digest", "top", "fx", "macro", "whale", "funding", "momentum", "trend", "sr", "vol", "risk", "help"}
	for i, cmd := range menuCmds {
		// Reference: the typed-command path.
		wantText, _ := bot.buildReply(ctx, cmd, nil)

		// Button path: must land in the SAME dispatch and send a NEW message.
		sendsBefore := f.count("sendMessage")
		bot.handleCallback(ctx, &CallbackQuery{
			ID:      "cb",
			Data:    "cmd|" + cmd,
			Message: &Message{MessageID: 9, Chat: Chat{ID: int64(1000 + i)}}, // fresh chat → no cooldown interference
		})
		if got := f.count("sendMessage"); got != sendsBefore+1 {
			t.Fatalf("cmd|%s: got %d new messages, want exactly 1", cmd, got-sendsBefore)
		}
		sent := f.request("sendMessage", f.count("sendMessage")-1)
		gotText, _ := sent["text"].(string)
		if stripFooter(gotText) != stripFooter(wantText) {
			t.Errorf("cmd|%s: button reply diverged from typed reply:\nbutton:\n%s\ntyped:\n%s", cmd, gotText, wantText)
		}
		// Edits would replace the menu — buttons must always send anew.
		if f.count("editMessageText") != 0 {
			t.Fatalf("cmd|%s: menu was edited away", cmd)
		}
	}
}

// cmd|menu posts the menu card as a new message (never edits the card).
func TestCmdMenuCallback(t *testing.T) {
	f, srv := newFakeTG(func(method string, body map[string]any, n int) (int, string) {
		if method == "sendMessage" {
			return 200, okEnvelope(`{"message_id":2}`)
		}
		return 200, okEnvelope(`true`)
	})
	defer srv.Close()
	bot := testBot(srv.URL)
	bot.handleCallback(context.Background(), &CallbackQuery{
		ID: "cb", Data: "cmd|menu", Message: &Message{MessageID: 5, Chat: Chat{ID: 7}},
	})
	if f.count("sendMessage") != 1 {
		t.Fatalf("cmd|menu: got %d sends, want 1", f.count("sendMessage"))
	}
	sent := f.request("sendMessage", 0)
	if text, _ := sent["text"].(string); text != menuText {
		t.Errorf("cmd|menu must post the menu card, got %q", text)
	}
}

// ── Cooldown applies to button taps exactly like typed commands ──────────────

func TestCallbackCooldown(t *testing.T) {
	f, srv := newFakeTG(func(method string, body map[string]any, n int) (int, string) {
		if method == "sendMessage" {
			return 200, okEnvelope(`{"message_id":3}`)
		}
		return 200, okEnvelope(`true`)
	})
	defer srv.Close()
	bot := testBot(srv.URL)
	// Real 3s cooldown semantics; the map is fresh so first tap passes.
	cb := func(id string) *CallbackQuery {
		return &CallbackQuery{ID: id, Data: "cmd|help", Message: &Message{MessageID: 4, Chat: Chat{ID: 55}}}
	}
	ctx := context.Background()
	bot.handleCallback(ctx, cb("first"))
	bot.handleCallback(ctx, cb("second")) // immediate second tap → cooldown

	if got := f.count("sendMessage"); got != 1 {
		t.Fatalf("second tap inside cooldown must not send: got %d sends, want 1", got)
	}
	if got := f.count("answerCallbackQuery"); got != 2 {
		t.Fatalf("every tap must be acknowledged: got %d answers, want 2", got)
	}
	second := f.request("answerCallbackQuery", 1)
	if text, _ := second["text"].(string); text != cooldownMessage {
		t.Errorf("cooldown violation toast: got %q, want %q", text, cooldownMessage)
	}
}

// ── Risk example button ──────────────────────────────────────────────────────

func TestRiskExampleCallback(t *testing.T) {
	f, srv := newFakeTG(func(method string, body map[string]any, n int) (int, string) {
		if method == "sendMessage" {
			return 200, okEnvelope(`{"message_id":6}`)
		}
		return 200, okEnvelope(`true`)
	})
	defer srv.Close()
	bot := testBot(srv.URL)
	ctx := context.Background()

	// The alias expands to the exact documented example calculation.
	wantText, _ := bot.buildReply(ctx, keyRisk, []string{"10000", "1", "64000", "62500"})
	gotText, _ := bot.buildReply(ctx, "risk_example", nil)
	if stripFooter(gotText) != stripFooter(wantText) {
		t.Fatalf("risk_example must equal /risk 10000 1 64000 62500:\ngot:\n%s\nwant:\n%s", gotText, wantText)
	}
	if !strings.Contains(gotText, "LONG") {
		t.Errorf("example is a long setup (entry above stop), got: %s", gotText)
	}

	// The no-args usage card carries the one-tap example button.
	_, kb := bot.buildReply(ctx, keyRisk, nil)
	if kb == nil {
		t.Fatal("usage card needs a keyboard")
	}
	found := false
	for _, row := range kb.InlineKeyboard {
		for _, b := range row {
			if b.CallbackData == "cmd|risk_example" && strings.Contains(b.Text, "/risk 10000 1 64000 62500") {
				found = true
			}
		}
	}
	if !found {
		t.Fatalf("usage card must carry [Try: /risk 10000 1 64000 62500] → cmd|risk_example, got %+v", kb.InlineKeyboard)
	}

	// End-to-end через the button path.
	bot.handleCallback(ctx, &CallbackQuery{
		ID: "cb", Data: "cmd|risk_example", Message: &Message{MessageID: 8, Chat: Chat{ID: 66}},
	})
	if f.count("sendMessage") != 1 {
		t.Fatalf("risk_example callback: got %d sends, want 1", f.count("sendMessage"))
	}
	sent, _ := f.request("sendMessage", 0)["text"].(string)
	for _, want := range []string{"LONG", "64000", "62500"} {
		if !strings.Contains(sent, want) {
			t.Errorf("example calculation missing %q in: %s", want, sent)
		}
	}
}

// ── Native command menu on startup ───────────────────────────────────────────

func TestStartupSetsCommandMenu(t *testing.T) {
	f, srv := newFakeTG(func(method string, body map[string]any, n int) (int, string) {
		switch method {
		case "getMe":
			return 200, okEnvelope(`{"username":"testbot"}`)
		case "getUpdates":
			return 200, emptyUpdates
		default:
			return 200, okEnvelope(`true`)
		}
	})
	defer srv.Close()
	bot := testBot(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 400*time.Millisecond)
	defer cancel()
	if err := bot.Run(ctx); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := f.count("setMyCommands"); got != 1 {
		t.Fatalf("setMyCommands calls: got %d, want exactly 1", got)
	}
	body := f.request("setMyCommands", 0)
	cmds, _ := body["commands"].([]any)
	if len(cmds) != 12 {
		t.Fatalf("command menu entries: got %d, want 12", len(cmds))
	}
	// Every entry must have a command and a non-empty English description.
	for i, raw := range cmds {
		entry, _ := raw.(map[string]any)
		cmd, _ := entry["command"].(string)
		desc, _ := entry["description"].(string)
		if cmd == "" || desc == "" {
			t.Errorf("entry %d incomplete: %+v", i, entry)
		}
	}
}
