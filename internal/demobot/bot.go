package demobot

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	pollTimeoutSec  = 50
	chatCooldown    = 3 * time.Second
	cooldownMessage = "Easy, one request per few seconds 🙂"
	digestBudget    = 25 * time.Second
)

// Bot wires the Telegram long-poll loop to the agent card builders.
type Bot struct {
	tg *TGClient
	ag *Agents

	mu      sync.Mutex
	lastReq map[int64]time.Time // per-chat cooldown (in-memory)
}

func NewBot(token string, ag *Agents) *Bot {
	return newBotWithClient(NewTGClient(token), ag)
}

// newBotWithClient exists for tests (injected fake-server client).
func newBotWithClient(tg *TGClient, ag *Agents) *Bot {
	return &Bot{
		tg:      tg,
		ag:      ag,
		lastReq: map[int64]time.Time{},
	}
}

// Run long-polls until ctx is canceled (SIGTERM → graceful exit). The
// getUpdates offset lives in memory; on every start the bot fast-forwards
// past the pending backlog, so a launchd KeepAlive restart never replays
// commands that arrived while the process was down (item 1). Fatal auth
// errors (401 bad token / 409 second poller) terminate the process instead
// of looking alive while dead (item 2).
func (b *Bot) Run(ctx context.Context) error {
	if user, err := b.tg.GetMe(ctx); err == nil {
		log.Printf("[demobot] long-poll starting as @%s", user)
	} else {
		if isFatalAuth(err) {
			return fmt.Errorf("telegram rejected the token on getMe: %w", err)
		}
		log.Printf("[demobot] warning: getMe failed (%v) — polling anyway", err)
	}

	// Fill the native "/" menu button. Non-fatal — the bot works without it.
	if err := b.tg.SetMyCommands(ctx, botCommands); err != nil {
		log.Printf("[demobot] setMyCommands failed (slash menu stays empty): %v", err)
	}

	offset, err := b.fastForward(ctx)
	if err != nil {
		return err
	}
	if ctx.Err() != nil {
		return nil
	}
	log.Printf("[demobot] backlog skipped, starting offset=%d", offset)

	for {
		select {
		case <-ctx.Done():
			log.Printf("[demobot] shutdown: %v", ctx.Err())
			return nil
		default:
		}
		updates, err := b.tg.GetUpdates(ctx, offset, pollTimeoutSec)
		if err != nil {
			if ctx.Err() != nil {
				log.Printf("[demobot] shutdown during poll")
				return nil
			}
			if isFatalAuth(err) {
				return fmt.Errorf("fatal telegram error — bad token or a second poller running: %w", err)
			}
			log.Printf("[demobot] getUpdates: %v (retrying in 3s)", err)
			select {
			case <-time.After(3 * time.Second):
			case <-ctx.Done():
				return nil
			}
			continue
		}
		for _, u := range updates {
			if u.UpdateID >= offset {
				offset = u.UpdateID + 1
			}
			go b.handleUpdate(ctx, u)
		}
	}
}

// fastForward resolves the starting offset: one past the newest pending
// update. The backlog is never processed — replying to hours-old commands
// after a restart is worse than dropping them. Retries transient errors;
// fatal auth errors bubble up.
func (b *Bot) fastForward(ctx context.Context) (int64, error) {
	for {
		last, err := b.tg.LastUpdateID(ctx)
		if err == nil {
			if last == 0 {
				return 0, nil
			}
			return last + 1, nil
		}
		if isFatalAuth(err) {
			return 0, fmt.Errorf("fatal telegram error during startup fast-forward: %w", err)
		}
		if ctx.Err() != nil {
			return 0, nil // clean shutdown before we ever started
		}
		log.Printf("[demobot] fast-forward failed: %v (retrying in 3s)", err)
		select {
		case <-time.After(3 * time.Second):
		case <-ctx.Done():
			return 0, nil
		}
	}
}

func (b *Bot) handleUpdate(ctx context.Context, u Update) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("[demobot] panic in update handler: %v", r)
		}
	}()
	switch {
	case u.CallbackQuery != nil:
		b.handleCallback(ctx, u.CallbackQuery)
	case u.Message != nil:
		b.handleMessage(ctx, u.Message)
	}
}

// allow implements the 3-second per-chat cooldown.
func (b *Bot) allow(chatID int64) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := time.Now()
	if last, ok := b.lastReq[chatID]; ok && now.Sub(last) < chatCooldown {
		return false
	}
	b.lastReq[chatID] = now
	return true
}

func (b *Bot) handleMessage(ctx context.Context, m *Message) {
	text := strings.TrimSpace(m.Text)
	if text == "" {
		return
	}
	if !b.allow(m.Chat.ID) {
		_, _ = b.tg.SendMessage(ctx, m.Chat.ID, cooldownMessage, nil)
		return
	}
	cmd, args := parseCommand(text)
	reply, kb := b.buildReply(ctx, cmd, args)
	if _, err := b.tg.SendMessage(ctx, m.Chat.ID, reply, kb); err != nil {
		log.Printf("[demobot] send to %d: %v", m.Chat.ID, err)
	}
}

func (b *Bot) handleCallback(ctx context.Context, q *CallbackQuery) {
	parts := strings.SplitN(q.Data, "|", 2)
	if len(parts) != 2 || q.Message == nil {
		_ = b.tg.AnswerCallbackQuery(ctx, q.ID, "", false)
		return
	}
	// The payload may carry the asset argument: "r|trend eurusd".
	action := parts[0]
	payload := strings.Fields(parts[1])
	if len(payload) == 0 {
		_ = b.tg.AnswerCallbackQuery(ctx, q.ID, "", false)
		return
	}
	cmd, args := payload[0], payload[1:]
	switch action {
	case "h": // static text, no external calls → no cooldown needed
		how := howTexts[cmd]
		if how == "" {
			how = "No description for this agent yet."
		}
		_ = b.tg.AnswerCallbackQuery(ctx, q.ID, how, true)
	case "cmd":
		// Grid buttons route through the SAME dispatch as typed commands and
		// always send a NEW message — the menu card stays put for reuse.
		if !b.allow(q.Message.Chat.ID) {
			_ = b.tg.AnswerCallbackQuery(ctx, q.ID, cooldownMessage, false)
			return
		}
		_ = b.tg.AnswerCallbackQuery(ctx, q.ID, "", false) // ack before slow builds
		reply, kb := b.buildReply(ctx, cmd, args)
		if _, err := b.tg.SendMessage(ctx, q.Message.Chat.ID, reply, kb); err != nil {
			log.Printf("[demobot] button send to %d: %v", q.Message.Chat.ID, err)
		}
	case "r":
		if !b.allow(q.Message.Chat.ID) {
			_ = b.tg.AnswerCallbackQuery(ctx, q.ID, cooldownMessage, false)
			return
		}
		reply, kb := b.buildReply(ctx, cmd, args)
		err := b.tg.EditMessageText(ctx, q.Message.Chat.ID, q.Message.MessageID, reply, kb)
		switch {
		case err == nil:
			_ = b.tg.AnswerCallbackQuery(ctx, q.ID, "Updated", false)
		case err == ErrNotModified:
			_ = b.tg.AnswerCallbackQuery(ctx, q.ID, "Already up to date", false)
		default:
			log.Printf("[demobot] edit in %d: %v", q.Message.Chat.ID, err)
			_ = b.tg.AnswerCallbackQuery(ctx, q.ID, "Refresh failed, try again", false)
		}
	default:
		_ = b.tg.AnswerCallbackQuery(ctx, q.ID, "", false)
	}
}

// parseCommand splits "/momentum@SomeBot extra" → ("momentum", ["extra"]).
// Non-command text maps to the empty command (→ help hint).
func parseCommand(text string) (string, []string) {
	fields := strings.Fields(text)
	if len(fields) == 0 || !strings.HasPrefix(fields[0], "/") {
		return "", nil
	}
	cmd := strings.TrimPrefix(fields[0], "/")
	if at := strings.IndexByte(cmd, '@'); at >= 0 {
		cmd = cmd[:at]
	}
	return strings.ToLower(cmd), fields[1:]
}

// ── Zero-typing navigation ───────────────────────────────────────────────────

// botCommands fills the native Telegram "/" menu (setMyCommands). Exactly 14
// entries: /start is the platform's own entry point and needs no slot;
// /help lost its grid button to 📰 News, so it lives here (and stays a typed
// command), alongside the new /news.
var botCommands = []BotCommand{
	{Command: "menu", Description: "Button navigation"},
	{Command: "digest", Description: "All agents, prioritized"},
	{Command: "top", Description: "Top signal right now"},
	{Command: "news", Description: "Narrative radar + AI idea"},
	{Command: "fx", Description: "Forex overview"},
	{Command: "macro", Description: "Risk-on/off regime"},
	{Command: "whale", Description: "Large BTC transfers"},
	{Command: "funding", Description: "Funding pressure & liquidations"},
	{Command: "momentum", Description: "RSI/MACD reads"},
	{Command: "trend", Description: "Trend state machine"},
	{Command: "sr", Description: "Support/resistance levels"},
	{Command: "vol", Description: "Volatility expansion check"},
	{Command: "risk", Description: "Position-size calculator"},
	{Command: "help", Description: "Command list & usage"},
}

const menuText = "<b>AlphaVizor Demo Bot</b> 🛰\n" +
	"Pick an agent — no typing needed:\n" +
	"\n<i>Analytics, not financial advice.</i>"

// menuKeyboard is the 6×2 inline grid served on /start, /menu and /help.
func menuKeyboard() *InlineKeyboardMarkup {
	btn := func(label, cmd string) InlineKeyboardButton {
		return InlineKeyboardButton{Text: label, CallbackData: "cmd|" + cmd}
	}
	return &InlineKeyboardMarkup{InlineKeyboard: [][]InlineKeyboardButton{
		{btn("📊 Digest", keyDigest), btn("🔥 Top", keyTop)},
		{btn("💱 FX", keyFX), btn("🌍 Macro", keyMacro)},
		{btn("🐋 Whale", keyWhale), btn("💸 Funding", keyFunding)},
		{btn("⚡ Momentum", keyMomentum), btn("📈 Trend", keyTrend)},
		{btn("🎯 Levels", keySR), btn("🌪 Volatility", keyVol)},
		{btn("🧮 Risk", keyRisk), btn("📰 News", keyNews)},
	}}
}

func menuButton() InlineKeyboardButton {
	return InlineKeyboardButton{Text: "◀️ Menu", CallbackData: "cmd|menu"}
}

// riskExampleArgs backs the one-tap [Try: …] button and the no-args example.
var riskExampleArgs = []string{"10000", "1", "64000", "62500"}

// cardKeyboard builds the standard card rows: [Refresh | How it works] and a
// [Menu] row so every card can route back to the grid.
func cardKeyboard(cmd string, withRefresh bool) *InlineKeyboardMarkup {
	row := []InlineKeyboardButton{}
	if withRefresh {
		row = append(row, InlineKeyboardButton{Text: "🔄 Refresh", CallbackData: "r|" + cmd})
	}
	row = append(row, InlineKeyboardButton{Text: "ℹ️ How it works", CallbackData: "h|" + cmd})
	return &InlineKeyboardMarkup{InlineKeyboard: [][]InlineKeyboardButton{row, {menuButton()}}}
}

func (b *Bot) buildReply(ctx context.Context, cmd string, args []string) (string, *InlineKeyboardMarkup) {
	switch cmd {
	case "start", "menu":
		return menuText, menuKeyboard()
	case "help":
		return helpText, menuKeyboard()
	case "risk_example":
		// One-tap example — expands to the exact documented calculation and
		// re-enters the normal /risk handler (single dispatch path).
		return b.buildReply(ctx, keyRisk, riskExampleArgs)
	case keyMacro:
		card, _ := b.ag.MacroCard(ctx)
		return card.RenderHTML(), cardKeyboard(keyMacro, true)
	case keyWhale:
		return b.ag.WhaleCard(ctx).RenderHTML(), cardKeyboard(keyWhale, true)
	case keyFunding:
		return b.ag.FundingCard(ctx).RenderHTML(), cardKeyboard(keyFunding, true)
	case keyFX:
		return b.ag.FXCard(ctx).RenderHTML(), cardKeyboard(keyFX, true)
	case keyNews:
		return b.ag.NewsCard(ctx).RenderHTML(), cardKeyboard(keyNews, true)
	case keyMomentum:
		return b.momentumReply(ctx, args)
	case keyTrend:
		return b.assetReply(ctx, cmd, args, func(spec assetSpec) Card {
			return b.ag.TrendCard(ctx, spec)
		})
	case keySR:
		return b.assetReply(ctx, cmd, args, func(spec assetSpec) Card {
			return b.ag.SRCard(ctx, spec)
		})
	case keyVol:
		return b.assetReply(ctx, cmd, args, func(spec assetSpec) Card {
			return b.ag.VolCard(ctx, spec)
		})
	case keyRisk:
		return b.riskReply(args)
	case keyDigest:
		return b.digestReply(ctx), cardKeyboard(keyDigest, true)
	case keyTop:
		return b.topReply(ctx), cardKeyboard(keyTop, true)
	default:
		return "Unknown command. Try /help for the agent list.", nil
	}
}

// momentumReply implements B1's /momentum command grammar (HTTP parity):
//
//	/momentum                  → default multi-asset card (unchanged)
//	/momentum eurusd           → single-asset card (unchanged)
//	/momentum eurusd 1d        → single asset re-based on the timeframe
//	/momentum btc,eurusd [1d]  → user scan: comma list + optional tf token
//	/momentum 1d               → default trio re-based on the timeframe
//
// Any token matching a known timeframe (1h/4h/1d) is the tf; the rest join
// into the asset list (so "btc eth" and "btc,eth" both scan). Refresh re-runs
// the exact same scan via the callback payload.
func (b *Bot) momentumReply(ctx context.Context, args []string) (string, *InlineKeyboardMarkup) {
	tf := ""
	var assetTokens []string
	for _, tok := range args {
		low := strings.ToLower(strings.TrimSpace(tok))
		if low == "" {
			continue
		}
		if momentumTFs[low] {
			if tf != "" && tf != low {
				return esc(fmt.Sprintf("one timeframe per scan — got %q and %q", tf, low)), nil
			}
			tf = low
			continue
		}
		assetTokens = append(assetTokens, low)
	}
	listExpr := strings.Join(assetTokens, ",")
	kbCmd := keyMomentum
	if listExpr != "" {
		kbCmd += " " + listExpr
	}
	if tf != "" {
		kbCmd += " " + tf
	}
	switch {
	case listExpr == "" && tf == "":
		return b.ag.MomentumCard(ctx).RenderHTML(), cardKeyboard(keyMomentum, true)
	case listExpr == "":
		return b.ag.MomentumScanCard(ctx, defaultMomentumKeys, tf).RenderHTML(), cardKeyboard(kbCmd, true)
	case !strings.Contains(listExpr, ","):
		spec, err := resolveAsset(listExpr)
		if err != nil {
			return esc(err.Error()), nil
		}
		spec, err = specWithTF(spec, tf)
		if err != nil {
			return esc(err.Error()), nil
		}
		return b.ag.MomentumAssetCard(ctx, spec).RenderHTML(), cardKeyboard(kbCmd, true)
	default:
		keys, err := parseAssetList(listExpr)
		if err != nil {
			return esc(err.Error()), nil
		}
		return b.ag.MomentumScanCard(ctx, keys, tf).RenderHTML(), cardKeyboard(kbCmd, true)
	}
}

// assetReply resolves an optional asset argument (default BTC), builds the
// card and wires the Refresh callback so it re-runs with the SAME asset.
func (b *Bot) assetReply(_ context.Context, cmd string, args []string, build func(assetSpec) Card) (string, *InlineKeyboardMarkup) {
	arg := ""
	if len(args) > 0 {
		arg = args[0]
	}
	spec, err := resolveAsset(arg)
	if err != nil {
		return esc(err.Error()), nil
	}
	kbCmd := cmd
	if arg != "" {
		kbCmd = cmd + " " + strings.ToLower(arg)
	}
	return build(spec).RenderHTML(), cardKeyboard(kbCmd, true)
}

func (b *Bot) riskReply(args []string) (string, *InlineKeyboardMarkup) {
	if len(args) == 0 {
		// Usage card: worked example + a one-tap button running it for real.
		kb := &InlineKeyboardMarkup{InlineKeyboard: [][]InlineKeyboardButton{
			{{Text: "Try: /risk 10000 1 64000 62500", CallbackData: "cmd|risk_example"}},
			{{Text: "ℹ️ How it works", CallbackData: "h|" + keyRisk}},
			{menuButton()},
		}}
		return b.ag.RiskCard([]float64{10000, 1, 64000, 62500}, true, nil).RenderHTML(), kb
	}
	if len(args) != 4 {
		return b.ag.RiskCard(nil, false, fmt.Errorf("expected 4 numbers, got %d", len(args))).RenderHTML(), cardKeyboard(keyRisk, false)
	}
	vals := make([]float64, 4)
	for i, tok := range args {
		v, err := parseMoney(tok)
		if err != nil {
			return b.ag.RiskCard(nil, false, err).RenderHTML(), cardKeyboard(keyRisk, false)
		}
		vals[i] = v
	}
	return b.ag.RiskCard(vals, false, nil).RenderHTML(), cardKeyboard(keyRisk, false)
}

// parseMoney parses one user-typed number, tolerating $ / % decoration and
// thousands commas — shared by the /risk command and the HTTP risk endpoint.
func parseMoney(tok string) (float64, error) {
	clean := strings.Trim(strings.ReplaceAll(tok, ",", ""), "$%")
	v, err := strconv.ParseFloat(clean, 64)
	if err != nil {
		return 0, fmt.Errorf("%q is not a number", tok)
	}
	return v, nil
}

// gathered is one concurrent sweep across every agent + context sources.
type gathered struct {
	cards   map[string]Card
	regime  string
	fx      []fxRead // /fx one-liners for the digest FX block
	fxAnyOK bool
	extras  []string // narrative / mood one-liners (best-effort)
	// Raw feed values for the ALPHAVIZOR AI payload — the extras above are
	// Telegram-escaped render strings, the AI prompt needs the raw data.
	topNarr *NarrativeSnapshot
	mood    string
}

// digestOrder is the render order of the one-liner section.
var digestOrder = []string{keyMacro, keyWhale, keyFunding, keyMomentum, keyTrend, keySR, keyVol}

// gather lives on Agents (not Bot) so the HTTP API runs the exact same sweep.
func (a *Agents) gather(ctx context.Context) gathered {
	ctx, cancel := context.WithTimeout(ctx, digestBudget)
	defer cancel()

	g := gathered{cards: map[string]Card{}}
	var mu sync.Mutex
	var wg sync.WaitGroup
	put := func(key string, c Card) {
		mu.Lock()
		g.cards[key] = c
		mu.Unlock()
	}
	run := func(f func()) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			f()
		}()
	}
	run(func() {
		card, regime := a.MacroCard(ctx)
		mu.Lock()
		g.regime = regime
		mu.Unlock()
		put(keyMacro, card)
	})
	run(func() { put(keyWhale, a.WhaleCard(ctx)) })
	run(func() { put(keyFunding, a.FundingCard(ctx)) })
	run(func() { put(keyMomentum, a.MomentumCard(ctx)) })
	run(func() { put(keyTrend, a.TrendCard(ctx, btcSpec)) })
	run(func() { put(keySR, a.SRCard(ctx, btcSpec)) })
	run(func() { put(keyVol, a.VolCard(ctx, btcSpec)) })
	run(func() { // FX block — informational, never competes for the top slot
		reads := a.fxReads(ctx)
		mu.Lock()
		g.fx = reads
		for _, r := range reads {
			if r.OK {
				g.fxAnyOK = true
				break
			}
		}
		mu.Unlock()
	})
	run(func() { // context lines — skipped silently when unavailable
		if n, err := a.api.Narratives(ctx); err == nil && len(n.Narratives) > 0 {
			top := pickTopNarrative(n.Narratives)
			// Radar silence threshold (see newsMinMentions): a theme with a
			// handful of mentions must not surface as a scored finding in the
			// digest either — no extra line, no AI-payload narrative.
			if top.MentionCount >= newsMinMentions {
				mu.Lock()
				g.topNarr = &top
				g.extras = append(g.extras, fmt.Sprintf("📖 <b>Narrative</b>: %s (%s, score %d)",
					esc(top.Narrative), esc(top.Stage), top.TrendScore))
				mu.Unlock()
			}
		}
	})
	run(func() {
		if m, err := a.api.MoodRead(ctx); err == nil && strings.TrimSpace(m.Read) != "" {
			mu.Lock()
			g.mood = strings.TrimSpace(m.Read)
			g.extras = append(g.extras, "🧠 <i>"+esc(truncate(firstSentence(m.Read), 160))+"</i>")
			mu.Unlock()
		}
	})
	wg.Wait()
	sort.Strings(g.extras) // deterministic order: narrative 📖 before mood 🧠? sorted by emoji bytes — stable across runs
	return g
}

// deviations extracts the priority inputs: offline agents drop out.
func (g gathered) deviations() map[string]int {
	out := map[string]int{}
	for _, k := range signalOrder {
		if c, ok := g.cards[k]; ok && !c.Offline {
			out[k] = c.Deviation
		}
	}
	return out
}

func (b *Bot) digestReply(ctx context.Context) string {
	g := b.ag.gather(ctx)
	return renderDigestHTML(g, b.ag.aiBrief(ctx, g))
}

// renderDigestHTML is the pure digest renderer: one gather sweep plus an
// optional AI brief → the exact Telegram HTML message. Shared verbatim by the
// Telegram /digest command and the HTTP /agents/digest endpoint.
func renderDigestHTML(g gathered, brief string) string {
	winner, top := topSelection(g)

	var sb strings.Builder
	// ALPHAVIZOR AI brief on top — styled like the site's diagnosis boxes.
	// Empty (no key / failure) → the digest opens with its own header as before.
	if brief != "" {
		sb.WriteString("<b>ALPHAVIZOR AI</b>\n<i>")
		sb.WriteString(esc(brief))
		sb.WriteString("</i>\n\n")
	}
	sb.WriteString("<b>AlphaVizor Digest</b> — top signal first\n\n")
	sb.WriteString(top.renderBody())
	sb.WriteString("\n<b>Everything else</b>\n")
	for _, k := range digestOrder {
		if k == winner {
			continue
		}
		c, ok := g.cards[k]
		if !ok {
			continue
		}
		sb.WriteString(c.OneLiner())
		sb.WriteString("\n")
	}
	// Compact FX block (informational — see priority.go: FX does not
	// compete for the top slot in v1).
	fxHeader := "\n<b>FX</b>"
	if !isForexOpen(time.Now()) {
		fxHeader += " <i>(market closed — Friday data)</i>"
	}
	sb.WriteString(fxHeader + "\n")
	if g.fxAnyOK {
		for _, r := range g.fx {
			sb.WriteString(fxLine(r))
			sb.WriteString("\n")
		}
	} else {
		sb.WriteString("⚪ FX: data unavailable right now\n")
	}
	for _, ex := range g.extras {
		sb.WriteString(ex)
		sb.WriteString("\n")
	}
	sb.WriteString("\n<i>Analytics, not financial advice · AlphaVizor · ")
	sb.WriteString(time.Now().UTC().Format("2006-01-02 15:04"))
	sb.WriteString(" UTC</i>")
	return sb.String()
}

func (b *Bot) topReply(ctx context.Context) string {
	g := b.ag.gather(ctx)
	winner, card := topSelection(g)
	brief, why := b.ag.aiTopTexts(ctx, winner, g)
	return renderTopHTML(card, brief, why)
}

// mergeTopWhy folds the AI why-line into the card's AI block (after any
// existing block, e.g. the macro mood read) so it renders above the footer.
// Value receiver-style: the caller's card is never mutated.
func mergeTopWhy(card Card, why string) Card {
	if why == "" {
		return card
	}
	if card.AIHTML != "" {
		card.AIHTML += "\n"
	}
	card.AIHTML += "<b>AI why:</b> <i>" + esc(why) + "</i>"
	return card
}

// renderTopHTML is the pure /top renderer: winner card plus optional AI
// strings → the exact Telegram HTML message. Shared verbatim by the Telegram
// /top command and the HTTP /agents/top endpoint.
func renderTopHTML(card Card, brief, why string) string {
	card = mergeTopWhy(card, why)
	var sb strings.Builder
	if brief != "" {
		sb.WriteString("<b>ALPHAVIZOR AI</b>\n<i>")
		sb.WriteString(esc(brief))
		sb.WriteString("</i>\n\n")
	}
	sb.WriteString("<b>Top signal right now</b>\n\n")
	sb.WriteString(card.RenderHTML())
	return sb.String()
}

// pickTopNarrative: highest trend score wins; ties break on mention count,
// then name, for determinism.
func pickTopNarrative(list []NarrativeSnapshot) NarrativeSnapshot {
	best := list[0]
	for _, n := range list[1:] {
		if n.TrendScore > best.TrendScore ||
			(n.TrendScore == best.TrendScore && n.MentionCount > best.MentionCount) ||
			(n.TrendScore == best.TrendScore && n.MentionCount == best.MentionCount && n.Narrative < best.Narrative) {
			best = n
		}
	}
	return best
}

func firstSentence(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexAny(s, ".!?"); i > 0 && i < len(s)-1 {
		return s[:i+1]
	}
	return s
}

const helpText = `<b>AlphaVizor Demo Bot</b> 🛰

Live market agents from the AlphaVizor platform:

/menu — button navigation, no typing needed
/macro — risk-on/off regime of big money
/whale — large on-chain BTC transfers, net flow
/funding — perp funding pressure &amp; liquidations
/fx — forex overview: EURUSD, GBPUSD, USDJPY, XAUUSD
/news — trending crypto narratives (48h) with an AI idea
/momentum — RSI/MACD reads for BTC, ETH &amp; gold
/trend — trend state machine (default BTC 4h)
/sr — key support/resistance levels
/vol — volatility expansion check
/risk — position-size calculator: /risk 10000 1 64000 62500
/digest — all agents in one message, prioritized
/top — the single strongest signal right now

/momentum, /trend, /sr and /vol take an optional asset:
<code>/trend eurusd</code> · <code>/sr gold</code> · <code>/vol usdjpy</code>
Assets: btc (default), eth, eurusd, gbpusd, usdjpy, xau/gold.

/momentum also scans a list at a timeframe (up to 6 assets, tf 1h/4h/1d):
<code>/momentum btc,eurusd 1d</code> · <code>/momentum 4h</code>

Every card has 🔄 Refresh and ℹ️ How it works buttons.

<i>Analytics, not financial advice. When a data source is down, the bot says so — it never fakes numbers.</i>`
