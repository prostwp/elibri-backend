// Command demobot is the AlphaVizor Telegram demo bot — a thin presentation
// layer over the backend REST API plus a few Binance-public-API computations
// (trend / support-resistance / volatility) that have no REST endpoint yet.
//
// Run:
//
//	TELEGRAM_BOT_TOKEN=123456:ABC... ./bin/demobot
//
// Flags / env:
//
//	-backend URL            backend base URL (default $DEMOBOT_BACKEND_URL
//	                        or http://localhost:8080)
//	-selftest               build every card once against live sources,
//	                        print them to stdout and exit (no Telegram)
//	ANTHROPIC_API_KEY       enables the ALPHAVIZOR AI briefs on /digest and
//	                        /top (same key the backend narrative worker
//	                        uses). Empty → AI blocks silently omitted.
//	DEMOBOT_HTTP_ADDR       listen address of the read-only HTTP JSON API
//	                        that runs alongside long polling (default
//	                        127.0.0.1:8090) — see docs/demobot-http.md.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prostwp/elibri-backend/internal/demobot"
)

func main() {
	log.SetFlags(log.LstdFlags | log.LUTC)

	defaultBackend := os.Getenv("DEMOBOT_BACKEND_URL")
	if defaultBackend == "" {
		defaultBackend = "http://localhost:8080"
	}
	backend := flag.String("backend", defaultBackend, "AlphaVizor backend base URL")
	selftest := flag.Bool("selftest", false, "render every agent card once and exit (no Telegram)")
	flag.Parse()

	agents := demobot.NewAgents(demobot.NewBackendClient(*backend))
	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
		agents.EnableAI(key)
		log.Printf("[demobot] ALPHAVIZOR AI briefs: enabled")
	} else {
		log.Printf("[demobot] ALPHAVIZOR AI briefs: disabled (ANTHROPIC_API_KEY empty)")
	}

	if *selftest {
		demobot.SelfTest(agents)
		return
	}

	token := os.Getenv("TELEGRAM_BOT_TOKEN")
	if token == "" {
		fmt.Fprintln(os.Stderr, "TELEGRAM_BOT_TOKEN is not set.")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "Create a bot with @BotFather in Telegram, copy its token, then run:")
		fmt.Fprintln(os.Stderr, "  TELEGRAM_BOT_TOKEN=123456:ABC-your-token ./bin/demobot")
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Read-only HTTP JSON API — runs alongside long polling on the SAME
	// Agents instance (shared kline cache + AI memo). A bind failure is
	// fatal: a half-up process that looks alive but serves nothing is worse
	// than a clean crash launchd can restart.
	httpAddr := os.Getenv("DEMOBOT_HTTP_ADDR")
	if httpAddr == "" {
		httpAddr = demobot.DefaultHTTPAddr
	}
	httpSrv := demobot.NewHTTPServer(httpAddr, agents)
	if err := httpSrv.Start(); err != nil {
		log.Fatalf("[demobot] HTTP API cannot bind %s: %v — is another demobot instance running? Set DEMOBOT_HTTP_ADDR to move it.", httpAddr, err)
	}

	log.Printf("[demobot] starting · backend=%s", *backend)
	bot := demobot.NewBot(token, agents)
	runErr := bot.Run(ctx)

	// Same SIGTERM path as long polling: Run has returned (signal or fatal
	// poll error) — drain in-flight HTTP requests before exiting.
	shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), 5*time.Second)
	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		log.Printf("[demobot] HTTP API shutdown: %v", err)
	}
	cancelShutdown()

	if runErr != nil {
		log.Fatalf("[demobot] fatal: %v", runErr)
	}
	log.Printf("[demobot] stopped cleanly")
}
