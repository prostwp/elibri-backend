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
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

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

	log.Printf("[demobot] starting · backend=%s", *backend)
	bot := demobot.NewBot(token, agents)
	if err := bot.Run(ctx); err != nil {
		log.Fatalf("[demobot] fatal: %v", err)
	}
	log.Printf("[demobot] stopped cleanly")
}
