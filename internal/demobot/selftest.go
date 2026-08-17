package demobot

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// SelfTest builds every agent card once against live sources and prints the
// rendered HTML to stdout. Used by `demobot -selftest` to verify the data
// paths on a box without touching Telegram at all.
func SelfTest(ag *Agents) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	bot := &Bot{ag: ag, lastReq: map[int64]time.Time{}}

	section := func(name, body string) {
		fmt.Println(strings.Repeat("─", 60))
		fmt.Println("· " + name)
		fmt.Println(strings.Repeat("─", 60))
		fmt.Println(body)
		fmt.Println()
	}

	macro, regime := ag.MacroCard(ctx)
	section("/macro (regime="+regime+")", macro.RenderHTML())
	section("/whale", ag.WhaleCard(ctx).RenderHTML())
	section("/funding", ag.FundingCard(ctx).RenderHTML())
	section("/fx", ag.FXCard(ctx).RenderHTML())
	section("/momentum", ag.MomentumCard(ctx).RenderHTML())
	section("/trend", ag.TrendCard(ctx, btcSpec).RenderHTML())
	if spec, err := resolveAsset("eurusd"); err == nil {
		section("/trend eurusd", ag.TrendCard(ctx, spec).RenderHTML())
		section("/sr eurusd", ag.SRCard(ctx, spec).RenderHTML())
	}
	if spec, err := resolveAsset("gold"); err == nil {
		section("/vol gold", ag.VolCard(ctx, spec).RenderHTML())
	}
	section("/sr", ag.SRCard(ctx, btcSpec).RenderHTML())
	section("/vol", ag.VolCard(ctx, btcSpec).RenderHTML())
	section("/risk (example)", ag.RiskCard([]float64{10000, 1, 64000, 62500}, true, nil).RenderHTML())
	section("/digest", bot.digestReply(ctx))
	section("/top", bot.topReply(ctx))
}
