package demobot

// radar_text.go — everything the Risk Radar card SAYS.
//
// Risk Radar is a daily BTC volatility-SIZE forecast: a gradient-boosted
// model trained and validated offline (Python, once a day at 00:00 UTC)
// predicts the size of the next 24h move (1σ of the day's log-return), never
// its direction. This agent only reads the model's output file (radar.go)
// and words it — every number below comes from that file, nothing here is a
// code constant standing in for a live result.
//
// Honesty rules for every line on this card:
//   - "expected volatility" is the only forecast word this card uses: no "will",
//     no "likely", no "expect price", no "target" (TestRadarNoForecastWords)
//     — a volatility-SIZE forecast is not a price or direction call, and
//     1σ of the day's log return is not a price range, so no "±" either;
//   - it must never contradict the Volatility Agent's own BTC read: the
//     fusion line is built from the exact Card a.VolCard renders for BTC
//     (radar.go), never a second, independently-worded read;
//   - a forecast is shown only for the caller's OWN UTC day (radar.go);
//     every other case is an honest degraded card, never a stale number
//     relabeled as today's;
//   - every line — verdict, facts, each content block — fits
//     radarFactMaxRunes, the same 110-rune budget the Volatility card uses.

import (
	"fmt"
	"math"
	"time"
)

// radarFactMaxRunes mirrors volFactMaxRunes: the readability budget for one
// line of Risk Radar text.
const radarFactMaxRunes = 110

const (
	radarAgent     = "Risk Radar"
	radarShortName = "Risk Radar"
)

// radarLimitations is the one line the card states about what the forecast
// is NOT: a direction, a range, or advice.
const radarLimitations = "Forecast of volatility size from BTC hourly data (and model inputs); not direction, not advice"

// radarSizeOnlyFact is the standing reminder the card carries beside the
// validation numbers: the forecast is a size, not a price or a range.
const radarSizeOnlyFact = "Size of the move only: no direction and no guaranteed range"

// radarDataTime is the card's Last-Modified basis: the file's own
// generated_at, parsed as RFC3339. An unparseable stamp (a malformed file
// would already have failed JSON decoding earlier; this guards a valid JSON
// document with a bad string in that one field) falls back to now — honest,
// never a zero time that reads as 1970.
func radarDataTime(f radarFile, now time.Time) time.Time {
	if t, err := time.Parse(time.RFC3339, f.GeneratedAt); err == nil {
		return t.UTC()
	}
	return now.UTC()
}

// radarPctOf turns a fraction (0.69) into its whole percent (69), rounded
// half away from zero — the file's gain/percentile fields are fractions, its
// vol_pct/yesterday_pct fields are already percentages (radarVerdict,
// radarFact1 print those directly).
func radarPctOf(frac float64) int { return int(math.Round(frac * 100)) }

// radarPValue words a p-value the way a reader expects: three decimals, or
// "<0.001" once formatting to three decimals would print "0.000" — a result
// this small must never read as exactly zero.
func radarPValue(p float64) string {
	if p < 0.0005 {
		return "<0.001"
	}
	return fmt.Sprintf("%.3f", p)
}

// radarVerdict: "Next 24h · BTC: expected volatility 2.1% (1σ) — more active than
// usual". vol_pct rounds to one decimal; level is the file's own word for
// the read (never a code constant — the model's own calibration decides it).
func radarVerdict(f radarFile) string {
	return fmt.Sprintf("Next 24h · BTC: expected volatility %.1f%% (1σ) — %s", f.VolPct, f.Level)
}

// radarShort is the compact digest-style line (radar carries no digest slot,
// but every card keeps one for the landing showcase's one-liner).
func radarShort(f radarFile) string {
	return fmt.Sprintf("expected volatility %.1f%% — %s", f.VolPct, f.Level)
}

// radarFact1: the forecast itself, at the file's own two-decimal precision,
// beside yesterday's realized number for scale.
func radarFact1(f radarFile) string {
	return fmt.Sprintf("Model forecast for %s UTC: %.2f%% daily volatility (1σ of the day's move); yesterday %.2f%%",
		f.ForecastDay, f.VolPct, f.YesterdayPct)
}

// radarFact2: where today's forecast ranks against its own trailing 180-day
// history — a rank, not a claim about direction, so it always reads
// "higher than" whatever the level word says.
func radarFact2(f radarFile) string {
	return fmt.Sprintf("Higher than %d%% of the last 180 days · issued 00:00 UTC from closed hourly candles",
		radarPctOf(f.Percentile180d))
}

// radarVolatilityLine is fact 3: the one place this card leans on another
// agent, worded from that agent's OWN live card so the two can never
// disagree (see RadarCard). vol is nil-safe: a Volatility read that did not
// produce a real reading (offline, insufficient history) reads as
// unavailable, exactly like a second-hand read that could not be confirmed.
func radarVolatilityLine(vol Card) string {
	if vol.effectiveStatus() != statusOK {
		return "Volatility agent reading unavailable; this forecast covers the whole UTC day"
	}
	return fmt.Sprintf("Volatility agent: last %s candles vs prior 30 (now: %s); this forecast covers the whole UTC day",
		candleWord(btcSpec.Interval), volWord(vol.State))
}

// radarFact4: the backtest headline — unseen days, and the error reduction
// against the two baselines the validation script scores against.
func radarFact4(f radarFile) string {
	return fmt.Sprintf("Tested on %d unseen days: %d%% less error than HAR-RV, %d%% less than ATR(14)",
		f.Validation.Days, radarPctOf(f.Validation.GainVsHAR), radarPctOf(f.Validation.GainVsATR14))
}

// radarFact5: the harder baseline (HAR-RV plus a day-of-week term) and
// whether the pre-registered significance test passed. gain is one decimal
// (the file's own precision for this field); the p-value is never rounded to
// "confirmed" — the wording branches on the file's own pass/fail flag, never
// on a threshold this code re-derives from the p-value itself.
func radarFact5(f radarFile) string {
	status := "not yet statistically confirmed"
	if f.Validation.PreRegisteredPass {
		status = "confirmed"
	}
	return fmt.Sprintf("Against HAR-RV + weekday: %.1f%% less error, %s (p=%s)",
		f.Validation.GainVsHARDow*100, status, radarPValue(f.Validation.PVsHARDow))
}

// radarFact7 is the live forward-test line, shown only once the forward
// window has at least one day in it — a zero-day window has nothing to
// report yet and stays silent rather than print "0 days, 0.0%".
func radarFact7(f radarFile) (string, bool) {
	if f.Forward.Days <= 0 || f.Forward.GainVsHARDow == nil {
		return "", false
	}
	return fmt.Sprintf("Live forward test: %d days, %.1f%% vs HAR-RV + weekday",
		f.Forward.Days, *f.Forward.GainVsHARDow*100), true
}

// radarFacts assembles the full, ordered fact list.
func radarFacts(f radarFile, volLine string) []string {
	facts := []string{
		radarFact1(f),
		radarFact2(f),
		volLine,
		radarFact4(f),
		radarFact5(f),
		radarSizeOnlyFact,
	}
	if extra, ok := radarFact7(f); ok {
		facts = append(facts, extra)
	}
	return facts
}

// radarBlocks is the content-ready form (ContentBlocks): what_happened
// mirrors the verdict, why_level explains the 1σ forecast has no price
// level, scenarios says what happens next (there is no state machine to
// transition — a fresh forecast simply replaces this one tomorrow),
// invalidates stays nil (no directional idea to invalidate) and limitations
// is the standing size-only, no-advice line.
func radarBlocks(f radarFile) *ContentBlocks {
	return &ContentBlocks{
		WhatHappened: fmt.Sprintf("BTC expected daily volatility %.1f%% (1σ): %s (higher than %d%% of the last 180 days).",
			f.VolPct, f.Level, radarPctOf(f.Percentile180d)),
		WhyLevel: "No price level: 1σ of BTC's daily move from an hourly-data ML model, not a support/resistance level",
		Scenarios: []string{
			"Tomorrow's forecast replaces this one at 00:00 UTC, from the next closed hourly candles",
			"A move far from this size does not revise today's forecast; it is only scored in the model's own window",
		},
		Limitations: radarLimitations,
	}
}

// radarCardFrom builds the happy-path card. noValidator: the Volatility
// fusion line (fact 3) follows a's kline cache, not the file's stamp, so the
// same generated_at can legitimately pair with a different fact 3 later in
// the same UTC day — a Last-Modified keyed on the file alone would then let
// a conditional GET 304 a body that changed (the same reasoning
// decorateFXAt documents for market-state wording).
func radarCardFrom(f radarFile, volLine string, now time.Time) Card {
	return Card{
		Emoji:       emojiNeutral,
		Agent:       radarAgent,
		ShortName:   radarShortName,
		Asset:       btcSpec.Display,
		Command:     keyRadar,
		HowItWorks:  howTexts[keyRadar],
		DataTime:    radarDataTime(f, now),
		Verdict:     radarVerdict(f),
		Short:       radarShort(f),
		Facts:       radarFacts(f, volLine),
		Blocks:      radarBlocks(f),
		noValidator: true,
	}
}

// radarNoForecastCard is the honest state for a file that exists and parses
// but is not for the caller's own UTC day: no verdict number, ok:false with
// reason "no_data" (statusNoData) — the source is alive, there is simply
// nothing for today yet, the same distinction the macro "unknown" card
// draws. noValidator: whether this card or the happy path renders depends on
// the request clock, so no stamp may claim a fixed body.
func radarNoForecastCard(lastDay string, now time.Time) Card {
	return Card{
		Emoji:       emojiNeutral,
		Agent:       radarAgent,
		ShortName:   radarShortName,
		Asset:       btcSpec.Display,
		Command:     keyRadar,
		HowItWorks:  howTexts[keyRadar],
		DataTime:    now.UTC(),
		Status:      statusNoData,
		noValidator: true,
		Verdict:     fmt.Sprintf("No model forecast for today yet (last: %s)", lastDay),
		Short:       "no forecast yet",
		Facts:       []string{fmt.Sprintf("Waiting for today's 00:00 UTC model run; last forecast: %s", lastDay)},
	}
}

// radarOfflineCard is the honest state for no file / unparseable JSON — the
// same offline wording and 503 every other agent gives a dead source.
func radarOfflineCard() Card {
	return offlineCard(radarAgent, radarShortName, btcSpec.Display, keyRadar, howTexts[keyRadar])
}
