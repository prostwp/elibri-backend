package demobot

// ai_confirmation_test.go — the authoritative-verdict guard (ai.go).
//
// Defect being pinned (team landing review, 2026-08): the trend state machine
// said "Grey zone — trend forming, not confirmed" (structure gate failed)
// while the AI why-line said "the only signal showing confirmed directional
// structure … ADX 58.8, well above the 25 threshold". Real numbers, but the
// conclusion the state machine had explicitly refused — the product
// contradicting itself on a marketing page.

import (
	"context"
	"strings"
	"testing"
)

// greyTrendGathered reproduces the live defect state: ADX is high, but the
// swing structure demoted the read to grey, so confirmation is WITHHELD.
func greyTrendGathered() gathered {
	g := fakeGathered()
	g.cards[keyTrend] = Card{
		ShortName: "Trend", Agent: "Trend Agent", Command: keyTrend, Asset: "BTC",
		Emoji:   emojiNeutral,
		Verdict: "Grey zone — trend forming, not confirmed",
		State:   trendGrey,
		Facts: []string{
			"ADX(14): 58.8 · RSI(14): 62.0",
			"Structure mixed — trend not confirmed",
		},
	}
	return g
}

// confirmedTrendGathered is the positive control: a genuinely confirmed
// uptrend, where "confirmed" wording is the truth and must survive.
func confirmedTrendGathered() gathered {
	g := fakeGathered()
	g.cards[keyTrend] = Card{
		ShortName: "Trend", Agent: "Trend Agent", Command: keyTrend, Asset: "BTC",
		Emoji:   emojiBull,
		Verdict: "Confirmed UPTREND",
		State:   trendUp,
		Facts:   []string{"ADX(14): 31.2 · RSI(14): 62.0"},
	}
	return g
}

// The exact sentence the live bot served, plus a second one that is fine.
const contradictingReply = "The trend agent holds the top slot because it is the only signal showing confirmed directional structure. ADX(14) reads 58.8, well above the 25 threshold required to confirm trend. Funding is balanced at +0.0100%/8h."

// End to end through the real AI path: a model reply asserting confirmation
// is not served when the winning card's state withheld it.
func TestAIWhyDropsConfirmationTheVerdictWithheld(t *testing.T) {
	_, client := newAIStub(t, func(n int, body map[string]any) (int, string) {
		return 200, aiEnvelope(contradictingReply)
	})
	ag := deadAgents(t)
	ag.ai = client

	got := ag.aiTopWhy(context.Background(), keyTrend, greyTrendGathered())
	for _, banned := range []string{"confirmed directional structure", "required to confirm trend"} {
		if strings.Contains(got, banned) {
			t.Errorf("served text still claims confirmation the state machine withheld: %q", got)
		}
	}
	if strings.Contains(strings.ToLower(got), "confirmed") {
		t.Errorf("no confirmation claim may survive on a grey card: %q", got)
	}
	// The sentence that was never in dispute is untouched — the filter is a
	// scalpel, not a mute button.
	if !strings.Contains(got, "Funding is balanced at +0.0100%/8h.") {
		t.Errorf("unrelated sentence was dropped: %q", got)
	}
}

// Positive control: a real confirmed uptrend keeps its wording.
func TestAIWhyKeepsConfirmationWhenVerdictConfirms(t *testing.T) {
	_, client := newAIStub(t, func(n int, body map[string]any) (int, string) {
		return 200, aiEnvelope(contradictingReply)
	})
	ag := deadAgents(t)
	ag.ai = client

	got := ag.aiTopWhy(context.Background(), keyTrend, confirmedTrendGathered())
	if !strings.Contains(got, "confirmed directional structure") {
		t.Errorf("a confirmed uptrend must keep its confirmation wording: %q", got)
	}
}

// The same guard covers the digest brief — one shared path, so /digest, /top
// and /showcase/example are all fixed by it.
func TestAIBriefDropsContradictingConfirmation(t *testing.T) {
	_, client := newAIStub(t, func(n int, body map[string]any) (int, string) {
		return 200, aiEnvelope(contradictingReply)
	})
	ag := deadAgents(t)
	ag.ai = client

	if got := ag.aiBrief(context.Background(), greyTrendGathered()); strings.Contains(strings.ToLower(got), "confirmed") {
		t.Errorf("brief kept a contradicting confirmation claim: %q", got)
	}
}

// The payload tells the model the verdict is authoritative and that this
// particular state withheld confirmation — layer one of the fix.
func TestAIPayloadCarriesAuthoritativeState(t *testing.T) {
	grey := aiPayload(greyTrendGathered())
	for _, want := range []string{
		`"authoritative_verdict":"Grey zone — trend forming, not confirmed"`,
		`"state":"grey"`,
		`"confirmation_withheld":true`,
	} {
		if !strings.Contains(grey, want) {
			t.Errorf("payload missing %s:\n%s", want, grey)
		}
	}
	if strings.Contains(grey, `"verdict":`) {
		t.Errorf("the bare 'verdict' key invites re-litigation — it must read authoritative_verdict:\n%s", grey)
	}
	// A confirmed trend does NOT carry the withheld flag.
	confirmed := aiPayload(confirmedTrendGathered())
	if !strings.Contains(confirmed, `"state":"up"`) {
		t.Errorf("confirmed trend payload missing its state:\n%s", confirmed)
	}
	idx := strings.Index(confirmed, `"authoritative_verdict":"Confirmed UPTREND"`)
	if idx < 0 {
		t.Fatalf("confirmed verdict missing:\n%s", confirmed)
	}
	if seg := confirmed[idx : idx+200]; strings.Contains(seg, `"confirmation_withheld":true`) {
		t.Errorf("a confirmed trend must not be flagged as withholding: %s", seg)
	}
}

// Unit level: which states assert confirmation, and which withhold it.
func TestStateConfirms(t *testing.T) {
	cases := []struct {
		card Card
		want bool
	}{
		{Card{State: trendUp}, true},
		{Card{State: trendDown}, true},
		{Card{State: "risk_on"}, true},
		{Card{State: "risk_off"}, true},
		{Card{State: volExpanding}, true},
		{Card{State: trendGrey}, false},
		{Card{State: trendFlat}, false},
		{Card{State: trendConflict}, false},
		{Card{State: "unknown"}, false},
		{Card{State: "mixed"}, false},
		{Card{State: volNormal}, false},
		{Card{State: volCompressed}, false},
		{Card{}, false}, // no state machine at all
		// A degraded card confirms nothing, whatever its stale state says.
		{Card{State: trendUp, Offline: true}, false},
		{Card{State: trendUp, Status: statusInsufficientHistory}, false},
	}
	for _, tc := range cases {
		if got := stateConfirms(tc.card); got != tc.want {
			t.Errorf("stateConfirms(state=%q offline=%v status=%d): got %v, want %v",
				tc.card.State, tc.card.Offline, tc.card.Status, got, tc.want)
		}
	}
}

// The filter's edges: agreement with the withholding survives, claims about
// agents that did confirm survive, and an all-contradiction text collapses to
// "" so the caller omits the block rather than shipping it.
func TestSanitizeConfirmationClaims(t *testing.T) {
	greyTopics := withheldTopics(greyTrendGathered())
	cases := []struct {
		name, in, want string
		topics         []string
	}{
		{
			name:   "drops the confirmation claim, keeps the rest",
			in:     "Trend structure is confirmed. Funding sits at +0.01%.",
			topics: greyTopics,
			want:   "Funding sits at +0.01%.",
		},
		{
			name:   "keeps a sentence that AGREES the trend is not confirmed",
			in:     "The trend is not confirmed yet. Watch the EMA cluster.",
			topics: greyTopics,
			want:   "The trend is not confirmed yet. Watch the EMA cluster.",
		},
		{
			name:   "keeps 'not yet fully confirmed' (words between negation and claim)",
			in:     "Structure is not yet fully confirmed.",
			topics: greyTopics,
			want:   "Structure is not yet fully confirmed.",
		},
		{
			name:   "keeps 'unconfirmed'",
			in:     "The uptrend remains unconfirmed.",
			topics: greyTopics,
			want:   "The uptrend remains unconfirmed.",
		},
		{
			name:   "leaves confirmation claims about ungated agents alone",
			in:     "Funding confirms crowded longs.",
			topics: greyTopics,
			want:   "Funding confirms crowded longs.",
		},
		{
			name:   "decimals never split a sentence",
			in:     "ADX 58.8 confirms the trend. Watch 118000.",
			topics: greyTopics,
			want:   "Watch 118000.",
		},
		{
			name:   "nothing survives → empty, so the caller omits the block",
			in:     "The trend is confirmed. Confirmed structure all around.",
			topics: greyTopics,
			want:   "",
		},
		{
			name:   "no withheld topics → text passes through untouched",
			in:     "Trend structure is confirmed.",
			topics: withheldTopics(confirmedTrendGathered()),
			want:   "Trend structure is confirmed.",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sanitizeConfirmationClaims(tc.in, tc.topics); got != tc.want {
				t.Errorf("got  %q\nwant %q", got, tc.want)
			}
		})
	}
}

// withheldTopics must list a gated agent's words only while it withholds —
// and a degraded/missing card counts as withholding (it confirmed nothing).
func TestWithheldTopics(t *testing.T) {
	has := func(topics []string, w string) bool {
		for _, t := range topics {
			if t == w {
				return true
			}
		}
		return false
	}
	if got := withheldTopics(greyTrendGathered()); !has(got, "trend") {
		t.Errorf("grey trend must contribute its topics, got %v", got)
	}
	if got := withheldTopics(confirmedTrendGathered()); has(got, "trend") {
		t.Errorf("a confirmed trend must NOT gag trend wording, got %v", got)
	}
	// fakeGathered's macro is risk_on but carries no State (older fixture
	// shape) — an unset state confirms nothing, so macro topics are gated.
	if got := withheldTopics(fakeGathered()); !has(got, "regime") {
		t.Errorf("a card without a state machine confirms nothing, got %v", got)
	}
}

func TestSplitSentences(t *testing.T) {
	got := splitSentences("ADX 58.8 is high. Watch 118000! Then what? Done")
	want := []string{"ADX 58.8 is high.", "Watch 118000!", "Then what?", "Done"}
	if len(got) != len(want) {
		t.Fatalf("got %d sentences %q, want %d", len(got), got, len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("sentence %d: got %q, want %q", i, got[i], want[i])
		}
	}
	if len(splitSentences("")) != 0 {
		t.Error("empty input must yield no sentences")
	}
}
