package narrative

import (
	"sort"
	"strings"
	"testing"
)

// TestMatchNarratives_Positive walks one positive example per narrative slug.
// Each fixture was hand-picked to use a realistic news headline phrasing so a
// future tweak that breaks substring matching (e.g. swapping Contains for an
// exact-token match) will fail at least one of these.
func TestMatchNarratives_Positive(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		title       string
		summary     string
		wantInclude string // slug that MUST appear in result
	}{
		{
			name:        "restaking via eigenlayer mention",
			title:       "EigenLayer hits $15B TVL as restaking demand surges",
			summary:     "Liquid restaking tokens dominate inflows.",
			wantInclude: "restaking",
		},
		{
			name:        "ai-tokens via bittensor",
			title:       "Bittensor TAO rallies on subnet auction news",
			summary:     "AI token momentum continues.",
			wantInclude: "ai-tokens",
		},
		{
			name:        "rwa via tokenized treasuries phrase",
			title:       "Ondo doubles down on tokenized treasuries product",
			summary:     "Real world assets sector rotates in.",
			wantInclude: "rwa",
		},
		{
			name:        "memecoins via dogwifhat",
			title:       "Dogwifhat (WIF) hits new all-time high",
			summary:     "Solana memecoin rally extends.",
			wantInclude: "memecoins",
		},
		{
			name:        "btc-l2 via stacks",
			title:       "Stacks STX rallies as Bitcoin L2 ecosystem grows",
			summary:     "BTC L2 narrative gathers steam.",
			wantInclude: "btc-l2",
		},
		{
			name:        "solana-defi via jupiter",
			title:       "Jupiter aggregates record volume on Solana",
			summary:     "Solana DeFi flywheel keeps spinning.",
			wantInclude: "solana-defi",
		},
		{
			name:        "modular via celestia",
			title:       "Celestia rolls out new modular blockchain features",
			summary:     "DA layer adoption metrics improving.",
			wantInclude: "modular",
		},
		{
			name:        "zk via zksync mention",
			title:       "zkSync Era surpasses Starknet in daily users",
			summary:     "ZK rollup race tightens.",
			wantInclude: "zk",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := MatchNarratives(tc.title, tc.summary)
			if !contains(got, tc.wantInclude) {
				t.Errorf("MatchNarratives(%q, %q) = %v, want to include %q",
					tc.title, tc.summary, got, tc.wantInclude)
			}
		})
	}
}

// TestMatchNarratives_MultiMatch verifies that an item touching multiple
// narratives returns all of them. Real news articles routinely span two or
// three of our slugs (e.g. "EigenLayer launches new AI-tokens subnet"), so
// the worker MUST insert one mention row per slug — collapsing them into the
// "first match wins" would corrupt every count downstream.
func TestMatchNarratives_MultiMatch(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		title   string
		summary string
		want    []string // every slug that must appear
	}{
		{
			name:    "restaking + ai-tokens",
			title:   "EigenLayer announces partnership with Bittensor",
			summary: "AI token holders to access restaking yields.",
			want:    []string{"restaking", "ai-tokens"},
		},
		{
			name:    "modular + zk",
			title:   "Celestia integrates with zkSync to push modular ZK rollup adoption",
			summary: "DA layer + ZK rollup combined.",
			want:    []string{"modular", "zk"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := MatchNarratives(tc.title, tc.summary)
			for _, w := range tc.want {
				if !contains(got, w) {
					t.Errorf("MatchNarratives(%q, %q) = %v, missing slug %q",
						tc.title, tc.summary, got, w)
				}
			}
		})
	}
}

// TestMatchNarratives_NoMatch covers the off-topic path. The classifier MUST
// return an empty slice (not nil — but the test accepts nil too to keep the
// API contract relaxed) for plain-vanilla content unrelated to our taxonomy.
func TestMatchNarratives_NoMatch(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		title   string
		summary string
	}{
		{
			name:    "generic Bitcoin price story",
			title:   "Bitcoin slips below 50,000 amid risk-off sentiment",
			summary: "Equities also lower as Treasury yields rise.",
		},
		{
			name:    "off-topic non-crypto",
			title:   "FOMC keeps rates unchanged at 5.25%",
			summary: "Powell cites sticky inflation in press conference.",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := MatchNarratives(tc.title, tc.summary)
			if len(got) != 0 {
				t.Errorf("MatchNarratives(%q, %q) = %v, want empty", tc.title, tc.summary, got)
			}
		})
	}
}

// TestMatchNarratives_EmptyInput pins the contract that empty/whitespace
// input returns an empty slice rather than crashing.
func TestMatchNarratives_EmptyInput(t *testing.T) {
	t.Parallel()

	if got := MatchNarratives("", ""); len(got) != 0 {
		t.Errorf("MatchNarratives empty = %v, want empty", got)
	}
	if got := MatchNarratives("   ", "  \t\n"); len(got) != 0 {
		t.Errorf("MatchNarratives whitespace = %v, want empty", got)
	}
}

// TestMatchNarratives_Dedup verifies that the same narrative is not added
// twice when MULTIPLE keywords for that narrative appear in the same text.
// This is the contract InsertMention relies on — duplicate slugs in one call
// would create duplicate (narrative, url) inserts → ON CONFLICT, but waste
// a round trip to PG.
func TestMatchNarratives_Dedup(t *testing.T) {
	t.Parallel()

	// Both "eigenlayer" and "restaking" keywords map to slug "restaking".
	got := MatchNarratives("EigenLayer restaking pool", "")
	count := 0
	for _, s := range got {
		if s == "restaking" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("MatchNarratives EigenLayer+restaking returned slug %d times, want 1", count)
	}
}

// TestExtractTickers covers the four extraction modes called out in the
// classifier docstring: $-prefixed, bare whitelisted, lowercase ignored,
// and embedded-in-word (which must NOT match thanks to the word boundary).
func TestExtractTickers(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		text string
		want []string // tickers expected, order-insensitive
	}{
		{
			name: "$BTC alone is extracted",
			text: "$BTC just printed a new ATH today.",
			want: []string{"BTC"},
		},
		{
			name: "BTC and ETH bare are extracted",
			text: "Reuters: BTC and ETH both rallied on the news.",
			want: []string{"BTC", "ETH"},
		},
		{
			name: "lowercase symbols ignored, but lowercase coin name picked up via Path 3",
			// "btc" / "eth" as bare symbols stay ignored (Path 2 requires
			// uppercase). "bitcoin" as a full name DOES match via Path 3 —
			// that's the fix for the day-one smoke gap (related_assets
			// empty when articles use names instead of symbols).
			text: "btc and eth held the line; bitcoin's price ticked higher.",
			want: []string{"BTC"},
		},
		{
			name: "embedded-in-word does not match",
			// "BTCUSDT" should NOT yield "BTC" — the regex requires \b after,
			// and "U" is a word char so the boundary fails. This protects us
			// from headlines like "ETHUSD up 3%" producing a phantom ETH.
			text: "Pair BTCUSDT closed at 50000.",
			want: []string{},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ExtractTickers(tc.text)
			if !sameSet(got, tc.want) {
				t.Errorf("ExtractTickers(%q) = %v, want %v", tc.text, got, tc.want)
			}
		})
	}
}

// TestExtractTickers_DedupAndOrder pins (a) deduplication and (b) the
// "first occurrence wins" ordering. The worker's TopTickers query relies on
// a clean tickers[] array per row — this test catches a future refactor
// that drops dedup.
func TestExtractTickers_DedupAndOrder(t *testing.T) {
	t.Parallel()

	got := ExtractTickers("$BTC then ETH then BTC again then $ETH")
	if len(got) != 2 {
		t.Fatalf("ExtractTickers expected 2 unique tickers, got %v", got)
	}
	// Order: $BTC seen first, then ETH (bare) seen after.
	if got[0] != "BTC" || got[1] != "ETH" {
		t.Errorf("ExtractTickers ordering = %v, want [BTC, ETH]", got)
	}
}

// TestExtractTickers_NonWhitelisted verifies that uppercase words NOT in the
// whitelist don't leak through. "FBI" is a classic false-positive trigger —
// matches the regex but isn't an asset.
func TestExtractTickers_NonWhitelisted(t *testing.T) {
	t.Parallel()

	got := ExtractTickers("FBI raids exchange; CEO arrested.")
	if len(got) != 0 {
		t.Errorf("ExtractTickers(FBI/CEO) = %v, want empty (whitelist gate)", got)
	}
}

// TestExtractTickers_NamePath covers Path 3 — full coin-name matches
// (case-insensitive). Real-world hit on day-one smoke: a coindesk article
// "Jito Labs launches self-custody trading tool as activity heats up on
// Solana" matched the solana-defi narrative on keyword "jito" but
// related_assets came back empty because neither $JTO nor $SOL appeared.
// This pins the new behavior.
func TestExtractTickers_NamePath(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		text string
		want []string // tickers expected (order-insensitive)
	}{
		{
			name: "real coindesk title - Jito + Solana",
			text: "Jito Labs launches self-custody trading tool as activity heats up on Solana",
			want: []string{"JTO", "SOL"},
		},
		{
			name: "Bitcoin (mainstream name)",
			text: "Bitcoin breaks $80k as ETF inflows surge",
			want: []string{"BTC"},
		},
		{
			name: "Solana lowercased",
			text: "solana defi tvl hits new high",
			want: []string{"SOL"},
		},
		{
			name: "EigenLayer (compound name)",
			text: "EigenLayer rolls out restaking v2",
			want: []string{"EIGEN"},
		},
		{
			name: "Jupiter + Raydium",
			text: "Jupiter aggregator integrates Raydium pools",
			want: []string{"JUP", "RAY"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ExtractTickers(tc.text)
			for _, w := range tc.want {
				if !contains(got, w) {
					t.Errorf("ExtractTickers(%q) = %v, want includes %q", tc.text, got, w)
				}
			}
		})
	}
}

// TestExtractTickers_NameWordBoundary protects Path 3 from substring
// false-positives: "ondo" inside "fundo" must NOT yield ONDO; "sol" is
// not a registered coin name (only "solana"), but if it ever is added
// it must not match inside "asylum" or "console".
func TestExtractTickers_NameWordBoundary(t *testing.T) {
	t.Parallel()

	// "ondo" is a registered coin name → ONDO ticker. Inside "fundo" the
	// substring is bounded by letters on both sides → must NOT match.
	got := ExtractTickers("Welcome to the fundo party — purely fictional.")
	for _, t1 := range got {
		if t1 == "ONDO" {
			t.Errorf("ExtractTickers picked ONDO from 'fundo' — word-boundary check broken: %v", got)
		}
	}

	// Punctuation neighbours are fine: "Bitcoin." and "(Bitcoin)" must match.
	got2 := ExtractTickers("Markets reacted to the news. Bitcoin, ethereum, and others rallied.")
	if !contains(got2, "BTC") || !contains(got2, "ETH") {
		t.Errorf("ExtractTickers should accept name + punctuation neighbours, got %v", got2)
	}
}

// TestAllNarrativeSlugs sanity-checks the discovery helper used by the
// worker. The exact count is locked here so adding a narrative prompts a
// test update — that's intentional, since adding a narrative needs a
// review of the keyword choices.
//
// Phase 2A doubled the taxonomy from 8 → 16 narratives by adding btc-etf,
// bitcoin-halving, stablecoins-regulation, gaming-metaverse, depin, oracles,
// perp-dex, cosmos-appchains. Live smoke (Phase 1) found the worker logged
// 0 matches per 24h across all 8 original narratives — the wider taxonomy
// + per-narrative dictionary expansion target a higher hit rate.
func TestAllNarrativeSlugs(t *testing.T) {
	t.Parallel()

	got := AllNarrativeSlugs()
	if len(got) != 16 {
		t.Errorf("AllNarrativeSlugs returned %d slugs, want 16: %v", len(got), got)
	}
	required := []string{
		// Original 8.
		"restaking", "ai-tokens", "rwa", "memecoins",
		"btc-l2", "solana-defi", "modular", "zk",
		// Phase 2A additions.
		"btc-etf", "bitcoin-halving", "stablecoins-regulation",
		"gaming-metaverse", "depin", "oracles", "perp-dex", "cosmos-appchains",
	}
	sort.Strings(got)
	sort.Strings(required)
	if strings.Join(got, ",") != strings.Join(required, ",") {
		t.Errorf("AllNarrativeSlugs = %v, want %v", got, required)
	}
}

// TestMatchNarratives_NewNarratives is the Phase 2A coverage gate. It walks
// one realistic-headline fixture per NEW narrative slug and checks the slug
// shows up in the result. We deliberately use phrasings that look like real
// CoinDesk / Cointelegraph headlines so future tweaks (e.g. swapping Contains
// for token match) will fail at least one.
func TestMatchNarratives_NewNarratives(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		title       string
		summary     string
		wantInclude string
	}{
		{
			name:        "btc-etf via BlackRock IBIT inflows",
			title:       "BlackRock IBIT sees record inflows as spot Bitcoin ETF demand returns",
			summary:     "Grayscale GBTC outflows finally easing.",
			wantInclude: "btc-etf",
		},
		{
			name:        "btc-etf via generic phrase",
			title:       "Spot Bitcoin ETF flows turn positive for the first time in two weeks",
			summary:     "Fidelity Bitcoin product leads the rebound.",
			wantInclude: "btc-etf",
		},
		{
			name:        "bitcoin-halving via halving cycle",
			title:       "Miner capitulation deepens as halving cycle squeezes margins",
			summary:     "Hash rate drops 8% week-on-week.",
			wantInclude: "bitcoin-halving",
		},
		{
			name:        "stablecoins-regulation via MiCA",
			title:       "Tether faces fresh scrutiny as MiCA rules go live in EU",
			summary:     "Circle USDC issuer sees opportunity in payment stablecoin market.",
			wantInclude: "stablecoins-regulation",
		},
		{
			name:        "stablecoins-regulation via PYUSD",
			title:       "PayPal USD (PYUSD) supply hits new high",
			summary:     "Stablecoin law debate continues in Congress.",
			wantInclude: "stablecoins-regulation",
		},
		{
			name:        "gaming-metaverse via Immutable / IMX",
			title:       "Immutable IMX integrates with Pixels for cross-game items",
			summary:     "Web3 gaming sector rotates back into focus.",
			wantInclude: "gaming-metaverse",
		},
		{
			name:        "gaming-metaverse via play to earn",
			title:       "Axie revival: new play to earn season launches on Ronin",
			summary:     "Decentraland and Sandbox round out the GameFi week.",
			wantInclude: "gaming-metaverse",
		},
		{
			name:        "depin via Helium HNT migration",
			title:       "Helium HNT migration to Solana complete, mobile subscribers grow",
			summary:     "DePIN sector gains attention.",
			wantInclude: "depin",
		},
		{
			name:        "depin via Filecoin / Arweave",
			title:       "Filecoin and Arweave see surge in storage demand from AI workloads",
			summary:     "Decentralized physical infrastructure revenue trends up.",
			wantInclude: "depin",
		},
		{
			name:        "oracles via Chainlink upgrade",
			title:       "Chainlink rolls out CCIP upgrade as oracle network usage hits ATH",
			summary:     "Pyth Network chases enterprise data feed clients.",
			wantInclude: "oracles",
		},
		{
			name:        "oracles via Pyth Network",
			title:       "Pyth Network expands real-time price coverage to commodities",
			summary:     "Data feed competition heats up.",
			wantInclude: "oracles",
		},
		{
			name:        "perp-dex via Hyperliquid volume",
			title:       "Hyperliquid hits $1B daily volume on perpetuals as users rotate from CEXes",
			summary:     "On-chain perpetuals market share climbs.",
			wantInclude: "perp-dex",
		},
		{
			name:        "perp-dex via dYdX / GMX",
			title:       "dYdX v4 surpasses GMX in monthly fees as perp dex race tightens",
			summary:     "Vertex and Aevo round out the top five.",
			wantInclude: "perp-dex",
		},
		{
			name:        "cosmos-appchains via Osmosis / Injective",
			title:       "Osmosis and Injective lead Cosmos appchain revival as IBC volume rebounds",
			summary:     "Kava integrations bring new liquidity.",
			wantInclude: "cosmos-appchains",
		},
		{
			name:        "cosmos-appchains via ATOM rally",
			title:       "ATOM rallies 12% as Cosmos Hub upgrade activates",
			summary:     "Appchain thesis returns to the spotlight.",
			wantInclude: "cosmos-appchains",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := MatchNarratives(tc.title, tc.summary)
			if !contains(got, tc.wantInclude) {
				t.Errorf("MatchNarratives(%q, %q) = %v, want to include %q",
					tc.title, tc.summary, got, tc.wantInclude)
			}
		})
	}
}

// TestMatchNarratives_NoConflicts pins the conflict-resolution decisions
// made when expanding the dictionary. Several keywords could naturally fit
// in two slugs; the comments in classifier.go document the canonical home.
//
// Examples:
//   - "celestia" / "tia" / "dymension" / "dym" → modular ONLY
//     (NOT cosmos-appchains, even though they're built on the Cosmos SDK).
//   - "eigenlayer" / "eigen" → restaking ONLY
//     (NOT modular — eigenda stays in modular though).
//   - "chainlink" → oracles ONLY (the canonical narrative for an oracle).
//
// If a future contributor adds "celestia" to cosmos-appchains, this test
// will trip — forcing an explicit re-think of the taxonomy.
func TestMatchNarratives_NoConflicts(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		title     string
		summary   string
		wantOnly  []string // EXACT set the result must equal (order-insensitive)
		wantInc   []string // slugs that MUST be in the result (subset check)
		wantExcl  []string // slugs that MUST NOT be in the result
	}{
		{
			name:    "Celestia + EigenDA piece routes to modular only",
			title:   "Celestia and EigenDA partner on shared sequencer security",
			summary: "DA layer competition cools.",
			// "celestia" + "eigenda" both → modular. "eigen" is also a
			// keyword for restaking, but "eigenda" being a substring of
			// "eigenda" plus "eigen" being a substring of "eigenda" means
			// restaking ALSO fires. That's accepted: an EigenDA piece
			// genuinely touches both narratives. We just verify
			// cosmos-appchains is NOT triggered (celestia is NOT a cosmos-
			// appchains keyword).
			wantInc:  []string{"modular"},
			wantExcl: []string{"cosmos-appchains"},
		},
		{
			name:    "Chainlink oracle piece does NOT trigger restaking",
			title:   "Chainlink CCIP integrates with Aave for cross-chain lending",
			summary: "Oracle network expands utility surface.",
			wantInc:  []string{"oracles"},
			wantExcl: []string{"restaking", "modular"},
		},
		{
			name:    "Hyperliquid perp piece does NOT trigger solana-defi",
			title:   "Hyperliquid airdrop drives perp dex volume",
			summary: "On-chain perpetuals leader pulls ahead.",
			wantInc:  []string{"perp-dex"},
			wantExcl: []string{"solana-defi"},
		},
		{
			name:    "Helium piece does NOT trigger oracles or solana-defi",
			title:   "Helium HNT migration to Solana wraps up smoothly",
			summary: "DePIN flagship hits user-growth target.",
			// Helium DID migrate to Solana, but the canonical home is depin.
			// "solana defi" as a phrase isn't in the title/summary, so
			// solana-defi must NOT fire.
			wantInc:  []string{"depin"},
			wantExcl: []string{"oracles"},
		},
		{
			name:    "PYUSD article does NOT trigger memecoins",
			title:   "PYUSD payment stablecoin supply doubles in Q1",
			summary: "Circle responds with USDC roadmap.",
			wantOnly: []string{"stablecoins-regulation"},
		},
		{
			name:    "Generic Bitcoin halving piece does NOT trigger btc-etf",
			title:   "Bitcoin halving cycle: miners brace for hash rate drop",
			summary: "Mining difficulty already eased 3% post-event.",
			// No ETF mention → btc-etf must NOT fire even though the topic
			// is BTC-adjacent.
			wantInc:  []string{"bitcoin-halving"},
			wantExcl: []string{"btc-etf"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := MatchNarratives(tc.title, tc.summary)

			for _, w := range tc.wantInc {
				if !contains(got, w) {
					t.Errorf("MatchNarratives(%q, %q) = %v, missing required slug %q",
						tc.title, tc.summary, got, w)
				}
			}
			for _, x := range tc.wantExcl {
				if contains(got, x) {
					t.Errorf("MatchNarratives(%q, %q) = %v, unexpected slug %q (conflict-resolution gate)",
						tc.title, tc.summary, got, x)
				}
			}
			if tc.wantOnly != nil && !sameSet(got, tc.wantOnly) {
				t.Errorf("MatchNarratives(%q, %q) = %v, want exactly %v",
					tc.title, tc.summary, got, tc.wantOnly)
			}
		})
	}
}

// TestMatchNarratives_DictionarySize pins the absolute size of the dictionary
// so accidental deletions are caught. Phase 2A target: 16 narratives, 150-180
// total keyword phrases. Range check, not exact, since minor in-narrative
// tweaks are routine and shouldn't break this gate.
func TestMatchNarratives_DictionarySize(t *testing.T) {
	t.Parallel()

	if got := len(narrativeKeywords); got != 16 {
		t.Errorf("narrativeKeywords map has %d narratives, want 16", got)
	}

	totalPhrases := 0
	for slug, kws := range narrativeKeywords {
		if len(kws) < 7 {
			t.Errorf("narrative %q has only %d keyword phrases, want >= 7", slug, len(kws))
		}
		totalPhrases += len(kws)
	}
	if totalPhrases < 130 || totalPhrases > 200 {
		t.Errorf("narrativeKeywords totals %d phrases, want 130-200 (Phase 2A target)", totalPhrases)
	}
}

// ── helpers ──────────────────────────────────────────────────────────────

func contains(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}

// sameSet returns true if a and b contain the same elements regardless of order.
func sameSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	ac := append([]string(nil), a...)
	bc := append([]string(nil), b...)
	sort.Strings(ac)
	sort.Strings(bc)
	for i := range ac {
		if ac[i] != bc[i] {
			return false
		}
	}
	return true
}
