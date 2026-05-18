package narrative

// importance_rule.go — deterministic keyword-based importance classifier.
// Used as a fallback when ANTHROPIC_API_KEY is empty (so the system has sane
// behaviour out-of-the-box) and as the cheap path in tests.
//
// Design notes:
//   - Patterns are evaluated in priority order. The FIRST pattern that hits
//     wins. Order matters: macro/geopolitics > regulation > exchange >
//     partnership > altcoin > other. A "Fed crackdown on Binance" headline
//     should land in macro (highest urgency), not exchange.
//   - Every match yields a fixed (score, category, why) triple. We don't
//     try to interpolate between patterns — the rule classifier is a
//     bootstrap, not a replacement.
//   - All keyword phrases are lowercase; the input is lowercased once at
//     entry. Substring match (not regex) keeps this fast (O(N*M) with tiny
//     N, M).
//
// Score bands (matching the docstring on the interface):
//   - macro / geopolitics  → 70-95
//   - regulation           → 60-85
//   - exchange (hacks/btk) → 50-90
//   - exchange (listings)  → 30-50
//   - partnership          → 30-50
//   - altcoin / other      → 10-30

import (
	"context"
	"strings"
)

// RuleClassifier is the deterministic keyword fallback. No state — safe for
// concurrent use by definition.
type RuleClassifier struct{}

// rulePattern groups one keyword set into a (score, category, why) outcome.
// "why" is the rule name, not a sentence — kept short so the importance_why
// column doesn't fill with boilerplate.
type rulePattern struct {
	keywords []string
	score    int
	category string
	why      string
}

// rulePatterns is evaluated in declaration order. First match wins. Reorder
// with care — the priority above is a contract.
var rulePatterns = []rulePattern{
	// ── macro (70-95) ──────────────────────────────────────────────────
	// Highest-urgency geopolitical / macro tail-risk events. These are
	// ahead of "fed cuts rates" because a war or default is bigger than
	// any single rate decision.
	{
		keywords: []string{"declares war", "nuclear", "world war", "default", "sanctions"},
		score:    90,
		category: CategoryGeopolitics,
		why:      "rule:geopolitics-tail-risk",
	},
	// Wars, military escalations — high market impact.
	{
		keywords: []string{
			"war ", "invasion", "missile strike", "military operation",
			"trump declares", "putin orders", "biden orders",
		},
		score:    85,
		category: CategoryGeopolitics,
		why:      "rule:geopolitics-conflict",
	},
	// Fed / central bank rate moves — top-tier macro.
	{
		keywords: []string{
			"fed cut", "fed cuts", "fed hike", "fed hikes",
			"fed raises", "fed lowers",
			"rate cut", "rate hike", "rate decision",
			"federal reserve",
		},
		score:    85,
		category: CategoryMacro,
		why:      "rule:macro-rates",
	},
	// ECB / BoJ / BoE — slightly lower than the Fed but still macro.
	{
		keywords: []string{"ecb cut", "ecb hike", "boj cut", "boj hike", "bank of japan", "european central bank"},
		score:    75,
		category: CategoryMacro,
		why:      "rule:macro-foreign-cb",
	},
	// CPI / inflation prints — lower urgency than the rate decision itself
	// (the trade is reactive). Bottom of the macro band.
	{
		keywords: []string{"cpi", "ppi", "pce", "nonfarm payrolls", "jobs report", "gdp", "unemployment"},
		score:    70,
		category: CategoryMacro,
		why:      "rule:macro-data-release",
	},
	// Election / presidential / White House content — political risk.
	// Below outright war but above plain rate decisions if it's a Trump-
	// level surprise.
	{
		keywords: []string{
			"president trump", "white house", "presidential election",
			"election result", "tariffs", "trade war",
		},
		score:    75,
		category: CategoryGeopolitics,
		why:      "rule:geopolitics-political",
	},

	// ── exchange — hack/bankruptcy (50-90) ────────────────────────────
	// Big exchange failures move the entire crypto market. Top of the
	// non-macro band.
	{
		keywords: []string{
			"hack", "hacked", "exploit", "exploited",
			"bankruptcy", "bankrupt", "files chapter 11",
			"insolven", "frozen withdrawals", "halts withdrawals",
		},
		score:    80,
		category: CategoryExchange,
		why:      "rule:exchange-failure",
	},

	// ── regulation (60-85) ────────────────────────────────────────────
	// SEC actions, ETF approvals, lawsuits. ETF approvals especially can
	// re-rate the entire crypto market overnight.
	{
		keywords: []string{
			"etf approval", "etf approved", "etf rejected", "etf denied",
			"spot etf",
		},
		score:    80,
		category: CategoryRegulation,
		why:      "rule:regulation-etf",
	},
	{
		keywords: []string{
			"sec sues", "sec charges", "sec fines", "sec settlement",
			"sec investigation", "doj charges", "doj indictment",
			"lawsuit", "settles with sec", "guilty plea",
		},
		score:    70,
		category: CategoryRegulation,
		why:      "rule:regulation-enforcement",
	},
	{
		keywords: []string{
			"crypto regulation", "regulatory framework", "stablecoin bill",
			"mica", "fit21",
		},
		score:    65,
		category: CategoryRegulation,
		why:      "rule:regulation-policy",
	},

	// ── exchange listings (30-50) ────────────────────────────────────
	// New listing news. Mid-tier — moves the listed coin, not the market.
	{
		keywords: []string{"lists on", "listing on", "binance lists", "coinbase lists", "kraken lists"},
		score:    40,
		category: CategoryExchange,
		why:      "rule:exchange-listing",
	},

	// ── partnership (30-50) ──────────────────────────────────────────
	// Collabs, integrations. Bottom of the mid-tier.
	{
		keywords: []string{
			"partners with", "partnership with", "integration with",
			"integrates with", "collaboration with", "joins forces",
			"strategic partnership",
		},
		score:    40,
		category: CategoryPartnership,
		why:      "rule:partnership",
	},

	// ── altcoin (10-30) ──────────────────────────────────────────────
	// Small-token noise. Lowest tier.
	{
		keywords: []string{
			"memecoin", "shitcoin", "rugpull", "rug pull",
			"new token launch", "token launch on",
			"airdrop", "ico",
		},
		score:    20,
		category: CategoryAltcoin,
		why:      "rule:altcoin",
	},
}

// Classify is the RuleClassifier implementation. ctx is accepted for
// interface conformance but never blocks — the classifier is in-memory
// substring matching. url is unused (no caching at this layer; that's the
// CachedClassifier's job).
func (r *RuleClassifier) Classify(ctx context.Context, title, summary, url string) (Importance, error) {
	if err := validateInputs(title, summary); err != nil {
		return emptyImportance(), nil // soft fail: no point logging "empty input" everywhere
	}
	// Honour cancellation even though we don't block. A caller deadlining
	// the whole worker tick should see this surface; otherwise we'd be the
	// one path that ignores ctx.
	if err := ctx.Err(); err != nil {
		return Importance{}, err
	}

	combined := strings.ToLower(title + " " + summary)

	for _, p := range rulePatterns {
		for _, kw := range p.keywords {
			if strings.Contains(combined, kw) {
				return Importance{
					Score:    clampScore(p.score),
					Category: normalizeCategory(p.category),
					Why:      p.why,
				}, nil
			}
		}
	}

	// No rule matched → treat as low-importance "other". Score 15 (mid-low)
	// is a deliberate pick: too high biases the trend score on uncategorised
	// noise, too low (e.g. 0) makes the radar dismiss legitimate signals
	// the LLM would otherwise grade higher than the keyword-only fallback.
	return Importance{
		Score:    15,
		Category: CategoryOther,
		Why:      "rule:no-match",
	}, nil
}
