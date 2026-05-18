package narrative

import (
	"regexp"
	"strings"
)

// classifier.go — pure functions that map free-form news text into a set of
// narrative slugs and a list of asset tickers. Zero side effects, zero I/O,
// zero allocations beyond the returned slices.
//
// VERSION: v0 (keyword + regex). The point of this layer is to bootstrap the
// Narrative Radar with deterministic, easy-to-debug behaviour. Phase 2 will
// replace MatchNarratives with an LLM-graded classifier (multi-label, prompt
// against a fixed taxonomy, fall back to keywords on rate-limit). The
// signature returned here is intentionally compatible with that future
// version: (title, summary) → []string.
//
// The dictionary is package-private (narrativeKeywords) so callers cannot
// silently extend the canonical narrative set — any new narrative needs a
// code change here, which forces a code review of the keyword choices.

// narrativeKeywords maps a canonical slug (matches narrative_snapshots.narrative
// in migration 009) to the case-insensitive substrings that, if found in
// title+summary, mark the item as "about" that narrative.
//
// Choosing keyword phrases:
//   - Prefer multi-word phrases over single words to avoid false positives
//     ("ai token" rather than "ai" — the latter would tag every news item).
//   - Tickers and project names are the highest-precision keywords; generic
//     category terms ("rwa", "lst") catch the long tail.
//   - Case-insensitive match: every key here is lowercased once at startup
//     in the init() below, and the lookup compares against lowercased text.
var narrativeKeywords = map[string][]string{
	// Restaking / liquid restaking — driven by EigenLayer, Ethena, plus the
	// LRT/LST taxonomy used in DeFi research notes. Phase 2A expansion adds
	// the post-EigenLayer wave of LRT issuers.
	"restaking": {
		"eigenlayer", "restaking", "ethena", "eigen",
		"liquid restaking", "lrt", "lst",
		"kelp", "puffer", "swell", "renzo", "etherfi", "ether.fi",
	},
	// AI tokens — the "AI x crypto" thematic. Coverage spans the dominant
	// project tickers (FET/AGIX/RNDR/TAO) plus generic phrases. Phase 2A
	// adds Worldcoin (WLD) and Arkham as on-chain-data AI plays.
	"ai-tokens": {
		"fetch.ai", "render", "ocean protocol", "ai token",
		"agix", "rndr", "fet", "bittensor", "tao",
		"singularitynet", "worldcoin", "wld", "arkham",
	},
	// RWA / tokenised treasuries — Ondo, Mantra, real-estate tokenisation.
	// "tokenized treasur" is a prefix match (covers "treasury"/"treasuries").
	// Phase 2A adds T-bill / private-credit terminology used in research notes.
	"rwa": {
		"real world assets", "rwa", "tokenized treasur",
		"mantra", "ondo", "real estate token",
		"t-bill", "centrifuge", "chintai", "private credit",
		"real estate tokenization",
	},
	// Memecoins — primarily SOL/ETH ecosystem memes that move the tape. We
	// keep "shitcoin" for the editorial coverage of the same sector. Phase 2A
	// adds the 2024-2025 wave: pump.fun, brett, mog, michi, floki.
	"memecoins": {
		"memecoin", "dogwifhat", "wif", "popcat", "pepe",
		"bonk", "doge meta", "shitcoin",
		"pump.fun", "brett", "mog coin", "michi", "floki",
	},
	// Bitcoin L2 / Runes ecosystem — BTC scaling narratives. Phase 2A pulls
	// in the broader Ordinals / BRC-20 / inscription taxonomy plus newer L2s.
	"btc-l2": {
		"bitcoin l2", "btc l2", "stacks", "merlin",
		"bitlayer", "stx", "runes",
		"ordinals", "brc-20", "brc20", "bsquared", "bvm",
		"bitcoin layer 2",
	},
	// Solana DeFi — top apps that drove the 2024-2025 SOL DeFi flywheel.
	// Phase 2A adds the perp/aggregator/NFT-marketplace tickers commonly
	// used in CoinDesk Solana-section coverage.
	"solana-defi": {
		"jupiter", "jito", "raydium", "kamino",
		"marginfi", "solana defi",
		"jup", "jto", "drift protocol", "solend", "tensor", "magic eden",
	},
	// Modular blockchains / data availability — Celestia and the DA-layer
	// ecosystem (EigenDA, Dymension as Cosmos-RaaS). Phase 2A adds the
	// next-gen DA + execution-layer plays (Avail, Eclipse).
	"modular": {
		"celestia", "modular blockchain", "da layer",
		"eigenda", "dymension",
		"tia", "dym", "avail", "eclipse",
	},
	// ZK rollups — the major L2s + the generic "zk rollup" tag. Phase 2A
	// adds Linea, Aleo, Mina (zk-SNARK L1) and the generic ZK terminology.
	"zk": {
		"zksync", "starknet", "scroll",
		"polygon zkevm", "zk rollup",
		"strk", "linea", "zero knowledge", "aleo", "mina",
	},

	// ── Phase 2A: 8 NEW narratives ──────────────────────────────────────

	// BTC ETF flows / institutional Bitcoin — driven by the January 2024
	// spot-ETF approvals. Coverage almost always names BlackRock/IBIT,
	// Fidelity, or the Grayscale GBTC trust + outflows narrative.
	"btc-etf": {
		"bitcoin etf", "btc etf", "spot bitcoin etf",
		"blackrock", "ibit", "fidelity bitcoin",
		"etf inflow", "etf outflow", "grayscale", "gbtc",
	},
	// Bitcoin halving cycle / mining economics — supply-shock thesis,
	// hash-rate dynamics, miner-balance-sheet stress (capitulation).
	"bitcoin-halving": {
		"bitcoin halving", "halving cycle", "mining difficulty",
		"hash rate", "miner capitulation", "miner reserves",
		"bitcoin supply shock",
	},
	// Stablecoin laws / USDC/USDT regulation — MiCA, U.S. payment-stablecoin
	// bill, Tether/Circle. PYUSD/PayPal USD as the new entrant.
	"stablecoins-regulation": {
		"stablecoin regulation", "mica", "usdt", "tether",
		"usdc", "circle", "payment stablecoin", "stablecoin law",
		"ousd", "pyusd", "paypal usd",
	},
	// Web3 gaming / GameFi — the "play to earn" thematic. Heavy reliance
	// on project names (Axie, Immutable, Pixels, Ronin) since the generic
	// "gaming" term is too noisy outside this taxonomy.
	"gaming-metaverse": {
		"gamefi", "web3 gaming", "axie", "immutable", "imx",
		"pixels", "ronin", "sandbox", "decentraland",
		"play to earn", "p2e",
	},
	// Decentralized Physical Infrastructure (DePIN) — Helium, Filecoin,
	// Arweave, Theta, Livepeer. "iotex" stays multi-char to avoid the
	// substring-trap of bare "iot" matching "patriot"/"idiot".
	"depin": {
		"depin", "decentralized physical infrastructure",
		"helium", "hnt", "iotex", "filecoin", "fil",
		"arweave", "theta", "livepeer", "lpt",
	},
	// On-chain data feeds / oracles — Chainlink dominates coverage; Pyth
	// is the second-most-mentioned. Standalone "link" omitted on purpose
	// (would substring-match "linked"/"linking" in any English text).
	"oracles": {
		"chainlink", "pyth", "pyth network", "band protocol",
		"api3", "oracle network", "data feed", "tellor", "redstone",
	},
	// Perpetual DEXes / on-chain perps — Hyperliquid, dYdX, GMX, plus the
	// Synthetix v3 perp pivot. "perp dex" / "on-chain perpetuals" act as
	// the generic-phrase catch-all.
	"perp-dex": {
		"hyperliquid", "dydx", "gmx", "vertex", "aevo",
		"perp dex", "on-chain perpetuals", "synthetix", "snx",
	},
	// Cosmos / inter-chain ecosystem (appchains, IBC). Note: "celestia",
	// "dymension", "tia", "dym" stay in `modular` only — that's their
	// primary coverage frame. "injective"/"osmosis" stay multi-char to
	// avoid bare "inj" matching "injection"/"injury".
	"cosmos-appchains": {
		"cosmos", "atom", "osmosis", "osmo", "injective",
		"kava", "evmos", "appchain", "ibc",
	},
}

// tickerWhitelist is the set of asset tickers ExtractTickers will accept when
// found as a bare word (without a leading "$"). Kept small (~30 tickers) on
// purpose — every entry here can produce a false positive on an English word
// that happens to be uppercase, so we restrict to symbols that (a) trade with
// real volume and (b) are unlikely to appear as ALL-CAPS English ("PEPE" the
// memecoin vs "PEPE" the acronym — accepted; "DOT" the network vs "DOT" the
// punctuation — accepted because uppercase/word-boundary makes it rare in news).
//
// $-prefixed tickers ($BTC, $ETH, …) are accepted regardless of this list,
// so adding a niche ticker is only required when readers commonly drop the $.
var tickerWhitelist = map[string]bool{
	"BTC":   true,
	"ETH":   true,
	"SOL":   true,
	"BNB":   true,
	"XRP":   true,
	"DOGE":  true,
	"ADA":   true,
	"AVAX":  true,
	"DOT":   true,
	"LINK":  true,
	"MATIC": true,
	"LTC":   true,
	"BCH":   true,
	"UNI":   true,
	"ATOM":  true,
	"NEAR":  true,
	"APT":   true,
	"SUI":   true,
	"TRX":   true,
	"SHIB":  true,
	"OP":    true,
	"ARB":   true,
	"PEPE":  true,
	"WIF":   true,
	"BONK":  true,
	"FET":   true,
	"AGIX":  true,
	"RNDR":  true,
	"TAO":   true,
	"TIA":   true, // Celestia
	"STX":   true, // Stacks
	"ONDO":  true,
	// Solana DeFi narrative tokens — referenced by name in coindesk/cointelegraph
	// articles ("Jito Labs launches…", "Jupiter integrates…").
	"JTO":  true, // Jito
	"JUP":  true, // Jupiter
	"RAY":  true, // Raydium
	"KMNO": true, // Kamino
	// Restaking + Ethena (referenced in restaking narrative dictionary).
	"EIGEN": true, // EigenLayer
	"ENA":   true, // Ethena
	// Phase 2A new tickers.
	"WLD":  true, // Worldcoin (ai-tokens)
	"HNT":  true, // Helium (depin)
	"FIL":  true, // Filecoin (depin)
	"LPT":  true, // Livepeer (depin)
	"GMX":  true, // GMX (perp-dex)
	"SNX":  true, // Synthetix (perp-dex)
	"DYDX": true, // dYdX (perp-dex)
	"OSMO": true, // Osmosis (cosmos-appchains)
	"INJ":  true, // Injective (cosmos-appchains)
	"KAVA": true, // Kava (cosmos-appchains)
	"DYM":  true, // Dymension (modular)
	"PYTH": true, // Pyth (oracles)
	"USDC": true, // Circle USDC (stablecoins-regulation) — "$USDC" in articles
	"USDT": true, // Tether USDT (stablecoins-regulation)
	"GBTC": true, // Grayscale BTC trust (btc-etf)
	"IBIT": true, // BlackRock spot BTC ETF (btc-etf)
}

// nameToTicker maps a lowercase coin/protocol NAME to its ticker symbol.
// Path 3 of ExtractTickers consults this so "Jito Labs launches…" yields
// JTO, "as activity heats up on Solana" yields SOL, etc — the v0 limitation
// that articles use names not symbols was real-world hit on day-one smoke.
//
// Keep this map in sync with narrativeKeywords: any name we keyword-match
// to detect a narrative should ideally have a ticker mapping here too, so
// related_assets isn't empty for matched articles.
var nameToTicker = map[string]string{
	// Mainstream coins (whitelist's full names).
	"bitcoin":   "BTC",
	"ethereum":  "ETH",
	"solana":    "SOL",
	"ripple":    "XRP",
	"dogecoin":  "DOGE",
	"cardano":   "ADA",
	"avalanche": "AVAX",
	"polkadot":  "DOT",
	"chainlink": "LINK",
	"polygon":   "MATIC",
	"litecoin":  "LTC",
	"uniswap":   "UNI",
	"cosmos":    "ATOM",
	"aptos":     "APT",
	"arbitrum":  "ARB",
	"optimism":  "OP",
	"celestia":  "TIA",
	"stacks":    "STX",
	"ondo":      "ONDO",
	// Solana DeFi.
	"jito":     "JTO",
	"jupiter":  "JUP",
	"raydium":  "RAY",
	"kamino":   "KMNO",
	// Restaking + adjacent.
	"eigenlayer": "EIGEN",
	"ethena":     "ENA",
	// AI tokens.
	"fetch.ai":  "FET",
	"render":    "RNDR",
	"bittensor": "TAO",
	"worldcoin": "WLD",
	// Phase 2A — DePIN.
	"helium":    "HNT",
	"filecoin":  "FIL",
	"livepeer":  "LPT",
	// Phase 2A — Perp DEX. Hyperliquid has no listed spot token at the time
	// of this dictionary update — when one ships we add it here.
	"dydx":      "DYDX",
	"synthetix": "SNX",
	// Phase 2A — Cosmos appchains.
	"osmosis":   "OSMO",
	"injective": "INJ",
	"kava":      "KAVA",
	"dymension": "DYM",
	// Phase 2A — Oracles.
	"pyth": "PYTH",
}

// dollarTickerRe matches a leading "$" followed by 2-6 uppercase letters
// followed by a word boundary, e.g. "$BTC", "$ETH". Group 1 is the ticker.
var dollarTickerRe = regexp.MustCompile(`\$([A-Z]{2,6})\b`)

// bareTickerRe matches a 2-6 uppercase-letter word at a word boundary. Used
// only against the whitelist — without that filter every uppercase noun
// ("CEO", "FBI") would become a "ticker".
var bareTickerRe = regexp.MustCompile(`\b([A-Z]{2,6})\b`)

// MatchNarratives scans an item's title+summary against the package-private
// narrative dictionary and returns every narrative slug whose keywords appear.
// One item can match 0, 1, or many narratives.
//
// Match semantics:
//   - Substring (case-insensitive). "EigenLayer" inside "EigenLayer announces…"
//     matches "eigenlayer".
//   - Order of returned slugs is the iteration order of the dictionary, which
//     is deterministic only if the caller does not depend on it. We
//     deduplicate within one call but do NOT sort — callers that need stable
//     ordering should sort the result themselves.
//   - Empty input returns an empty (non-nil) slice.
func MatchNarratives(title, summary string) []string {
	combined := strings.ToLower(title + " " + summary)
	if strings.TrimSpace(combined) == "" {
		return []string{}
	}

	out := make([]string, 0, 2) // most items match 0-2 narratives
	for slug, keywords := range narrativeKeywords {
		for _, kw := range keywords {
			if strings.Contains(combined, kw) {
				out = append(out, slug)
				break // one keyword is enough — don't double-add this slug
			}
		}
	}
	return out
}

// ExtractTickers returns the set of asset tickers mentioned in the input text.
// Three extraction paths, run in order; first occurrence of a ticker wins:
//
//  1. $-prefixed: any "$XXX" with 2-6 uppercase letters is accepted (e.g.
//     "$BTC", "$EIGEN"). High-precision — readers who write "$" in front
//     of a ticker almost always mean the asset.
//
//  2. Bare uppercase: a word matching /\b[A-Z]{2,6}\b/ is accepted only if
//     the word is in tickerWhitelist. Without the whitelist, "FBI", "CEO",
//     "USA" etc. would all be returned as tickers.
//
//  3. Coin name: case-insensitive match against nameToTicker. Catches
//     headlines like "Jito Labs launches …" or "activity heats up on
//     Solana" where the article uses the project name, not the symbol.
//
// Result is deduplicated and order-preserving (first occurrence across all
// paths wins). Empty input returns an empty (non-nil) slice.
func ExtractTickers(text string) []string {
	if strings.TrimSpace(text) == "" {
		return []string{}
	}
	seen := make(map[string]bool, 4)
	out := make([]string, 0, 4)

	// Path 1: $-prefixed tickers.
	for _, m := range dollarTickerRe.FindAllStringSubmatch(text, -1) {
		t := m[1]
		if !seen[t] {
			seen[t] = true
			out = append(out, t)
		}
	}
	// Path 2: bare-uppercase-against-whitelist.
	for _, m := range bareTickerRe.FindAllStringSubmatch(text, -1) {
		t := m[1]
		if !tickerWhitelist[t] {
			continue
		}
		if !seen[t] {
			seen[t] = true
			out = append(out, t)
		}
	}
	// Path 3: coin-name match (case-insensitive). We lowercase the whole
	// input once and then test each name as a substring with manual word-
	// boundary checks to avoid "ondo" matching inside "fundo" or similar.
	// Defensive guard: skip empty ticker values (placeholder rows for names
	// whose token isn't listed yet, e.g. Hyperliquid pre-TGE).
	lower := strings.ToLower(text)
	for name, ticker := range nameToTicker {
		if ticker == "" {
			continue
		}
		if !containsWordCI(lower, name) {
			continue
		}
		if !seen[ticker] {
			seen[ticker] = true
			out = append(out, ticker)
		}
	}
	return out
}

// containsWordCI returns true if `needle` appears in `haystack` (already
// lowercased) as a whole word. "Whole word" means the surrounding chars
// are not letters/digits — handles "Solana." (ends in dot) and "on Solana"
// (preceded by space) but rejects "fundo" containing "ondo".
func containsWordCI(haystack, needle string) bool {
	idx := 0
	for {
		i := strings.Index(haystack[idx:], needle)
		if i < 0 {
			return false
		}
		start := idx + i
		end := start + len(needle)
		// Boundary check: char before start and after end must NOT be
		// a letter/digit. Treat string boundaries as valid.
		if (start == 0 || !isAlnum(haystack[start-1])) &&
			(end == len(haystack) || !isAlnum(haystack[end])) {
			return true
		}
		idx = start + 1
	}
}

func isAlnum(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9')
}

// init lowercases every keyword once at package load. The dictionary is
// declared with mixed case for readability; we want the runtime path to
// avoid re-lowercasing on every call.
func init() {
	for slug, kws := range narrativeKeywords {
		lc := make([]string, len(kws))
		for i, kw := range kws {
			lc[i] = strings.ToLower(kw)
		}
		narrativeKeywords[slug] = lc
	}
}

// AllNarrativeSlugs returns every canonical narrative slug known to the
// classifier, in dictionary-iteration order. Used by the worker so it knows
// which slugs to compute snapshots for, even when no mentions arrived this
// tick (a narrative going from 5 mentions to 0 is itself a signal — we want
// the snapshot row written so the growth_pct shows -100%).
func AllNarrativeSlugs() []string {
	out := make([]string, 0, len(narrativeKeywords))
	for slug := range narrativeKeywords {
		out = append(out, slug)
	}
	return out
}
