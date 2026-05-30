package whale

// source_etherscan.go — free-tier ETH + ERC-20 stablecoin whale source.
//
// Uses the Etherscan V2 API (V1 is dead since 08.2025). Base:
//   https://api.etherscan.io/v2/api?chainid=1
// Two actions per watched hot-wallet:
//   module=account&action=txlist    → native ETH transfers
//   module=account&action=tokentx   → ERC-20 transfers (USDT/USDC)
//
// Attribution: for each transfer we look up `to`/`from` in the ExchangeRegistry:
//   to_addr   ∈ registry → inflow  (deposit TO exchange,  bearish ▼)
//   from_addr ∈ registry → outflow (withdrawal FROM exch, bullish ▲)
//   neither              → neutral (skipped — we only care about exchange flow)
//
// Narrow-block polling (discovery §5): from 2026-07-01 Etherscan caps results
// at 1k/request, so we track the last-seen block per (address, action) and
// query startblock=lastBlock+1 each tick — bounded windows, no re-scan.
//
// Rate limit: free key = 5 req/s. We gate at ≤4 req/s via a simple time-based
// token bucket (minInterval between requests) so a burst of hot-wallets × 2
// actions can't trip the limit.
//
// USD valuation: stablecoins (USDT/USDC) are valued ≈ amount_native. ETH is
// amount_native × spot, where spot is the Binance ETHUSDT ticker cached ~60s;
// 0 when unavailable (UI renders "—"). All HTTP/parse errors return the
// partial set + an error for the caller to log — never panic.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	// etherscanV2Base is the V2 endpoint (chainid is appended per request).
	etherscanV2Base = "https://api.etherscan.io/v2/api"
	// ethChainID is Ethereum mainnet for the V2 chainid param.
	ethChainID = "1"

	// etherscanMinInterval gates requests to ≤4 req/s (250ms apart). Free key
	// allows 5/s; we leave headroom so clock jitter can't trip the limit.
	etherscanMinInterval = 250 * time.Millisecond

	// etherscanPageSize caps rows per txlist/tokentx call. 1000 is the
	// post-2026-07-01 ceiling; narrow-block polling keeps each window small
	// so we rarely approach it.
	etherscanPageSize = 1000

	// spotCacheTTL caps how long a cached ETH spot price is reused. 60s keeps
	// USD figures roughly live without hammering Binance each tick.
	spotCacheTTL = 60 * time.Second
)

// Known ERC-20 stablecoin contracts (lowercase) and their decimals. Only
// stablecoins are tracked for tokentx — they carry the bulk of labeled
// exchange flow and their USD value ≈ amount_native (no spot lookup needed).
const (
	contractUSDT = "0xdac17f958d2ee523a2206206994597c13d831ec7" // 6 decimals
	contractUSDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48" // 6 decimals
)

// stableTokens maps lowercase contract → (asset symbol, decimals) for the
// ERC-20s we value as ≈$1.
var stableTokens = map[string]struct {
	Asset    string
	Decimals int
}{
	contractUSDT: {Asset: "USDT", Decimals: 6},
	contractUSDC: {Asset: "USDC", Decimals: 6},
}

// EtherscanSource implements WhaleFlowSource for ETH + stablecoins.
type EtherscanSource struct {
	// APIKey is the free Etherscan key (env ETHERSCAN_API_KEY). Empty key
	// still issues requests but Etherscan rate-limits anonymous calls hard —
	// production should always set it.
	APIKey string
	// HTTP is the client used for all calls. nil → a 10s-timeout default.
	HTTP *http.Client
	// reg attributes addresses to exchanges. nil → NewStaticRegistry().
	reg ExchangeRegistry

	// lastBlock tracks the last-seen block per "addr|action" key for
	// narrow-block polling. Guarded by mu.
	lastBlock map[string]uint64
	mu        sync.Mutex

	// nextAllowed gates the rate limiter — the earliest time the next request
	// may fire. Guarded by rlMu.
	nextAllowed time.Time
	rlMu        sync.Mutex

	// spotPrice / spotAt cache the ETH/USD spot for spotCacheTTL. Guarded by
	// spotMu.
	spotPrice float64
	spotAt    time.Time
	spotMu    sync.Mutex
}

// NewEtherscanSource builds a source with sane defaults. A nil registry falls
// back to the curated static registry; a nil HTTP client gets a 10s timeout.
func NewEtherscanSource(apiKey string, httpClient *http.Client, reg ExchangeRegistry) *EtherscanSource {
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 10 * time.Second}
	}
	if reg == nil {
		reg = NewStaticRegistry()
	}
	return &EtherscanSource{
		APIKey:    apiKey,
		HTTP:      httpClient,
		reg:       reg,
		lastBlock: map[string]uint64{},
	}
}

// Name returns the honest source badge.
func (e *EtherscanSource) Name() string { return "etherscan" }

// Chains returns the chains this source serves (ETH only).
func (e *EtherscanSource) Chains() []Chain { return []Chain{ChainETH} }

// registry returns the configured registry or the static default.
func (e *EtherscanSource) registry() ExchangeRegistry {
	if e.reg != nil {
		return e.reg
	}
	return NewStaticRegistry()
}

// client returns the configured HTTP client or a 10s default.
func (e *EtherscanSource) client() *http.Client {
	if e.HTTP != nil {
		return e.HTTP
	}
	return &http.Client{Timeout: 10 * time.Second}
}

// scrubKey removes the API key from a string before it can reach logs. Go's
// *url.Error embeds the full request URL (including apikey=...) in its Error()
// output, so any transport error would otherwise leak the key. No-op when the
// key is empty.
func (e *EtherscanSource) scrubKey(s string) string {
	if e.APIKey == "" {
		return s
	}
	return strings.ReplaceAll(s, e.APIKey, "***")
}

// rateGate blocks until the rate limiter permits the next request (or ctx is
// cancelled). Simple single-token bucket: each call reserves the next slot
// minInterval after the previous one.
func (e *EtherscanSource) rateGate(ctx context.Context) error {
	e.rlMu.Lock()
	now := time.Now()
	wait := time.Duration(0)
	if e.nextAllowed.After(now) {
		wait = e.nextAllowed.Sub(now)
	}
	// Reserve the slot: next request may fire one interval after this one's
	// scheduled start.
	start := now
	if wait > 0 {
		start = e.nextAllowed
	}
	e.nextAllowed = start.Add(etherscanMinInterval)
	e.rlMu.Unlock()

	if wait <= 0 {
		return nil
	}
	timer := time.NewTimer(wait)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// etherscanResp is the envelope every V2 account call returns. status="1" on
// success; status="0" with message="No transactions found" is the empty-but-
// OK case (not an error). result is the raw array we decode per-action.
type etherscanResp struct {
	Status  string          `json:"status"`
	Message string          `json:"message"`
	Result  json.RawMessage `json:"result"`
}

// txlistEntry is one row from action=txlist (native ETH transfers).
type txlistEntry struct {
	BlockNumber string `json:"blockNumber"`
	TimeStamp   string `json:"timeStamp"`
	Hash        string `json:"hash"`
	From        string `json:"from"`
	To          string `json:"to"`
	Value       string `json:"value"` // wei
	IsError     string `json:"isError"`
}

// tokentxEntry is one row from action=tokentx (ERC-20 transfers).
type tokentxEntry struct {
	BlockNumber     string `json:"blockNumber"`
	TimeStamp       string `json:"timeStamp"`
	Hash            string `json:"hash"`
	From            string `json:"from"`
	To              string `json:"to"`
	Value           string `json:"value"`
	ContractAddress string `json:"contractAddress"`
	TokenSymbol     string `json:"tokenSymbol"`
	TokenDecimal    string `json:"tokenDecimal"`
}

// RecentWhaleTransfers polls txlist + tokentx for every watched hot-wallet,
// attributes each transfer to an exchange, values it in USD, filters by
// minUSD, and returns the newest-first slice capped at `limit`.
//
// Only ChainETH is served; any other chain returns (nil, nil) — the worker
// fans BTC out to MempoolSource instead. Errors from individual address polls
// are accumulated; the partial set collected so far is still returned.
func (e *EtherscanSource) RecentWhaleTransfers(ctx context.Context, chain Chain, minUSD float64, limit int) ([]WhaleTransfer, error) {
	if chain != ChainETH {
		return nil, nil
	}
	if limit <= 0 {
		return []WhaleTransfer{}, nil
	}

	wallets := e.registry().HotWallets(ChainETH)
	if len(wallets) == 0 {
		return []WhaleTransfer{}, nil
	}

	spot := e.ethSpotUSD(ctx) // 0 if unavailable → ETH amount_usd stays 0
	var (
		out  []WhaleTransfer
		errs []string
	)

	for addr, exch := range wallets {
		// Native ETH.
		ethTransfers, err := e.fetchTxlist(ctx, addr, exch, spot)
		if err != nil {
			errs = append(errs, fmt.Sprintf("txlist %s: %v", addr, err))
		}
		out = append(out, ethTransfers...)

		// ERC-20 stablecoins.
		tokenTransfers, err := e.fetchTokenTx(ctx, addr, exch)
		if err != nil {
			errs = append(errs, fmt.Sprintf("tokentx %s: %v", addr, err))
		}
		out = append(out, tokenTransfers...)
	}

	// Filter by minUSD. A transfer with amount_usd=0 (unknown spot) is kept
	// only when minUSD<=0 — otherwise we can't assert it clears the floor.
	filtered := out[:0]
	for _, t := range out {
		if minUSD > 0 && t.AmountUSD < minUSD {
			continue
		}
		filtered = append(filtered, t)
	}
	out = filtered

	// Newest-first, then cap to limit.
	sortTransfersDesc(out)
	if len(out) > limit {
		out = out[:limit]
	}

	if len(errs) > 0 {
		return out, fmt.Errorf("etherscan: %d partial errors: %s", len(errs), strings.Join(errs, "; "))
	}
	return out, nil
}

// fetchTxlist pulls native ETH transfers for one watched address since the
// last-seen block, attributing direction by which side the watched (exchange)
// address is on.
func (e *EtherscanSource) fetchTxlist(ctx context.Context, addr, exch string, spot float64) ([]WhaleTransfer, error) {
	startBlock := e.getLastBlock(addr, "txlist")
	params := url.Values{}
	params.Set("chainid", ethChainID)
	params.Set("module", "account")
	params.Set("action", "txlist")
	params.Set("address", addr)
	params.Set("startblock", strconv.FormatUint(startBlock, 10))
	params.Set("endblock", "99999999")
	params.Set("sort", "desc")
	params.Set("offset", strconv.Itoa(etherscanPageSize))
	params.Set("page", "1")

	raw, err := e.doRequest(ctx, params)
	if err != nil {
		return nil, err
	}
	var entries []txlistEntry
	if uerr := json.Unmarshal(raw, &entries); uerr != nil {
		return nil, fmt.Errorf("decode txlist: %w", uerr)
	}

	wallets := e.registry().HotWallets(ChainETH)
	out := make([]WhaleTransfer, 0, len(entries))
	var maxBlock uint64
	for _, en := range entries {
		if en.IsError == "1" {
			continue // reverted tx — not a real transfer
		}
		blk := parseUint(en.BlockNumber)
		if blk > maxBlock {
			maxBlock = blk
		}
		amt := weiToEth(en.Value)
		if amt == 0 {
			continue // zero-value (contract call) — not a whale transfer
		}
		dir, matchedExch := attribute(en.From, en.To, exch, wallets)
		if dir == FlowNeutral {
			continue // counterparty isn't a known exchange — skip
		}
		usd := 0.0
		if spot > 0 {
			usd = amt * spot
		}
		out = append(out, WhaleTransfer{
			Chain:        ChainETH,
			TxHash:       en.Hash,
			Timestamp:    unixToTime(en.TimeStamp),
			FromAddr:     strings.ToLower(en.From),
			ToAddr:       strings.ToLower(en.To),
			Asset:        "ETH",
			AmountNative: amt,
			AmountUSD:    usd,
			Direction:    dir,
			Exchange:     matchedExch,
		})
	}
	if maxBlock > 0 {
		e.setLastBlock(addr, "txlist", maxBlock)
	}
	return out, nil
}

// fetchTokenTx pulls ERC-20 transfers for one watched address, keeping only
// the tracked stablecoins (USDT/USDC) valued ≈ amount_native.
func (e *EtherscanSource) fetchTokenTx(ctx context.Context, addr, exch string) ([]WhaleTransfer, error) {
	startBlock := e.getLastBlock(addr, "tokentx")
	params := url.Values{}
	params.Set("chainid", ethChainID)
	params.Set("module", "account")
	params.Set("action", "tokentx")
	params.Set("address", addr)
	params.Set("startblock", strconv.FormatUint(startBlock, 10))
	params.Set("endblock", "99999999")
	params.Set("sort", "desc")
	params.Set("offset", strconv.Itoa(etherscanPageSize))
	params.Set("page", "1")

	raw, err := e.doRequest(ctx, params)
	if err != nil {
		return nil, err
	}
	var entries []tokentxEntry
	if uerr := json.Unmarshal(raw, &entries); uerr != nil {
		return nil, fmt.Errorf("decode tokentx: %w", uerr)
	}

	wallets := e.registry().HotWallets(ChainETH)
	out := make([]WhaleTransfer, 0, len(entries))
	var maxBlock uint64
	for _, en := range entries {
		blk := parseUint(en.BlockNumber)
		if blk > maxBlock {
			maxBlock = blk
		}
		tok, ok := stableTokens[strings.ToLower(en.ContractAddress)]
		if !ok {
			continue // not a tracked stablecoin
		}
		// Prefer the contract-derived decimals; fall back to the response's
		// tokenDecimal field if the contract map is ever extended.
		decimals := tok.Decimals
		if d := parseInt(en.TokenDecimal); d > 0 {
			decimals = d
		}
		amt := scaleByDecimals(en.Value, decimals)
		if amt == 0 {
			continue
		}
		dir, matchedExch := attribute(en.From, en.To, exch, wallets)
		if dir == FlowNeutral {
			continue
		}
		out = append(out, WhaleTransfer{
			Chain:        ChainETH,
			TxHash:       en.Hash,
			Timestamp:    unixToTime(en.TimeStamp),
			FromAddr:     strings.ToLower(en.From),
			ToAddr:       strings.ToLower(en.To),
			Asset:        tok.Asset,
			AmountNative: amt,
			AmountUSD:    amt, // stablecoin ≈ $1
			Direction:    dir,
			Exchange:     matchedExch,
		})
	}
	if maxBlock > 0 {
		e.setLastBlock(addr, "tokentx", maxBlock)
	}
	return out, nil
}

// doRequest runs one rate-gated GET against the V2 API and returns the raw
// `result` array bytes. A status="0" envelope with the "No transactions found"
// message is treated as an empty result (`[]`), not an error. Other status="0"
// messages (e.g. "Max rate limit reached", "Invalid API Key") are errors.
func (e *EtherscanSource) doRequest(ctx context.Context, params url.Values) (json.RawMessage, error) {
	if e.APIKey != "" {
		params.Set("apikey", e.APIKey)
	}
	if err := e.rateGate(ctx); err != nil {
		return nil, err
	}

	reqURL := etherscanV2Base + "?" + params.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %s", e.scrubKey(err.Error()))
	}
	resp, err := e.client().Do(req)
	if err != nil {
		// *url.Error embeds the full request URL (incl. apikey=...) in its
		// Error() string; scrub the key so a transport failure can't leak it.
		return nil, fmt.Errorf("http: %s", e.scrubKey(err.Error()))
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20)) // cap 8 MB
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("http status %d", resp.StatusCode)
	}

	var env etherscanResp
	if uerr := json.Unmarshal(body, &env); uerr != nil {
		return nil, fmt.Errorf("decode envelope: %w", uerr)
	}
	if env.Status != "1" {
		// "No transactions found" → empty, not an error.
		if strings.Contains(strings.ToLower(env.Message), "no transactions found") {
			return json.RawMessage("[]"), nil
		}
		return nil, fmt.Errorf("etherscan status=%s message=%s", env.Status, env.Message)
	}
	if len(env.Result) == 0 {
		return json.RawMessage("[]"), nil
	}
	return env.Result, nil
}

// Netflow computes inflow/outflow USD sums over `window` for one (exchange,
// asset) by pulling RecentWhaleTransfers and summing. exchange="" sums across
// all known exchanges. Always Partial=true — free-tier sees labeled wallets
// only. This is provided for interface completeness; the worker computes its
// snapshot netflow from the persisted store instead.
func (e *EtherscanSource) Netflow(ctx context.Context, chain Chain, exchange, asset string, window time.Duration) (ExchangeNetflow, error) {
	res := ExchangeNetflow{
		Chain:    chain,
		Asset:    asset,
		Exchange: exchange,
		Window:   window,
		Source:   e.Name(),
		Partial:  true,
	}
	if chain != ChainETH {
		// Non-ETH not served by this source.
		return res, nil
	}
	// Pull a generous page; the worker's real path uses the store, so this is
	// a best-effort convenience.
	transfers, err := e.RecentWhaleTransfers(ctx, chain, 0, etherscanPageSize)
	if err != nil {
		// Partial set still usable — return what we summed plus the error.
		err = fmt.Errorf("Netflow: %w", err)
	}
	cutoff := time.Now().Add(-window)
	for _, t := range transfers {
		if asset != "" && t.Asset != asset {
			continue
		}
		if exchange != "" && t.Exchange != exchange {
			continue
		}
		if t.Timestamp.Before(cutoff) {
			continue
		}
		switch t.Direction {
		case FlowInflow:
			res.InflowUSD += t.AmountUSD
			res.TxCount++
		case FlowOutflow:
			res.OutflowUSD += t.AmountUSD
			res.TxCount++
		}
	}
	return res, err
}

// ── narrow-block state helpers ────────────────────────────────────────────

func (e *EtherscanSource) getLastBlock(addr, action string) uint64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.lastBlock == nil {
		e.lastBlock = map[string]uint64{}
		return 0
	}
	return e.lastBlock[addr+"|"+action]
}

func (e *EtherscanSource) setLastBlock(addr, action string, block uint64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.lastBlock == nil {
		e.lastBlock = map[string]uint64{}
	}
	key := addr + "|" + action
	// startblock = lastBlock+1 next tick, so store the max block we saw.
	if block > e.lastBlock[key] {
		e.lastBlock[key] = block
	}
}

// ── ETH spot price (Binance, cached) ──────────────────────────────────────

// binanceTickerResp is the shape of /api/v3/ticker/price.
type binanceTickerResp struct {
	Symbol string `json:"symbol"`
	Price  string `json:"price"`
}

// ethSpotUSD returns the cached ETH/USD spot, refreshing from Binance when
// stale. Returns 0 (not an error) when Binance is unavailable — callers leave
// amount_usd=0 in that case so the UI renders "—".
func (e *EtherscanSource) ethSpotUSD(ctx context.Context) float64 {
	e.spotMu.Lock()
	if time.Since(e.spotAt) < spotCacheTTL && e.spotPrice > 0 {
		p := e.spotPrice
		e.spotMu.Unlock()
		return p
	}
	e.spotMu.Unlock()

	price := fetchBinanceSpot(ctx, e.client(), "ETHUSDT")
	if price > 0 {
		e.spotMu.Lock()
		e.spotPrice = price
		e.spotAt = time.Now()
		e.spotMu.Unlock()
	}
	return price
}

// fetchBinanceSpot fetches one symbol's last price from Binance. Returns 0 on
// any error (never panics) so the caller can treat "no price" as amount_usd=0.
// Shared by EtherscanSource (ETH) and MempoolSource (BTC).
func fetchBinanceSpot(ctx context.Context, client *http.Client, symbol string) float64 {
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Second}
	}
	reqURL := "https://api.binance.com/api/v3/ticker/price?symbol=" + url.QueryEscape(symbol)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return 0
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
	if err != nil {
		return 0
	}
	var t binanceTickerResp
	if uerr := json.Unmarshal(body, &t); uerr != nil {
		return 0
	}
	p, perr := strconv.ParseFloat(t.Price, 64)
	if perr != nil || p <= 0 || math.IsNaN(p) || math.IsInf(p, 0) {
		return 0
	}
	return p
}

// ── small parse helpers ────────────────────────────────────────────────────

// attribute decides the FlowDirection for a transfer relative to the watched
// exchange address. `watchedExch` is the label of the address we polled; we
// re-check both sides against the full wallet map so a transfer between two
// exchanges still attributes to the side that matches first (to=inflow wins).
func attribute(from, to, watchedExch string, wallets map[string]string) (FlowDirection, string) {
	toL := strings.ToLower(to)
	fromL := strings.ToLower(from)
	if ex, ok := wallets[toL]; ok {
		return FlowInflow, ex // deposit TO an exchange
	}
	if ex, ok := wallets[fromL]; ok {
		return FlowOutflow, ex // withdrawal FROM an exchange
	}
	// Neither side is a known wallet. Shouldn't happen since we polled a
	// watched address, but Etherscan can include internal/related txns — fall
	// back to the watched label only if it's non-empty, else neutral.
	_ = watchedExch
	return FlowNeutral, ""
}

// weiToEth converts a wei string to ETH (÷1e18). Returns 0 on parse failure.
func weiToEth(wei string) float64 {
	return scaleByDecimals(wei, 18)
}

// scaleByDecimals parses a base-10 integer-valued string and divides by
// 10^decimals, returning a float64. Big values (USDT has 6 decimals on huge
// transfers) are parsed via big-free float math: we parse the integer string
// into a float64 directly. For the magnitudes whale transfers reach (≤ ~1e12
// native units) float64 has ample precision for a USD estimate. Returns 0 on
// parse failure.
func scaleByDecimals(raw string, decimals int) float64 {
	raw = strings.TrimSpace(raw)
	if raw == "" || raw == "0" {
		return 0
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0
	}
	if decimals <= 0 {
		return v
	}
	return v / math.Pow10(decimals)
}

// parseUint parses a base-10 uint string; returns 0 on failure.
func parseUint(s string) uint64 {
	v, err := strconv.ParseUint(strings.TrimSpace(s), 10, 64)
	if err != nil {
		return 0
	}
	return v
}

// parseInt parses a base-10 int string; returns 0 on failure.
func parseInt(s string) int {
	v, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return 0
	}
	return v
}

// unixToTime converts a unix-seconds string to time.Time (UTC). Returns the
// zero time on parse failure — callers store it as-is; the DB column is NOT
// NULL so a zero time round-trips as the epoch (acceptable for a malformed
// upstream row, which is vanishingly rare).
func unixToTime(s string) time.Time {
	sec, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
	if err != nil {
		return time.Time{}
	}
	return time.Unix(sec, 0).UTC()
}

// sortTransfersDesc sorts transfers newest-first in place (by Timestamp DESC,
// ties broken by TxHash for determinism). Shared by EtherscanSource and
// MempoolSource.
func sortTransfersDesc(ts []WhaleTransfer) {
	sort.Slice(ts, func(i, j int) bool {
		if !ts[i].Timestamp.Equal(ts[j].Timestamp) {
			return ts[i].Timestamp.After(ts[j].Timestamp)
		}
		return ts[i].TxHash < ts[j].TxHash
	})
}
