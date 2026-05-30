package whale

// registry.go — curated map of exchange hot-wallet addresses (ETH).
//
// Source: .ops/discovery-whale_flow.md (research collected ~19 ETH-labeled
// addresses for Binance / Coinbase / Kraken / OKX). All addresses are stored
// LOWERCASE — the Etherscan source lowercases inbound addresses before lookup,
// so case can never cause a miss.
//
// Refresh method: exchanges rotate hot-wallets periodically. To refresh, pull
// the Etherscan label cloud at
//   https://etherscan.io/accounts/label/<exchange>
// (e.g. /accounts/label/binance) and add any new "Hot Wallet N" addresses
// below. There is no free API for the label cloud; this is a manual curation
// step. The free netflow is therefore an ESTIMATE over the addresses we know —
// the snapshot's Partial=true badge is the honest disclosure of that.
//
// BTC is intentionally NOT registered: BTC's UTXO model + exchange address
// churn make labeled-wallet attribution unreliable, so BTC is a live feed only
// (direction=neutral) in the MVP (discovery §3).

// ExchangeBinance / etc. are the canonical exchange labels emitted in
// WhaleTransfer.Exchange and used as keys in Snapshot.ExchangeBreakdown.
const (
	ExchangeBinance  = "Binance"
	ExchangeCoinbase = "Coinbase"
	ExchangeKraken   = "Kraken"
	ExchangeOKX      = "OKX"
)

// ethHotWallets maps lowercase ETH address → exchange label. Keep all keys
// lowercase. The verified set below is the curated MVP list from discovery.
var ethHotWallets = map[string]string{
	// ── Binance ──────────────────────────────────────────────────────────
	"0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be": ExchangeBinance, // Binance 1 (verified)
	"0xf977814e90da44bfa03b6295a0616a897441acec": ExchangeBinance, // Binance 8 / cold (verified)
	"0x28c6c06298d514db089934071355e5743bf21d60": ExchangeBinance, // Binance 14 (verified)
	"0x21a31ee1afc51d94c2efccaa2092ad1028285549": ExchangeBinance, // Binance 15
	"0xdfd5293d8e347dfe59e90efd55b2956a1343963d": ExchangeBinance, // Binance 16
	"0x56eddb7aa87536c09ccc2793473599fd21a8b17f": ExchangeBinance, // Binance 17
	"0x9696f59e4d72e237be84ffd425dcad154bf96976": ExchangeBinance, // Binance 18

	// ── Coinbase ─────────────────────────────────────────────────────────
	"0x71660c4005ba85c37ccec55d0c4493e66fe775d3": ExchangeCoinbase, // Coinbase 1 (verified)
	"0x503828976d22510aad0201ac7ec88293211d23da": ExchangeCoinbase, // Coinbase 2 (verified)
	"0xddfabcdc4d8ffc6d5beaf154f18b778f892a0740": ExchangeCoinbase, // Coinbase 3
	"0x3cd751e6b0078be393132286c442345e5dc49699": ExchangeCoinbase, // Coinbase 4
	"0xb5d85cbf7cb3ee0d56b3bb207d5fc4b82f43f511": ExchangeCoinbase, // Coinbase 5
	"0xeb2629a2734e272bcc07bda959863f316f4bd4cf": ExchangeCoinbase, // Coinbase 6

	// ── Kraken ───────────────────────────────────────────────────────────
	"0xe9f7ecae3a53d2a67105292894676b00d1fab785": ExchangeKraken, // Kraken (verified)
	"0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0": ExchangeKraken, // Kraken 4 (verified)
	"0xfa52274dd61e1643d2205169732f29114bc240b3": ExchangeKraken, // Kraken 5
	"0x53d284357ec70ce289d6d64134dfac8e511c8a3d": ExchangeKraken, // Kraken 6

	// ── OKX ──────────────────────────────────────────────────────────────
	"0x4b4e14a3773ee558b6597070797fd51eb48606e5": ExchangeOKX, // OKX (verified)
	"0x6cc5f688a315f3dc28a7781717a9a798a59fda7b": ExchangeOKX, // OKX 2
	"0x236f9f97e0e62388479bf9e5ba4889e46b0273c3": ExchangeOKX, // OKX 3
}

// staticRegistry is the in-memory ExchangeRegistry backed by the curated
// ethHotWallets map. Zero-value ready: &staticRegistry{} works without any
// constructor. Stateless and concurrency-safe (read-only map).
type staticRegistry struct{}

// NewStaticRegistry returns the curated-map registry. Provided as a
// constructor for symmetry with NewStore; callers may also use
// &staticRegistry{} directly.
func NewStaticRegistry() ExchangeRegistry { return &staticRegistry{} }

// HotWallets returns the lowercase-address → exchange map for `chain`. ETH
// returns the curated set; every other chain (including BTC) returns an empty
// map — BTC attribution is out of MVP scope, so BTC transfers stay neutral.
//
// The returned map is the shared package-level map for ETH — callers MUST NOT
// mutate it (the interface docstring states this). Returning the shared map
// (rather than a copy) avoids an allocation on every Etherscan poll iteration.
func (r *staticRegistry) HotWallets(chain Chain) map[string]string {
	if chain == ChainETH {
		return ethHotWallets
	}
	return map[string]string{}
}
