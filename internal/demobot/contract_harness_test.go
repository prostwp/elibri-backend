package demobot

// contract_harness_test.go — every GET address of the API answered end to end
// by the real handler chain over deterministic stub upstreams (Binance klines
// and premiumIndex, Yahoo chart, the backend), plus cards from the pure
// builders' own fixtures. Two consumers:
//
//   - openapi_contract_test.go validates every body against docs/openapi;
//   - DEMOBOT_CONTRACT_DUMP=<dir> writes the bodies there, one file per
//     address, for the number audit (.ops/…/audit_numbers.py <dir>).

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
	"time"
)

// contractBody is one served body with the address that served it.
type contractBody struct {
	path   string // request URI, or "builder:<name>" for a pure-builder card
	status int
	body   []byte
}

// stubBinanceTrend serves closed bars for any symbol and interval: a drift
// plus a slow wave, so pivots, a confirmed trend and pullbacks all exist. BTC
// drifts up, everything else down — both confirmed directions are served.
// Volume varies, so the volume context and the S/R volume flags read.
func stubBinanceTrend(t *testing.T) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		sec := intervalSeconds[q.Get("interval")]
		if sec == 0 {
			sec = 14400
		}
		base, drift := 60000.0, 40.0
		if !strings.HasPrefix(q.Get("symbol"), "BTC") {
			base, drift = 3000.0, -1.5
		}
		const bars = 600
		start := (time.Now().Unix()/sec - bars) * sec
		rows := make([][]any, bars+1) // the last row is the forming bar
		for i := range rows {
			p := base + drift*float64(i) + math.Abs(drift)*12*math.Sin(float64(i)/4)
			rows[i] = []any{
				float64(start+int64(i)*sec) * 1000,
				fmt.Sprintf("%f", p-math.Abs(drift)/2), fmt.Sprintf("%f", p+math.Abs(drift)*3),
				fmt.Sprintf("%f", p-math.Abs(drift)*3), fmt.Sprintf("%f", p),
				fmt.Sprintf("%f", 100+40*math.Sin(float64(i)/3)),
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(rows)
	}))
	t.Cleanup(srv.Close)
	orig := binanceKlinesBase
	binanceKlinesBase = srv.URL
	t.Cleanup(func() { binanceKlinesBase = orig })
}

// stubPremiumIndex serves a last funding rate and a mark price per symbol.
func stubPremiumIndex(t *testing.T, rates map[string]string) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sym := r.URL.Query().Get("symbol")
		rate, ok := rates[sym]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"symbol":%q,"markPrice":"118000.0","lastFundingRate":%q}`, sym, rate)
	}))
	t.Cleanup(srv.Close)
	orig := premiumIndexURL
	premiumIndexURL = srv.URL + "/fapi/v1/premiumIndex?symbol="
	t.Cleanup(func() { premiumIndexURL = orig })
}

// contractAddresses is every GET address the API serves, each agent with its
// parameter forms.
func contractAddresses() []string {
	paths := []string{"/", "/agents", "/showcase", "/showcase/example",
		"/agents/digest", "/agents/top", "/agents/fx", "/agents/macro", "/agents/macro?asset=btc",
		"/agents/macro?asset=gold", "/agents/whale", "/agents/funding", "/agents/momentum",
		"/agents/momentum?assets=btc,eurusd,gold&tf=1d", "/agents/news", "/agents/gold",
		"/agents/risk?balance=10000&risk=1&entry=64000&stop=62500",
		"/agents/trend/chart?asset=btc", "/agents/trend/chart?asset=eurusd",
	}
	for _, k := range assetKeys() {
		for _, a := range []string{keyMomentum, keyTrend, keySR, keyVol} {
			paths = append(paths, "/agents/"+a+"?asset="+k)
		}
	}
	return paths
}

// contractServe answers every address end to end over the stubs.
func contractServe(t *testing.T) []contractBody {
	t.Helper()
	stubExternalBases(t)
	stubBinanceTrend(t)
	stubPremiumIndex(t, map[string]string{
		"BTCUSDT": "0.00010000", "ETHUSDT": "0.00007219", "SOLUSDT": "0.00009364",
		"BNBUSDT": "0.00008777", "XRPUSDT": "0.00000055",
	})
	stubYahooWave(t, time.Now().UTC().Truncate(time.Hour).Add(-time.Hour), 600)
	ag := newStubBackend(t, map[string]string{
		"/api/v1/macro":                macroLiveFixture,
		"/api/v1/whale-flow":           whaleFixture(true),
		"/api/v1/narratives":           narrativesFixture,
		"/api/v1/funding/liquidations": liqFixture(time.Now().UTC().Format(time.RFC3339)),
		"/api/v1/market/momentum":      `{"baseline":"BTC","items":{"ETH":{"rs_7d":-1.83,"rs_30d":-9.41}}}`,
	})
	_, srv := newTestAPI(t, ag, true)
	var out []contractBody
	for _, p := range contractAddresses() {
		st, _, body := httpGet(t, srv.URL+p)
		out = append(out, contractBody{path: p, status: st, body: body})
	}
	return out
}

// contractBuilderCards are the pure builders' own fixtures: every trend state,
// every gold state, every whale path, the FX table fixtures — the states the
// stubs above cannot reach.
func contractBuilderCards(t *testing.T) []contractBody {
	t.Helper()
	var out []contractBody
	add := func(name string, c Card) {
		b, err := encodeJSON(cardEnvelope(c))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		out = append(out, contractBody{path: "builder:" + name, status: http.StatusOK, body: b})
	}
	for _, k := range sortedMapKeys(trendFixtures) {
		add("trend/"+k, renderFixture(trendFixtures[k]))
	}
	for _, st := range []string{"up", "down", "grey", "flat", "conflict"} {
		add("gold/"+st, goldCardFrom(goldFixture(st)))
	}
	cards := whaleEveryCard(t)
	for _, k := range sortedMapKeys(cards) {
		add("whale/"+k, cards[k])
	}
	now := time.Date(2026, 9, 15, 12, 0, 0, 0, time.UTC)
	fx := fxTextFixtures()
	for _, k := range sortedMapKeys(fx) {
		add("fx/"+k, fxOverviewCard(fx[k], now))
	}
	return out
}

func sortedMapKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

var contractFileRe = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

// TestContractDump writes every body for the number audit. Skipped unless
// DEMOBOT_CONTRACT_DUMP names a directory.
func TestContractDump(t *testing.T) {
	dir := os.Getenv("DEMOBOT_CONTRACT_DUMP")
	if dir == "" {
		t.Skip("set DEMOBOT_CONTRACT_DUMP=<dir> to write the bodies")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	all := append(contractServe(t), contractBuilderCards(t)...)
	for _, b := range all {
		name := strings.Trim(contractFileRe.ReplaceAllString(strings.TrimPrefix(b.path, "/"), "_"), "_")
		if name == "" {
			name = "root"
		}
		if err := os.WriteFile(filepath.Join(dir, name+".json"), b.body, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("%d bodies written to %s", len(all), dir)
}
