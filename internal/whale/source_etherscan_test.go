package whale

import (
	"strings"
	"testing"
)

// TestScrubKey guards the security fix: a transport error's *url.Error embeds
// the full request URL including apikey=..., so scrubKey must remove the key
// before the error string can reach the worker's logs.
func TestScrubKey(t *testing.T) {
	const key = "SECRETKEY123"
	e := &EtherscanSource{APIKey: key}

	in := `http: Get "https://api.etherscan.io/v2/api?module=account&apikey=` + key + `&address=0xabc": dial tcp: i/o timeout`
	got := e.scrubKey(in)
	if strings.Contains(got, key) {
		t.Fatalf("scrubKey leaked the api key: %q", got)
	}
	if !strings.Contains(got, "***") {
		t.Fatalf("scrubKey should replace the key with ***: %q", got)
	}

	// Empty key → no-op (anonymous calls have nothing to scrub).
	empty := &EtherscanSource{APIKey: ""}
	if out := empty.scrubKey("no key here"); out != "no key here" {
		t.Fatalf("scrubKey with empty key should be a no-op, got %q", out)
	}
}
