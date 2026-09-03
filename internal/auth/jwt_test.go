package auth

import "testing"

// The demobot proxy prefix must NOT be public. The engine's own rate limiter
// is switched off (its container consumer polices itself), so an anonymous
// route there would be unmetered access to sweeps that fan out to Binance,
// Yahoo and the AI provider.
func TestDemobotPrefixRequiresAuth(t *testing.T) {
	for _, p := range []string{
		"/api/v1/demobot/showcase",
		"/api/v1/demobot/agents/gold",
		"/api/v1/demobot/agents",
	} {
		if isPublicPath(p) {
			t.Errorf("%s is reachable without a token — nothing rate-limits it", p)
		}
	}
	// Control: the catalog IS public. Without this, a change that made every
	// path private would pass the assertions above while breaking the site.
	if !isPublicPath("/api/v1/agents/narrative_radar/output") {
		t.Error("control failed: the public catalog route stopped being public")
	}
}
