package auth

// tokens_test.go — offline coverage for the hashed action-token primitives.
// No DB needed: GenerateAuthToken / hashToken are pure crypto helpers.

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"testing"
)

func TestGenerateAuthToken_Roundtrip(t *testing.T) {
	raw, hash, err := GenerateAuthToken()
	if err != nil {
		t.Fatalf("GenerateAuthToken returned error: %v", err)
	}

	// raw must be valid RawURLEncoding and decode to exactly 32 bytes.
	decoded, err := base64.RawURLEncoding.DecodeString(raw)
	if err != nil {
		t.Fatalf("raw token is not valid RawURLEncoding: %v", err)
	}
	if len(decoded) != 32 {
		t.Fatalf("decoded raw token = %d bytes, want 32", len(decoded))
	}

	// hash must be hex(sha256(raw)) and 64 chars long.
	want := sha256.Sum256([]byte(raw))
	wantHex := hex.EncodeToString(want[:])
	if hash != wantHex {
		t.Fatalf("hash mismatch:\n got  %s\n want %s", hash, wantHex)
	}
	if len(hash) != 64 {
		t.Fatalf("hash length = %d, want 64", len(hash))
	}
}

func TestHashToken_MatchesSHA256(t *testing.T) {
	// hashToken is what both issue + consume rely on; pin it to sha256-hex.
	in := "some-known-raw-token"
	sum := sha256.Sum256([]byte(in))
	want := hex.EncodeToString(sum[:])
	if got := hashToken(in); got != want {
		t.Fatalf("hashToken(%q) = %s, want %s", in, got, want)
	}
}

func TestGenerateAuthToken_Uniqueness(t *testing.T) {
	const n = 200
	rawSeen := make(map[string]struct{}, n)
	hashSeen := make(map[string]struct{}, n)
	for i := 0; i < n; i++ {
		raw, hash, err := GenerateAuthToken()
		if err != nil {
			t.Fatalf("iteration %d: %v", i, err)
		}
		if _, dup := rawSeen[raw]; dup {
			t.Fatalf("duplicate raw token at iteration %d: %s", i, raw)
		}
		if _, dup := hashSeen[hash]; dup {
			t.Fatalf("duplicate hash at iteration %d: %s", i, hash)
		}
		rawSeen[raw] = struct{}{}
		hashSeen[hash] = struct{}{}
	}
}

func TestTokenTTLConstants(t *testing.T) {
	// Guard the contract values (verify 24h, reset 1h) so a refactor can't
	// silently change link lifetimes.
	if EmailVerifyTTL.Hours() != 24 {
		t.Errorf("EmailVerifyTTL = %v, want 24h", EmailVerifyTTL)
	}
	if PasswordResetTTL.Hours() != 1 {
		t.Errorf("PasswordResetTTL = %v, want 1h", PasswordResetTTL)
	}
}
