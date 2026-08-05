package email

// resend_test.go — offline coverage for the email builders + dev-mode send.
// SendEmail with an empty key never touches the network, so it is testable
// here; the real Resend POST path is not exercised (no key, no mock server —
// honesty over a fake integration test).

import (
	"context"
	"strings"
	"testing"
)

func TestVerifyEmailHTML(t *testing.T) {
	const base = "http://localhost:5173/alphavision"
	const tok = "RAWTOKEN_abc-123"
	subject, html := VerifyEmailHTML(base, tok)

	if subject != "Verify your AlphaVizor email" {
		t.Errorf("verify subject = %q, want %q", subject, "Verify your AlphaVizor email")
	}
	wantLink := base + "/verify-email?token=" + tok
	if !strings.Contains(html, wantLink) {
		t.Errorf("verify html missing link %q\n--- html ---\n%s", wantLink, html)
	}
	if !strings.Contains(html, "expires in 24 hours") {
		t.Errorf("verify html missing 24-hour expiry copy\n--- html ---\n%s", html)
	}
}

func TestResetPasswordHTML(t *testing.T) {
	const base = "http://localhost:5173/alphavision"
	const tok = "RESET_xyz-789"
	subject, html := ResetPasswordHTML(base, tok)

	if subject != "Reset your AlphaVizor password" {
		t.Errorf("reset subject = %q, want %q", subject, "Reset your AlphaVizor password")
	}
	wantLink := base + "/reset-password?token=" + tok
	if !strings.Contains(html, wantLink) {
		t.Errorf("reset html missing link %q\n--- html ---\n%s", wantLink, html)
	}
	if !strings.Contains(html, "expires in 1 hour") {
		t.Errorf("reset html missing 1-hour expiry copy\n--- html ---\n%s", html)
	}
}

func TestVerifyEmailHTML_TokenIsRawNotHashed(t *testing.T) {
	// The link must carry the RAW token verbatim — if it were hashed the
	// recipient could never verify. Pin that the exact token string appears.
	_, html := VerifyEmailHTML("https://app.example/alphavision", "verbatim-token")
	if !strings.Contains(html, "token=verbatim-token") {
		t.Errorf("expected raw token in link, html:\n%s", html)
	}
}

func TestSendEmail_DevModeReturnsNil(t *testing.T) {
	// Empty API key => dev mode: no network, returns nil. (If this tried to
	// reach the network the test would either hang or fail — it must not.)
	err := SendEmail(context.Background(), "", "no-reply@alphavizor.com",
		"user@example.com", "Subject", "<p>body with link</p>")
	if err != nil {
		t.Fatalf("SendEmail in dev mode returned error: %v", err)
	}
}
