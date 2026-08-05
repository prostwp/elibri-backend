// Package email sends transactional mail via Resend (https://resend.com).
//
// Sub-chat C uses it for two flows: email verification and password reset.
// Design choices baked in here:
//
//   - Secrets never live in code. The API key is passed in by the caller
//     (threaded from config.ResendAPIKey, which reads RESEND_API_KEY).
//   - Dev mode: when the API key is empty, SendEmail logs the message (and
//     thus the action link) to stdout and returns nil. This lets the whole
//     verify/reset flow be exercised end-to-end locally with no Resend
//     account — the developer copies the link from the server log.
//   - Failure semantics are the CALLER's call. SendEmail returns an error;
//     register/forgot-password treat a send failure as non-fatal (WARN log,
//     still 200) so a flaky mail provider never blocks signup or leaks
//     account existence.
package email

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

// resendEndpoint is the Resend transactional-send endpoint.
const resendEndpoint = "https://api.resend.com/emails"

// sendTimeout caps how long we wait on the Resend API before giving up. The
// caller's context still applies; this is an upper bound so a hung provider
// can't pin a request goroutine for the full server timeout.
const sendTimeout = 10 * time.Second

// sendRequest is the JSON body Resend expects: a single recipient is fine
// for our flows (verify/reset are always 1:1 with a user).
type sendRequest struct {
	From    string   `json:"from"`
	To      []string `json:"to"`
	Subject string   `json:"subject"`
	HTML    string   `json:"html"`
}

// SendEmail posts a single transactional email to Resend.
//
// When apiKey == "" we are in dev mode: the email (including its action
// link, which lives in htmlBody) is logged to stdout and nil is returned —
// no network call is made. This is how the verify/reset flow is tested
// without a Resend key.
//
// On a real send, any non-2xx response or transport error is returned to the
// caller, who decides whether it is fatal (it is not, for register/forgot).
func SendEmail(ctx context.Context, apiKey, from, to, subject, htmlBody string) error {
	if apiKey == "" {
		// Dev mode — surface the full HTML so the developer can copy the
		// action link out of the server log. Kept on one logical line so it
		// is easy to grep ("[email][dev]").
		log.Printf("[email][dev] would send to=%s subject=%q (no RESEND_API_KEY set)\n%s", to, subject, htmlBody)
		return nil
	}

	payload, err := json.Marshal(sendRequest{
		From:    from,
		To:      []string{to},
		Subject: subject,
		HTML:    htmlBody,
	})
	if err != nil {
		return fmt.Errorf("email: marshal request: %w", err)
	}

	reqCtx, cancel := context.WithTimeout(ctx, sendTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, resendEndpoint, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("email: build request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("email: send: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// Read a small slice of the body for diagnostics — Resend returns a
		// JSON error object. Cap the read so a misbehaving endpoint can't
		// stream us megabytes into an error string.
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return fmt.Errorf("email: resend returned %d: %s", resp.StatusCode, string(body))
	}
	return nil
}

// VerifyEmailHTML builds the subject + HTML body for the email-verification
// message. The link points at the frontend route {appBaseURL}/verify-email
// carrying the RAW token (the DB only ever stores its sha256). appBaseURL
// already includes the Vite base path (local: ".../elibri-fx").
func VerifyEmailHTML(appBaseURL, rawToken string) (subject, html string) {
	link := appBaseURL + "/verify-email?token=" + rawToken
	subject = "Verify your AlphaVizor email"
	html = `<div style="font-family:system-ui,-apple-system,sans-serif;max-width:480px;margin:0 auto;color:#111">
  <h2 style="margin:0 0 16px">Verify your email</h2>
  <p>Welcome to AlphaVizor. Please confirm your email address to finish setting up your account.</p>
  <p style="margin:24px 0">
    <a href="` + link + `" style="background:#4f46e5;color:#fff;padding:12px 20px;border-radius:8px;text-decoration:none;display:inline-block">Verify email</a>
  </p>
  <p style="color:#555;font-size:13px">Or paste this link into your browser:<br><a href="` + link + `">` + link + `</a></p>
  <p style="color:#555;font-size:13px">Link expires in 24 hours.</p>
</div>`
	return subject, html
}

// ResetPasswordHTML builds the subject + HTML body for the password-reset
// message. Link points at {appBaseURL}/reset-password carrying the RAW token.
func ResetPasswordHTML(appBaseURL, rawToken string) (subject, html string) {
	link := appBaseURL + "/reset-password?token=" + rawToken
	subject = "Reset your AlphaVizor password"
	html = `<div style="font-family:system-ui,-apple-system,sans-serif;max-width:480px;margin:0 auto;color:#111">
  <h2 style="margin:0 0 16px">Reset your password</h2>
  <p>We received a request to reset your AlphaVizor password. Click the button below to choose a new one.</p>
  <p style="margin:24px 0">
    <a href="` + link + `" style="background:#4f46e5;color:#fff;padding:12px 20px;border-radius:8px;text-decoration:none;display:inline-block">Reset password</a>
  </p>
  <p style="color:#555;font-size:13px">Or paste this link into your browser:<br><a href="` + link + `">` + link + `</a></p>
  <p style="color:#555;font-size:13px">This link expires in 1 hour. If this wasn't you, ignore this email.</p>
</div>`
	return subject, html
}
