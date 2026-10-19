package demobot

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

// The push hook's events match webhooks.agentEvent: every event a sweep
// POSTs, over the full v1 address set, validates as HookEvent, with data a
// Card (200) or a Degraded body (503).
func TestOpenAPIContractHookEvents(t *testing.T) {
	stubExternalBases(t)
	stubBinanceKlinesWave(t, binanceFetchLimit)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	sink := newHookSink(t, http.StatusCreated)
	th := newTestHook(t, ag, sink.srv.URL, nil)
	th.sweep(context.Background())

	sink.mu.Lock()
	raws := append([]string{}, sink.raws...)
	sink.mu.Unlock()
	if len(raws) != len(hookTargets()) {
		t.Fatalf("%d events for %d addresses", len(raws), len(hookTargets()))
	}
	agents := map[string]bool{}
	schemas := openapiSchemas(t, "HookEvent", "Card", "Degraded")
	props := cardSchemaProps(t)
	kinds := map[string]bool{}
	for _, raw := range raws {
		var ev hookEvent
		if err := json.Unmarshal([]byte(raw), &ev); err != nil {
			t.Fatal(err)
		}
		agents[ev.Agent] = true
		contractValidate(t, schemas, props, ev.EventID, "HookEvent", []byte(raw))
		var probe struct {
			Error *string `json:"error"`
		}
		_ = json.Unmarshal(ev.Data, &probe)
		if probe.Error != nil {
			kinds["Degraded"] = true
			contractValidate(t, schemas, props, ev.EventID+" data", "Degraded", ev.Data)
		} else {
			kinds["Card"] = true
			contractValidate(t, schemas, props, ev.EventID+" data", "Card", ev.Data)
		}
	}
	for _, tg := range hookTargets() {
		if !agents[tg.Agent] {
			t.Errorf("no event from %s", tg.Agent)
		}
	}
	for _, k := range []string{"Card", "Degraded"} {
		if !kinds[k] {
			t.Errorf("no event carried a %s body", k)
		}
	}

	// state_changed set and params present: the {from, to} branch of the
	// schema and a params segment in event_id.
	asOf := "2026-10-19T08:00:00Z"
	ev := hookEvent{
		EventID: "momentum:BTC:tf=1d:" + asOf + ":a1b2c3d4", Agent: keyMomentum, Asset: "BTC",
		Params:   map[string]string{"tf": "1d"},
		DataAsOf: &asOf, SentAt: "2026-10-19T08:00:09Z",
		StateChanged: &hookStateChange{From: "neutral", To: hookUnavailable},
		Data:         json.RawMessage(`{"error":"x","ok":false,"reason":"source_offline"}`),
	}
	b, _ := json.Marshal(ev)
	contractValidate(t, schemas, props, "state_changed", "HookEvent", b)
	ev.StateChanged = &hookStateChange{From: "neutral", To: "up"}
	b, _ = json.Marshal(ev)
	v, _ := jsonschema.UnmarshalJSON(bytes.NewReader(b))
	if schemas["HookEvent"].Validate(v) == nil {
		t.Error("a state outside bullish|bearish|neutral|unavailable must fail")
	}

	doc := openapiDoc(t)
	wh, _ := doc["webhooks"].(map[string]any)
	if wh["agentEvent"] == nil {
		t.Error("webhooks.agentEvent is missing")
	}
}
