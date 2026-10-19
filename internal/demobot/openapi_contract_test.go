package demobot

// openapi_contract_test.go — the OpenAPI document (openapi.yaml) is the
// contract the site parses. Every body this package serves is validated
// against it:
//
//   - every GET address end to end over stub upstreams, and the pure
//     builders' fixture cards (contract_harness_test.go);
//   - real production answers of 2026-10-19 (testdata/prod_2026-10-19).
//
// Machine objects carry additionalProperties: false in the document, so a
// field a builder adds without describing it fails here. The envelope itself
// stays open in the document (it may grow); this test still requires every
// top-level key to be a declared Card property.
//
// Validator: github.com/santhosh-tekuri/jsonschema/v6 — pure Go, no cgo,
// full JSON Schema 2020-12 (the dialect of OpenAPI 3.1, type arrays with
// "null" included), and it compiles a schema by JSON pointer inside a larger
// document, so components resolve their $refs in place.

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/santhosh-tekuri/jsonschema/v6"
	"gopkg.in/yaml.v3"
)

const openapiURL = "mem://demobot/openapi.json"

// openapiDoc is the document as a generic JSON value.
func openapiDoc(t *testing.T) map[string]any {
	t.Helper()
	if openapiJSONErr != nil {
		t.Fatal(openapiJSONErr)
	}
	var doc map[string]any
	if err := json.Unmarshal(openapiJSON, &doc); err != nil {
		t.Fatal(err)
	}
	return doc
}

// openapiSchemas compiles the named component schemas.
func openapiSchemas(t *testing.T, names ...string) map[string]*jsonschema.Schema {
	t.Helper()
	return openapiSchemasFrom(t, openapiJSON, names...)
}

// openapiSchemasBefore compiles the schemas as they stood before `added`:
// fields marked x-added: <added> are no longer required (they stay
// described, so their types are still checked when present). For answers
// recorded before those fields existed.
func openapiSchemasBefore(t *testing.T, added string, names ...string) map[string]*jsonschema.Schema {
	t.Helper()
	doc := openapiDoc(t)
	for _, s := range doc["components"].(map[string]any)["schemas"].(map[string]any) {
		sm := s.(map[string]any)
		props, _ := sm["properties"].(map[string]any)
		req, _ := sm["required"].([]any)
		kept := []any{}
		for _, r := range req {
			if p, ok := props[r.(string)].(map[string]any); ok && p["x-added"] == added {
				continue
			}
			kept = append(kept, r)
		}
		if req != nil {
			sm["required"] = kept
		}
	}
	b, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	return openapiSchemasFrom(t, b, names...)
}

func openapiSchemasFrom(t *testing.T, raw []byte, names ...string) map[string]*jsonschema.Schema {
	t.Helper()
	doc, err := jsonschema.UnmarshalJSON(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	c := jsonschema.NewCompiler()
	c.DefaultDraft(jsonschema.Draft2020)
	if err := c.AddResource(openapiURL, doc); err != nil {
		t.Fatal(err)
	}
	out := map[string]*jsonschema.Schema{}
	for _, n := range names {
		s, err := c.Compile(openapiURL + "#/components/schemas/" + n)
		if err != nil {
			t.Fatalf("compile %s: %v", n, err)
		}
		out[n] = s
	}
	return out
}

// contractSchemaFor names the schema a body must match.
func contractSchemaFor(path string, status int) string {
	p := path
	if i := strings.IndexByte(p, '?'); i >= 0 {
		p = p[:i]
	}
	switch {
	case status == http.StatusServiceUnavailable:
		return "Degraded"
	case status >= 400:
		return "Error"
	case p == "/":
		return "Root"
	case p == "/agents":
		return "AgentList"
	case p == "/agents/trend/chart", strings.Contains(p, "trend_chart"):
		return "TrendChart"
	case p == "/showcase/example", strings.HasPrefix(p, "showcase_example"):
		return "ShowcaseExample"
	case p == "/showcase", p == "showcase":
		return "Showcase"
	}
	return "Card"
}

func contractValidate(t *testing.T, schemas map[string]*jsonschema.Schema, cardProps map[string]any, name, schema string, body []byte) {
	t.Helper()
	v, err := jsonschema.UnmarshalJSON(bytes.NewReader(body))
	if err != nil {
		t.Errorf("%s: not JSON: %v", name, err)
		return
	}
	if err := schemas[schema].Validate(v); err != nil {
		t.Errorf("%s does not match %s:\n%v", name, schema, err)
	}
	if schema != "Card" {
		return
	}
	for k := range v.(map[string]any) {
		if _, ok := cardProps[k]; !ok {
			t.Errorf("%s: top-level field %q is not described in the Card schema", name, k)
		}
	}
}

var contractSchemaNames = []string{"Card", "Degraded", "Error", "Root", "AgentList", "TrendChart", "Showcase", "ShowcaseExample"}

func cardSchemaProps(t *testing.T) map[string]any {
	t.Helper()
	doc := openapiDoc(t)
	return doc["components"].(map[string]any)["schemas"].(map[string]any)["Card"].(map[string]any)["properties"].(map[string]any)
}

func TestOpenAPIContractServedBodies(t *testing.T) {
	schemas := openapiSchemas(t, contractSchemaNames...)
	props := cardSchemaProps(t)
	bodies := append(contractServe(t), contractBuilderCards(t)...)
	seen := map[string]bool{}
	for _, b := range bodies {
		sch := contractSchemaFor(b.path, b.status)
		seen[sch] = true
		contractValidate(t, schemas, props, b.path, sch, b.body)
	}
	for _, n := range []string{"Card", "Root", "AgentList", "TrendChart", "Showcase", "ShowcaseExample"} {
		if !seen[n] {
			t.Errorf("no served body exercised %s", n)
		}
	}
}

// Degraded and error bodies: a dead world answers 503 and bad requests 400.
func TestOpenAPIContractDegradedAndErrors(t *testing.T) {
	stubExternalBases(t)
	ag := NewAgents(NewBackendClient("http://127.0.0.1:1"))
	_, srv := newTestAPI(t, ag, true)
	schemas := openapiSchemas(t, contractSchemaNames...)
	props := cardSchemaProps(t)
	statuses := map[int]bool{}
	for _, p := range append(contractAddresses(), "/agents/nope", "/agents/trend?tf=1d", "/agents/risk", "/nope",
		"/agents/momentum?asset=btc&asset=eth") {
		st, _, body := httpGet(t, srv.URL+p)
		statuses[st] = true
		contractValidate(t, schemas, props, p, contractSchemaFor(p, st), body)
	}
	for _, st := range []int{http.StatusOK, http.StatusBadRequest, http.StatusNotFound, http.StatusServiceUnavailable} {
		if !statuses[st] {
			t.Errorf("no %d body exercised", st)
		}
	}
}

// Production answers recorded on 2026-10-19, before that day's numeric
// fields: every other field must match the document as is.
func TestOpenAPIContractProdFixtures(t *testing.T) {
	schemas := openapiSchemasBefore(t, "2026-10-19", contractSchemaNames...)
	props := cardSchemaProps(t)
	files, err := filepath.Glob(filepath.Join("testdata", "prod_2026-10-19", "*.json"))
	if err != nil || len(files) < 13 {
		t.Fatalf("prod fixtures: %d files, %v", len(files), err)
	}
	for _, f := range files {
		body, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		name := filepath.Base(f)
		contractValidate(t, schemas, props, name, contractSchemaFor(strings.TrimSuffix(name, ".json"), http.StatusOK), body)
	}
}

// The strictness is real: an undescribed field inside a machine object, or
// a wrong type, fails validation.
func TestOpenAPIContractRejectsUndescribedFields(t *testing.T) {
	schemas := openapiSchemas(t, "Card")
	c := trendCardFrom(trendReadOf(srFixtureCandles(400, func(int) float64 { return 1 })), btcSpec, goldenTime)
	good, err := encodeJSON(cardEnvelope(c))
	if err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(map[string]any){
		"trend.extra":   func(m map[string]any) { m["trend"].(map[string]any)["extra"] = 1 },
		"trend.adx str": func(m map[string]any) { m["trend"].(map[string]any)["adx"] = "40" },
		"no verdict":    func(m map[string]any) { delete(m, "verdict") },
		"blocks.extra":  func(m map[string]any) { m["blocks"].(map[string]any)["extra"] = "x" },
	} {
		var m map[string]any
		if err := json.Unmarshal(good, &m); err != nil {
			t.Fatal(err)
		}
		mutate(m)
		b, _ := json.Marshal(m)
		v, _ := jsonschema.UnmarshalJSON(bytes.NewReader(b))
		if schemas["Card"].Validate(v) == nil {
			t.Errorf("%s: validation passed, want a failure", name)
		}
	}
	v, _ := jsonschema.UnmarshalJSON(bytes.NewReader(good))
	if err := schemas["Card"].Validate(v); err != nil {
		t.Fatalf("unmutated card fails: %v", err)
	}
}

// Every route of the mux is in the document, every agent in the name enum.
func TestOpenAPIPathsCoverRoutes(t *testing.T) {
	doc := openapiDoc(t)
	paths := doc["paths"].(map[string]any)
	var got []string
	for p := range paths {
		got = append(got, p)
	}
	sort.Strings(got)
	want := []string{"/", "/agents", "/agents/trend/chart", "/agents/{name}", "/docs", "/openapi.json", "/openapi.yaml",
		"/showcase", "/showcase/example"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Errorf("paths %v, want %v", got, want)
	}
	params := paths["/agents/{name}"].(map[string]any)["get"].(map[string]any)["parameters"].([]any)
	enum := params[0].(map[string]any)["schema"].(map[string]any)["enum"].([]any)
	names := map[string]bool{}
	for _, e := range enum {
		names[e.(string)] = true
	}
	for _, n := range httpAgentNames {
		if !names[n] {
			t.Errorf("agent %q missing from the {name} enum", n)
		}
	}
	if len(enum) != len(httpAgentNames) {
		t.Errorf("{name} enum has %d agents, the router %d", len(enum), len(httpAgentNames))
	}
}

func TestOpenAPIServed(t *testing.T) {
	_, srv := newTestAPI(t, NewAgents(NewBackendClient("http://127.0.0.1:1")), true)
	st, h, body := httpGet(t, srv.URL+"/openapi.json")
	if st != http.StatusOK || !strings.HasPrefix(h.Get("Content-Type"), "application/json") || !bytes.Equal(body, openapiJSON) {
		t.Fatalf("/openapi.json → %d %q", st, h.Get("Content-Type"))
	}
	var doc map[string]any
	if err := json.Unmarshal(body, &doc); err != nil || doc["openapi"] != "3.1.0" {
		t.Fatalf("/openapi.json is not an OpenAPI 3.1 document: %v %v", err, doc["openapi"])
	}
	st, h, body = httpGet(t, srv.URL+"/openapi.yaml")
	if st != http.StatusOK || !strings.HasPrefix(h.Get("Content-Type"), "application/yaml") || !bytes.Equal(body, openapiYAML) {
		t.Fatalf("/openapi.yaml → %d %q", st, h.Get("Content-Type"))
	}
	st, h, body = httpGet(t, srv.URL+"/docs")
	page := string(body)
	if st != http.StatusOK || !strings.HasPrefix(h.Get("Content-Type"), "text/html") ||
		!strings.Contains(page, "swagger-ui-dist@"+swaggerUIVersion+"/swagger-ui-bundle.js") ||
		!strings.Contains(page, `integrity="`+swaggerUIBundleSRI+`"`) || !strings.Contains(page, `url: "openapi.json"`) {
		t.Fatalf("/docs → %d %q\n%s", st, h.Get("Content-Type"), page)
	}
	if strings.Contains(page, "@latest") || strings.Contains(page, "swagger-ui-dist/") {
		t.Error("/docs must load a pinned Swagger UI version")
	}
}

// The ordered JSON says exactly what a plain YAML decode says.
func TestOpenAPIJSONMatchesYAML(t *testing.T) {
	var fromYAML any
	if err := yaml.Unmarshal(openapiYAML, &fromYAML); err != nil {
		t.Fatal(err)
	}
	a, _ := json.Marshal(fromYAML)
	var x, y any
	if err := json.Unmarshal(a, &x); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(openapiJSON, &y); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(x, y) {
		t.Fatal("openapi.json differs from openapi.yaml")
	}
	if !bytes.HasPrefix(bytes.TrimSpace(openapiJSON), []byte("{\n  \"openapi\": \"3.1.0\"")) {
		t.Errorf("openapi.json does not keep the source order: %.60s", openapiJSON)
	}
}
