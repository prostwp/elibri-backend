package demobot

// openapi.go — the API's own description: the OpenAPI 3.1 document embedded
// in the binary and served as YAML (/openapi.yaml, the source) and JSON
// (/openapi.json, converted once at start-up), plus a Swagger UI page (/docs)
// that renders the JSON. The document is the contract the site parses; every
// served body is validated against it by openapi_contract_test.go.

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"net/http"

	"gopkg.in/yaml.v3"
)

//go:embed openapi.yaml
var openapiYAML []byte

// openapiJSON is the same document as JSON, built once from the YAML so the
// two can never say different things. A document that does not parse is a
// build defect, caught by the tests; the handler then answers 500.
var openapiJSON, openapiJSONErr = openapiToJSON(openapiYAML)

// openapiToJSON converts the YAML document to JSON keeping the key order of
// the source (a Go map would sort it, and Swagger UI shows fields in the
// order it gets them).
func openapiToJSON(src []byte) ([]byte, error) {
	var root yaml.Node
	if err := yaml.Unmarshal(src, &root); err != nil {
		return nil, fmt.Errorf("openapi.yaml: %w", err)
	}
	if root.Kind != yaml.DocumentNode || len(root.Content) != 1 {
		return nil, fmt.Errorf("openapi.yaml: not one YAML document")
	}
	var buf bytes.Buffer
	if err := yamlNodeJSON(&buf, root.Content[0]); err != nil {
		return nil, fmt.Errorf("openapi.yaml → json: %w", err)
	}
	var out bytes.Buffer
	if err := json.Indent(&out, buf.Bytes(), "", "  "); err != nil {
		return nil, fmt.Errorf("openapi.yaml → json: %w", err)
	}
	return out.Bytes(), nil
}

// yamlNodeJSON writes one YAML node as JSON: mappings in source order,
// scalars by their resolved tag.
func yamlNodeJSON(buf *bytes.Buffer, n *yaml.Node) error {
	switch n.Kind {
	case yaml.AliasNode:
		return yamlNodeJSON(buf, n.Alias)
	case yaml.MappingNode:
		buf.WriteByte('{')
		for i := 0; i+1 < len(n.Content); i += 2 {
			if i > 0 {
				buf.WriteByte(',')
			}
			k, err := json.Marshal(n.Content[i].Value)
			if err != nil {
				return err
			}
			buf.Write(k)
			buf.WriteByte(':')
			if err := yamlNodeJSON(buf, n.Content[i+1]); err != nil {
				return err
			}
		}
		buf.WriteByte('}')
	case yaml.SequenceNode:
		buf.WriteByte('[')
		for i, c := range n.Content {
			if i > 0 {
				buf.WriteByte(',')
			}
			if err := yamlNodeJSON(buf, c); err != nil {
				return err
			}
		}
		buf.WriteByte(']')
	case yaml.ScalarNode:
		var v any
		if err := n.Decode(&v); err != nil {
			return err
		}
		b, err := json.Marshal(v)
		if err != nil {
			return err
		}
		buf.Write(b)
	default:
		return fmt.Errorf("unsupported YAML node kind %d", n.Kind)
	}
	return nil
}

// Swagger UI from jsDelivr, pinned to one released version with subresource
// integrity, so the page runs exactly the files that were reviewed.
const (
	swaggerUIVersion   = "5.33.0"
	swaggerUICSSSRI    = "sha384-Ov4/wv3j2bmct8cDc5X4ngJZohVPzEmc6uDPH8WeljUxO5vtoykvMEfbu9Vh6RaW"
	swaggerUIBundleSRI = "sha384-YDALVcy8kj8yltLBVi1vBiBAUqdxvus673gM8XKwiy6aDUJFXivF/KCufekjYbVf"
)

var swaggerUIPage = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AlphaVizor Demo Bot API</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@` + swaggerUIVersion + `/swagger-ui.css"
  integrity="` + swaggerUICSSSRI + `" crossorigin="anonymous">
<style>body { margin: 0; background: #fafafa; }</style>
</head>
<body>
<div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@` + swaggerUIVersion + `/swagger-ui-bundle.js"
  integrity="` + swaggerUIBundleSRI + `" crossorigin="anonymous"></script>
<script>
window.ui = SwaggerUIBundle({ url: "openapi.json", dom_id: "#swagger-ui", deepLinking: true });
</script>
</body>
</html>
`

func (s *HTTPServer) handleOpenAPIJSON(w http.ResponseWriter, _ *http.Request) {
	if openapiJSONErr != nil {
		writeErr(w, http.StatusInternalServerError, "openapi document unavailable")
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_, _ = w.Write(openapiJSON)
}

func (s *HTTPServer) handleOpenAPIYAML(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/yaml; charset=utf-8")
	_, _ = w.Write(openapiYAML)
}

func (s *HTTPServer) handleDocs(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(swaggerUIPage))
}
