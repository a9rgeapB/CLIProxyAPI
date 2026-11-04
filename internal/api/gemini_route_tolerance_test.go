package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestGeminiRouteToleranceModels(t *testing.T) {
	server := newTestServer(t)

	routes := []string{
		"/v1beta/models",
		"/v1beta/v1beta/models",
		"/v1/v1beta/models",
		"/models",
	}

	for _, route := range routes {
		t.Run(route, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, route, nil)
			req.Header.Set("Authorization", "Bearer test-key")
			rr := httptest.NewRecorder()
			server.engine.ServeHTTP(rr, req)

			if rr.Code != http.StatusOK {
				t.Fatalf("route %s returned status %d, body: %s", route, rr.Code, rr.Body.String())
			}

			var resp struct {
				Models []any `json:"models"`
			}
			if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
				t.Fatalf("failed to decode models from %s: %v", route, err)
			}
		})
	}
}

func TestGeminiRouteToleranceModelDetail(t *testing.T) {
	server := newTestServer(t)

	routes := []string{
		"/v1beta/models/Gemini%203.8%20Flash%20(High)",
		"/v1beta/v1beta/models/Gemini%203.8%20Flash%20(High)",
		"/v1/v1beta/models/Gemini%203.8%20Flash%20(High)",
		"/models/Gemini%203.8%20Flash%20(High)",
	}

	for _, route := range routes {
		t.Run(route, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, route, nil)
			req.Header.Set("Authorization", "Bearer test-key")
			rr := httptest.NewRecorder()
			server.engine.ServeHTTP(rr, req)

			if rr.Code != http.StatusOK {
				t.Fatalf("route %s returned status %d, body: %s", route, rr.Code, rr.Body.String())
			}

			var resp map[string]any
			if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
				t.Fatalf("failed to decode model detail from %s: %v", route, err)
			}
			name, ok := resp["name"].(string)
			if !ok || !strings.Contains(name, "gemini-3.8-flash-high") {
				t.Errorf("unexpected model name in response from %s: %v", route, resp["name"])
			}
		})
	}
}

func TestGeminiRouteToleranceActionDispatch(t *testing.T) {
	server := newTestServer(t)

	// When an action request reaches GeminiHandler with an invalid method,
	// GeminiHandler returns JSON with type="invalid_request_error".
	// An unregistered Gin route would return a raw Gin 404 without our error JSON structure.
	routes := []string{
		"/v1beta/models/gemini-2.5-flash:testProbe",
		"/v1beta/v1beta/models/gemini-2.5-flash:testProbe",
		"/v1/v1beta/models/gemini-2.5-flash:testProbe",
		"/v1/models/gemini-2.5-flash:testProbe",
		"/models/gemini-2.5-flash:testProbe",
	}

	for _, route := range routes {
		t.Run(route, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, route, strings.NewReader(`{}`))
			req.Header.Set("Authorization", "Bearer test-key")
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()
			server.engine.ServeHTTP(rr, req)

			if rr.Code != http.StatusNotFound {
				t.Fatalf("route %s returned status %d, body: %s", route, rr.Code, rr.Body.String())
			}

			var resp struct {
				Error struct {
					Message string `json:"message"`
					Type    string `json:"type"`
				} `json:"error"`
			}
			if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
				t.Fatalf("route %s failed to parse JSON error: %v, raw: %s", route, err, rr.Body.String())
			}
			if resp.Error.Type != "invalid_request_error" {
				t.Errorf("route %s did not reach GeminiHandler, error type = %q", route, resp.Error.Type)
			}
		})
	}
}
