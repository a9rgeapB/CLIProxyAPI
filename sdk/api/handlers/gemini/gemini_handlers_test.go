package gemini

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/api/handlers"
)

func TestCleanGeminiModelName(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"gemini-2.5-flash", "gemini-2.5-flash"},
		{"models/gemini-2.5-flash", "gemini-2.5-flash"},
		{"models/models/gemini-2.5-flash", "gemini-2.5-flash"},
		{"/models/gemini-2.5-flash/", "gemini-2.5-flash"},
		{"   models/gemini-3.8-flash-high   ", "gemini-3.8-flash-high"},
		{"", ""},
		{"   ", ""},
		{"models/", ""},
		{"models/models/", ""},
	}

	for _, tt := range tests {
		got := CleanGeminiModelName(tt.input)
		if got != tt.want {
			t.Errorf("CleanGeminiModelName(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestResolveGeminiModelName(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"gemini-2.5-flash", "gemini-2.5-flash"},
		{"models/models/gemini-2.5-flash", "gemini-2.5-flash"},
		{"Gemini 3.8 Flash (High)", "gemini-3.8-flash-high"},
		{"Gemini 3.8 Flash (Medium)", "gemini-3.8-flash-high(medium)"},
		{"Claude Sonnet 5.5 (High)", "claude-sonnet-5-5-high"},
		{"Claude Sonnet 5.5 (Low)", "claude-sonnet-5-5-high(low)"},
		{"Claude Opus 4.6 (Thinking)", "claude-opus-4-6-thinking"},
		{"Claude Opus 4.6", "claude-opus-4-6"},
		{"Gemini 3 Flash", "gemini-3-flash"},
		{"gemini-3.8-flash-high(medium)", "gemini-3.8-flash-high(medium)"},
	}

	for _, tt := range tests {
		got := ResolveGeminiModelName(tt.input)
		if got != tt.want {
			t.Errorf("ResolveGeminiModelName(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestResolveGeminiModelNamePrefersRegisteredAntigravityModel(t *testing.T) {
	const clientID = "test-gemini-antigravity-resolution"
	const modelID = "gemini-3.8-flash-high"
	registryRef := registry.GetGlobalRegistry()
	registryRef.RegisterClient(clientID, "antigravity", []*registry.ModelInfo{{
		ID: modelID, Name: modelID, DisplayName: "Gemini 3.8 Flash",
	}})
	t.Cleanup(func() { registryRef.UnregisterClient(clientID) })

	for _, input := range []string{"gemini-3.8-flash", "models/gemini-3.8-flash", "Gemini 3.8 Flash"} {
		if got := ResolveGeminiModelName(input); got != modelID {
			t.Errorf("ResolveGeminiModelName(%q) = %q, want %q", input, got, modelID)
		}
	}

	const geminiClientID = "test-gemini-direct-resolution"
	registryRef.RegisterClient(geminiClientID, "gemini", []*registry.ModelInfo{{ID: "gemini-3.8-flash"}})
	t.Cleanup(func() { registryRef.UnregisterClient(geminiClientID) })
	if got := ResolveGeminiModelName("gemini-3.8-flash"); got != "gemini-3.8-flash" {
		t.Errorf("registered exact model resolved to %q", got)
	}
}

func TestGeminiGetHandlerResolvesAntigravityModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Params = gin.Params{{Key: "action", Value: "Gemini 3.8 Flash (High)"}}

	handler := NewGeminiAPIHandler(&handlers.BaseAPIHandler{})
	handler.GeminiGetHandler(ctx)

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d, body: %s", recorder.Code, recorder.Body.String())
	}

	var resp map[string]any
	if err := json.Unmarshal(recorder.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if name, ok := resp["name"].(string); !ok || name != "models/gemini-3.8-flash-high" {
		t.Errorf("expected name 'models/gemini-3.8-flash-high', got %q", resp["name"])
	}
}

func TestGeminiGetHandlerClonesRegistryMap(t *testing.T) {
	gin.SetMode(gin.TestMode)
	const clientID = "test-clone-client"
	const modelID = "test-clone-model-raw"

	registryRef := registry.GetGlobalRegistry()
	registryRef.RegisterClient(clientID, "gemini", []*registry.ModelInfo{{
		ID: modelID, Name: modelID, DisplayName: "Clone Test",
	}})
	t.Cleanup(func() {
		registryRef.UnregisterClient(clientID)
	})

	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Params = gin.Params{{Key: "action", Value: modelID}}

	handler := NewGeminiAPIHandler(&handlers.BaseAPIHandler{})
	handler.GeminiGetHandler(ctx)

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", recorder.Code)
	}

	// Verify the original item in the registry wasn't mutated to have "models/" prefix
	models := registryRef.GetAvailableModels("gemini")
	for _, m := range models {
		if m["id"] == modelID {
			if name, ok := m["name"].(string); ok && name != modelID {
				t.Errorf("registry cache was mutated! name = %q, want %q", name, modelID)
			}
			break
		}
	}
}

func TestGeminiGetHandlerNotFound(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Params = gin.Params{{Key: "action", Value: "completely-unknown-model-xyz"}}

	handler := NewGeminiAPIHandler(&handlers.BaseAPIHandler{})
	handler.GeminiGetHandler(ctx)

	if recorder.Code != http.StatusNotFound {
		t.Fatalf("expected status 404, got %d", recorder.Code)
	}
}

func TestGeminiHandlerInvalidAction(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// 1. Missing colon method
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash", nil)
	ctx.Params = gin.Params{{Key: "action", Value: "gemini-2.5-flash"}}

	handler := NewGeminiAPIHandler(&handlers.BaseAPIHandler{})
	handler.GeminiHandler(ctx)
	if recorder.Code != http.StatusNotFound {
		t.Fatalf("expected status 404 for missing colon, got %d", recorder.Code)
	}

	// 2. Unknown method
	recorder2 := httptest.NewRecorder()
	ctx2, _ := gin.CreateTestContext(recorder2)
	ctx2.Request = httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:unknownMethod", nil)
	ctx2.Params = gin.Params{{Key: "action", Value: "gemini-2.5-flash:unknownMethod"}}

	handler.GeminiHandler(ctx2)
	if recorder2.Code != http.StatusNotFound {
		t.Fatalf("expected status 404 for unknown method, got %d", recorder2.Code)
	}
}
