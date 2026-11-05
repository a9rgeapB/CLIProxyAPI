// Package gemini provides HTTP handlers for Gemini API endpoints.
// This package implements handlers for managing Gemini model operations including
// model listing, content generation, streaming content generation, and token counting.
// It serves as a proxy layer between clients and the Gemini backend service,
// handling request translation, client management, and response processing.
package gemini

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/MoeForge/moego/ojson"
	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/clienterror"
	. "github.com/router-for-me/CLIProxyAPI/v8/internal/constant"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/interfaces"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/thinking"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/util"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/api/handlers"
)

// GeminiAPIHandler contains the handlers for Gemini API endpoints.
// It holds a pool of clients to interact with the backend service.
type GeminiAPIHandler struct {
	*handlers.BaseAPIHandler
}

// NewGeminiAPIHandler creates a new Gemini API handlers instance.
// It takes an BaseAPIHandler instance as input and returns a GeminiAPIHandler.
func NewGeminiAPIHandler(apiHandlers *handlers.BaseAPIHandler) *GeminiAPIHandler {
	return &GeminiAPIHandler{
		BaseAPIHandler: apiHandlers,
	}
}

// HandlerType returns the identifier for this handler implementation.
func (h *GeminiAPIHandler) HandlerType() string {
	return Gemini
}

// Models returns the Gemini-compatible model metadata supported by this handler.
func (h *GeminiAPIHandler) Models() []map[string]any {
	// Get dynamic models from the global registry
	modelRegistry := registry.GetGlobalRegistry()
	return modelRegistry.GetAvailableModels("gemini")
}

// GeminiModels handles the Gemini models listing endpoint.
// It returns a JSON response containing available Gemini models and their specifications.
func (h *GeminiAPIHandler) GeminiModels(c *gin.Context) {
	rawModels := h.Models()
	normalizedModels := make([]map[string]any, 0, len(rawModels))
	defaultMethods := []string{"generateContent"}
	for _, model := range rawModels {
		normalizedModel := make(map[string]any, len(model))
		for k, v := range model {
			normalizedModel[k] = v
		}
		if name, ok := normalizedModel["name"].(string); ok && name != "" {
			if !strings.HasPrefix(name, "models/") {
				normalizedModel["name"] = "models/" + name
			}
			if displayName, _ := normalizedModel["displayName"].(string); displayName == "" {
				normalizedModel["displayName"] = name
			}
			if description, _ := normalizedModel["description"].(string); description == "" {
				normalizedModel["description"] = name
			}
		}
		if _, ok := normalizedModel["supportedGenerationMethods"]; !ok {
			normalizedModel["supportedGenerationMethods"] = defaultMethods
		}
		normalizedModels = append(normalizedModels, normalizedModel)
	}
	h.WriteModelListResponse(c, h.HandlerType(), gin.H{
		"models": normalizedModels,
	})
}

// CleanGeminiModelName cleans a model name by trimming whitespace, slashes,
// and stripping any repeated "models/" prefix (e.g., "models/models/gemini-2.5-flash").
func CleanGeminiModelName(raw string) string {
	clean := strings.Trim(raw, "/ \t\r\n")
	for {
		if clean == "models" {
			return ""
		}
		if strings.HasPrefix(clean, "models/") {
			clean = strings.TrimPrefix(clean, "models/")
			clean = strings.Trim(clean, "/ \t\r\n")
			continue
		}
		break
	}
	if clean == "models" {
		return ""
	}
	return clean
}

func normalizeSlug(s string) string {
	var b strings.Builder
	prevDash := false
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			prevDash = false
		} else if r >= 'A' && r <= 'Z' {
			b.WriteRune(r + ('a' - 'A'))
			prevDash = false
		} else {
			if !prevDash && b.Len() > 0 {
				b.WriteByte('-')
				prevDash = true
			}
		}
	}
	res := b.String()
	return strings.TrimRight(res, "-")
}

func stripCandidateLevel(id string) string {
	for _, suffix := range []string{"-high", "-thinking", "-medium", "-low"} {
		if strings.HasSuffix(id, suffix) {
			return strings.TrimSuffix(id, suffix)
		}
	}
	return id
}

func getAllGeminiCandidateModels() []*registry.ModelInfo {
	seen := make(map[string]struct{})
	var candidates []*registry.ModelInfo

	add := func(list []*registry.ModelInfo) {
		for _, m := range list {
			if m == nil || m.ID == "" {
				continue
			}
			if _, ok := seen[m.ID]; ok {
				continue
			}
			seen[m.ID] = struct{}{}
			candidates = append(candidates, m)
		}
	}

	if reg := registry.GetGlobalRegistry(); reg != nil {
		add(reg.GetAvailableModelInfos())
	}
	add(registry.GetAntigravityModels())
	add(registry.GetGeminiModels())
	add(registry.GetGeminiVertexModels())
	add(registry.GetAIStudioModels())
	add(registry.GetClaudeModels())

	return candidates
}

// ResolveGeminiModelName maps human-readable display names, client slugs,
// and thinking suffix variants to canonical model IDs registered in CPA.
func ResolveGeminiModelName(raw string) string {
	clean := CleanGeminiModelName(raw)
	if clean == "" {
		return ""
	}

	// 1. Direct provider resolution fast path
	if len(util.GetProviderName(clean)) > 0 {
		return clean
	}
	lowerClean := strings.ToLower(clean)
	if len(util.GetProviderName(lowerClean)) > 0 {
		return lowerClean
	}

	// 2. Thinking suffix fast path, e.g. "gemini-3.8-flash-high(medium)"
	if parsed := thinking.ParseSuffix(clean); parsed.HasSuffix {
		if len(util.GetProviderName(parsed.ModelName)) > 0 {
			return clean
		}
		if len(util.GetProviderName(strings.ToLower(parsed.ModelName))) > 0 {
			return fmt.Sprintf("%s(%s)", strings.ToLower(parsed.ModelName), parsed.RawSuffix)
		}
	}

	// Prefer aliases of models that can actually be routed. Static catalogs may
	// contain an exact ID for an inactive provider (for example, Gemini's bare
	// flash ID while only Antigravity's flash-high ID is registered).
	if reg := registry.GetGlobalRegistry(); reg != nil {
		if resolved := resolveGeminiModelCandidate(clean, reg.GetAvailableModelInfos()); resolved != "" {
			if len(util.GetProviderName(thinking.ParseSuffix(resolved).ModelName)) > 0 {
				return resolved
			}
		}
	}
	if resolved := resolveGeminiModelCandidate(clean, getAllGeminiCandidateModels()); resolved != "" {
		return resolved
	}
	return clean
}

func resolveGeminiModelCandidate(clean string, candidates []*registry.ModelInfo) string {

	// 3. Exact case-insensitive match on ID, Name, DisplayName, Description
	for _, m := range candidates {
		if strings.EqualFold(clean, m.ID) ||
			(m.Name != "" && strings.EqualFold(clean, m.Name)) ||
			(m.DisplayName != "" && strings.EqualFold(clean, m.DisplayName)) ||
			(m.Description != "" && strings.EqualFold(clean, m.Description)) {
			return m.ID
		}
	}

	cleanSlug := normalizeSlug(clean)

	// 4. Slug match on ID, Name, DisplayName, Description
	for _, m := range candidates {
		if cleanSlug == normalizeSlug(m.ID) ||
			(m.Name != "" && cleanSlug == normalizeSlug(m.Name)) ||
			(m.DisplayName != "" && cleanSlug == normalizeSlug(m.DisplayName)) ||
			(m.Description != "" && cleanSlug == normalizeSlug(m.Description)) {
			return m.ID
		}
	}

	// 5. Check if clean has thinking level in parentheses, e.g. "Gemini 3.8 Flash (High)", "Gemini 3.8 Flash (Medium)"
	lastOpen := strings.LastIndex(clean, "(")
	if lastOpen > 0 && strings.HasSuffix(clean, ")") {
		baseText := strings.TrimSpace(clean[:lastOpen])
		levelText := strings.TrimSpace(clean[lastOpen+1 : len(clean)-1])
		levelLower := strings.ToLower(levelText)
		baseSlug := normalizeSlug(baseText)

		for _, m := range candidates {
			candBaseID := stripCandidateLevel(m.ID)
			candBaseSlug := normalizeSlug(candBaseID)
			candDispSlug := normalizeSlug(stripCandidateLevel(m.DisplayName))
			candDescSlug := normalizeSlug(stripCandidateLevel(m.Description))

			if baseSlug == candBaseSlug || (candDispSlug != "" && baseSlug == candDispSlug) || (candDescSlug != "" && baseSlug == candDescSlug) {
				// Default thinking level already part of ID (e.g. gemini-3.8-flash-high, claude-opus-4-6-thinking)
				if (levelLower == "high" && strings.HasSuffix(m.ID, "-high")) ||
					(levelLower == "thinking" && strings.HasSuffix(m.ID, "-thinking")) {
					return m.ID
				}
				// Otherwise attach specified level, e.g. gemini-3.8-flash-high(medium)
				return fmt.Sprintf("%s(%s)", m.ID, levelLower)
			}
		}
	}

	// 6. Base slug match without thinking level (e.g. "Claude Opus 4.6" -> "claude-opus-4-6-thinking")
	for _, m := range candidates {
		candBase := stripCandidateLevel(m.ID)
		if candBase != m.ID {
			if cleanSlug == normalizeSlug(candBase) {
				return m.ID
			}
		}
	}

	return ""
}

// GeminiGetHandler handles GET requests for specific Gemini model information.
// It returns detailed information about a specific Gemini model based on the action parameter.
func (h *GeminiAPIHandler) GeminiGetHandler(c *gin.Context) {
	var request struct {
		Action string `uri:"action"`
	}
	_ = c.ShouldBindUri(&request)
	actionStr := request.Action
	if actionStr == "" {
		actionStr = c.Param("action")
	}
	action := strings.TrimPrefix(actionStr, "/")
	cleanAction := CleanGeminiModelName(action)
	resolvedAction := ResolveGeminiModelName(cleanAction)

	// Get dynamic models from the global registry and find the matching one
	availableModels := h.Models()
	var targetModel map[string]any

	for _, model := range availableModels {
		name, _ := model["name"].(string)
		cleanName := CleanGeminiModelName(name)
		if cleanName == cleanAction || cleanName == resolvedAction || name == action || name == "models/"+action {
			targetModel = model
			break
		}
	}

	if targetModel != nil {
		cloned := make(map[string]any, len(targetModel))
		for k, v := range targetModel {
			cloned[k] = v
		}
		if name, ok := cloned["name"].(string); ok && name != "" {
			if !strings.HasPrefix(name, "models/") {
				cloned["name"] = "models/" + name
			}
		} else if cleanAction != "" {
			cloned["name"] = "models/" + cleanAction
		}
		c.JSON(http.StatusOK, cloned)
		return
	}

	for _, m := range getAllGeminiCandidateModels() {
		if m.ID == cleanAction || m.ID == resolvedAction || normalizeSlug(m.ID) == normalizeSlug(cleanAction) {
			result := map[string]any{
				"name":                       "models/" + m.ID,
				"displayName":                m.DisplayName,
				"description":                m.Description,
				"supportedGenerationMethods": []string{"generateContent", "countTokens"},
			}
			if result["displayName"] == "" {
				result["displayName"] = m.ID
			}
			if result["description"] == "" {
				result["description"] = m.ID
			}
			c.JSON(http.StatusOK, result)
			return
		}
	}

	c.JSON(http.StatusNotFound, handlers.ErrorResponse{
		Error: handlers.ErrorDetail{
			Message: "Not Found",
			Type:    "not_found",
		},
	})
}

// GeminiHandler handles POST requests for Gemini API operations.
// It routes requests to appropriate handlers based on the action parameter (model:method format).
func (h *GeminiAPIHandler) GeminiHandler(c *gin.Context) {
	var request struct {
		Action string `uri:"action"`
	}
	_ = c.ShouldBindUri(&request)
	actionStr := request.Action
	if actionStr == "" {
		actionStr = c.Param("action")
	}
	rawAction := strings.TrimPrefix(actionStr, "/")
	idx := strings.LastIndex(rawAction, ":")
	if idx <= 0 || idx == len(rawAction)-1 {
		path := ""
		if c.Request != nil && c.Request.URL != nil {
			path = c.Request.URL.Path
		}
		c.JSON(http.StatusNotFound, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: fmt.Sprintf("%s not found.", path),
				Type:    "invalid_request_error",
			},
		})
		return
	}

	rawModel := rawAction[:idx]
	method := rawAction[idx+1:]
	cleanModel := CleanGeminiModelName(rawModel)
	resolvedModel := ResolveGeminiModelName(cleanModel)
	if resolvedModel == "" {
		resolvedModel = cleanModel
	}

	var rawJSON []byte
	if c.Request != nil && c.Request.Body != nil {
		var errRead error
		rawJSON, errRead = c.GetRawData()
		if errRead != nil {
			c.JSON(clienterror.HTTPStatusFromErrorOr(errRead, http.StatusBadRequest), handlers.ErrorResponse{
				Error: handlers.ErrorDetail{
					Message: fmt.Sprintf("Invalid request: %v", errRead),
					Type:    "invalid_request_error",
				},
			})
			return
		}
	}

	if len(rawJSON) > 0 && ojson.ValidBytes(rawJSON) {
		if modelVal := ojson.GetBytes(rawJSON, "model"); modelVal.Exists() {
			if updated, errSet := ojson.SetBytes(rawJSON, "model", resolvedModel); errSet == nil {
				rawJSON = updated
			}
		}
	}

	switch method {
	case "generateContent":
		h.handleGenerateContent(c, resolvedModel, rawJSON)
	case "streamGenerateContent":
		h.handleStreamGenerateContent(c, resolvedModel, rawJSON)
	case "countTokens":
		h.handleCountTokens(c, resolvedModel, rawJSON)
	default:
		path := ""
		if c.Request != nil && c.Request.URL != nil {
			path = c.Request.URL.Path
		}
		c.JSON(http.StatusNotFound, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: fmt.Sprintf("%s not found.", path),
				Type:    "invalid_request_error",
			},
		})
	}
}

// handleStreamGenerateContent handles streaming content generation requests for Gemini models.
// This function establishes a Server-Sent Events connection and streams the generated content
// back to the client in real-time. It supports both SSE format and direct streaming based
// on the 'alt' query parameter.
//
// Parameters:
//   - c: The Gin context for the request
//   - modelName: The name of the Gemini model to use for content generation
//   - rawJSON: The raw JSON request body containing generation parameters
func (h *GeminiAPIHandler) handleStreamGenerateContent(c *gin.Context, modelName string, rawJSON []byte) {
	alt := h.GetAlt(c)

	// Get the http.Flusher interface to manually flush the response.
	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: "Streaming not supported",
				Type:    "server_error",
			},
		})
		return
	}

	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
	setSSEHeaders := func() {
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")
		c.Header("Access-Control-Allow-Origin", "*")
	}
	var upstreamHeaders http.Header
	committed := false
	commitHeaders := func() {
		if committed {
			return
		}
		if alt == "" {
			setSSEHeaders()
		}
		handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)
		committed = true
	}
	writeKeepAlive := func() bool {
		commitHeaders()
		_, _ = c.Writer.Write([]byte(": keep-alive\n\n"))
		flusher.Flush()
		return true
	}
	var keepAliveC <-chan time.Time
	stopKeepAlive := func() {}
	if alt == "" {
		keepAliveC, stopKeepAlive = handlers.NewPreFirstTokenKeepAliveTicker(h.CurrentConfig())
	}
	defer stopKeepAlive()

	dataChan, resolvedHeaders, errChan, okBootstrap := h.ExecuteStreamWithPreFTKeepAlive(
		c, cliCtx, func(err error) { cliCancel(err) }, h.HandlerType(), modelName, rawJSON, alt, keepAliveC, writeKeepAlive,
	)
	if !okBootstrap {
		return
	}
	upstreamHeaders = resolvedHeaders

	// Peek at the first chunk
	for {
		select {
		case <-c.Request.Context().Done():
			cliCancel(c.Request.Context().Err())
			return
		case errMsg, ok := <-errChan:
			if !ok {
				// Err channel closed cleanly; wait for data channel.
				errChan = nil
				continue
			}
			if committed {
				stopKeepAlive()
				pendingErrors := make(chan *interfaces.ErrorMessage, 1)
				pendingErrors <- errMsg
				close(pendingErrors)
				h.forwardGeminiStream(c, flusher, alt, func(err error) { cliCancel(err) }, make(chan []byte), pendingErrors)
				return
			}
			// Upstream failed immediately. Return proper error status and JSON.
			h.WriteErrorResponse(c, errMsg)
			if errMsg != nil {
				cliCancel(errMsg.Error)
			} else {
				cliCancel(nil)
			}
			return
		case chunk, ok := <-dataChan:
			if !ok {
				stopKeepAlive()
				if errMsg, hasPendingError := handlers.PendingStreamError(errChan); hasPendingError {
					if committed {
						pendingErrors := make(chan *interfaces.ErrorMessage, 1)
						pendingErrors <- errMsg
						close(pendingErrors)
						h.forwardGeminiStream(c, flusher, alt, func(err error) { cliCancel(err) }, make(chan []byte), pendingErrors)
						return
					}
					h.WriteErrorResponse(c, errMsg)
					if errMsg != nil {
						cliCancel(errMsg.Error)
					} else {
						cliCancel(nil)
					}
					return
				}
				// Closed without data
				commitHeaders()
				flusher.Flush()
				cliCancel(nil)
				return
			}

			// Success! Set headers.
			stopKeepAlive()
			commitHeaders()

			// Write first chunk
			if alt == "" {
				_, _ = c.Writer.Write([]byte("data: "))
				_, _ = c.Writer.Write(chunk)
				_, _ = c.Writer.Write([]byte("\n\n"))
			} else {
				_, _ = c.Writer.Write(chunk)
			}
			flusher.Flush()

			// Continue
			h.forwardGeminiStream(c, flusher, alt, func(err error) { cliCancel(err) }, dataChan, errChan)
			return
		case <-keepAliveC:
			writeKeepAlive()
		}
	}
}

// handleCountTokens handles token counting requests for Gemini models.
// This function counts the number of tokens in the provided content without
// generating a response. It's useful for quota management and content validation.
//
// Parameters:
//   - c: The Gin context for the request
//   - modelName: The name of the Gemini model to use for token counting
//   - rawJSON: The raw JSON request body containing the content to count
func (h *GeminiAPIHandler) handleCountTokens(c *gin.Context, modelName string, rawJSON []byte) {
	c.Header("Content-Type", "application/json")
	alt := h.GetAlt(c)
	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
	resp, upstreamHeaders, errMsg := h.ExecuteCountWithAuthManager(cliCtx, h.HandlerType(), modelName, rawJSON, alt)
	if errMsg != nil {
		h.WriteErrorResponse(c, errMsg)
		cliCancel(errMsg.Error)
		return
	}
	handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)
	_, _ = c.Writer.Write(resp)
	cliCancel()
}

// handleGenerateContent handles non-streaming content generation requests for Gemini models.
// This function processes the request synchronously and returns the complete generated
// response in a single API call. It supports various generation parameters and
// response formats.
//
// Parameters:
//   - c: The Gin context for the request
//   - modelName: The name of the Gemini model to use for content generation
//   - rawJSON: The raw JSON request body containing generation parameters and content
func (h *GeminiAPIHandler) handleGenerateContent(c *gin.Context, modelName string, rawJSON []byte) {
	c.Header("Content-Type", "application/json")
	alt := h.GetAlt(c)
	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
	stopKeepAlive := h.StartNonStreamingKeepAlive(c, cliCtx)
	resp, upstreamHeaders, errMsg := h.ExecuteWithAuthManager(cliCtx, h.HandlerType(), modelName, rawJSON, alt)
	stopKeepAlive()
	if errMsg != nil {
		h.WriteErrorResponse(c, errMsg)
		cliCancel(errMsg.Error)
		return
	}
	handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)
	_, _ = c.Writer.Write(resp)
	cliCancel()
}

func (h *GeminiAPIHandler) forwardGeminiStream(c *gin.Context, flusher http.Flusher, alt string, cancel func(error), data <-chan []byte, errs <-chan *interfaces.ErrorMessage) {
	var keepAliveInterval *time.Duration
	if alt != "" {
		keepAliveInterval = new(time.Duration(0))
	}

	h.ForwardStream(c, flusher, cancel, data, errs, handlers.StreamForwardOptions{
		KeepAliveInterval: keepAliveInterval,
		WriteChunk: func(chunk []byte) {
			if alt == "" {
				_, _ = c.Writer.Write([]byte("data: "))
				_, _ = c.Writer.Write(chunk)
				_, _ = c.Writer.Write([]byte("\n\n"))
			} else {
				_, _ = c.Writer.Write(chunk)
			}
		},
		WriteTerminalError: func(errMsg *interfaces.ErrorMessage) {
			if errMsg == nil {
				return
			}
			status := http.StatusInternalServerError
			if errMsg.StatusCode > 0 {
				status = errMsg.StatusCode
			}
			errText := http.StatusText(status)
			if errMsg.Error != nil && errMsg.Error.Error() != "" {
				errText = errMsg.Error.Error()
			}
			body := handlers.BuildErrorResponseBody(status, errText)
			if alt == "" {
				_, _ = fmt.Fprintf(c.Writer, "event: error\ndata: %s\n\n", string(body))
			} else {
				_, _ = c.Writer.Write(body)
			}
		},
	})
}
