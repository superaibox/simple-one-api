package ollama

// import myopenai "simple-one-api/pkg/openai"

type ChatResponse struct {
	Model     string      `json:"model"`
	CreatedAt string      `json:"created_at"`
	Message   ChatMessage `json:"message"`
	// Tools              []interface{}          `json:"tools"`
	Done               bool  `json:"done"`
	TotalDuration      int64 `json:"total_duration"`
	LoadDuration       int64 `json:"load_duration"`
	PromptEvalCount    int   `json:"prompt_eval_count"`
	PromptEvalDuration int64 `json:"prompt_eval_duration"`
	EvalCount          int   `json:"eval_count"`
	EvalDuration       int64 `json:"eval_duration"`
}

type OllamaFunctionCall struct {
	Index     *int                   `json:"index,omitempty"`
	Name      string                 `json:"name,omitempty"`
	Arguments map[string]interface{} `json:"arguments,omitempty"`
}

type OllamaToolCall struct {
	ID       string             `json:"id"`
	Function OllamaFunctionCall `json:"function"`
}

type ChatMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	Thinking  *string          `json:"thinking"`
	Images    []string         `json:"images,omitempty"`
	ToolCalls []OllamaToolCall `json:"tool_calls,omitempty"`
}
