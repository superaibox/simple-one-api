package oai

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"
)

func (e *Embedding) UnmarshalJSON(data []byte) error {
	// 临时结构，embedding 先用 json.RawMessage 接收
	var raw struct {
		Object    string          `json:"object"`
		Embedding json.RawMessage `json:"embedding"`
		Index     int             `json:"index"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	e.Object = raw.Object
	e.Index = raw.Index

	// 判断是字符串（base64）还是数组
	if len(raw.Embedding) > 0 && raw.Embedding[0] == '"' {
		// base64 编码的字符串
		var encoded string
		if err := json.Unmarshal(raw.Embedding, &encoded); err != nil {
			return err
		}
		decoded, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return fmt.Errorf("failed to decode base64 embedding: %w", err)
		}
		// 每个 float32 占 4 字节，小端序
		if len(decoded)%4 != 0 {
			return fmt.Errorf("invalid embedding byte length: %d", len(decoded))
		}
		floats := make([]float32, len(decoded)/4)
		for i := range floats {
			bits := binary.LittleEndian.Uint32(decoded[i*4 : (i+1)*4])
			floats[i] = math.Float32frombits(bits)
		}
		e.Embedding = floats
	} else {
		// 普通 JSON 数组
		if err := json.Unmarshal(raw.Embedding, &e.Embedding); err != nil {
			return err
		}
	}
	return nil
}

// GenerateEmbedding 生成文本的嵌入向量
func OpenAIEmbedding(embReq *EmbeddingRequest, apiKey string, proxyTransport *http.Transport, serverURL string) (*EmbeddingResponse, error) {

	url := serverURL + "/embeddings"
	requestBody, err := json.Marshal(embReq)
	if err != nil {
		return nil, fmt.Errorf("JSON 编码错误: %v", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(requestBody))
	if err != nil {
		return nil, fmt.Errorf("创建请求错误: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	var client *http.Client
	if proxyTransport != nil {
		client = &http.Client{
			Timeout:   60 * time.Second,
			Transport: proxyTransport,
		}
	} else {
		client = &http.Client{
			Timeout:   60 * time.Second,
			Transport: http.DefaultTransport, // 使用默认的 Transport
		}
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("请求错误: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("读取响应错误: %v", err)
	}

	var response EmbeddingResponse
	err = json.Unmarshal(body, &response)
	if err != nil {
		return nil, err
	}

	return &response, nil
}
