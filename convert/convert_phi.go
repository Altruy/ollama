package convert

import "github.com/ollama/ollama/llm"

type phi struct {
}

func (p *phi) KV(v *Vocabulary) map[string]any {
	kv := map[string]any{
		"general.architecture": "phi",
	}

	return kv
}

func (p *phi) Tensors(ts []Tensor) []llm.Tensor {
	var out []llm.Tensor
	return out
}

func (p *phi) tensorName(name string) (string, error) {
	return name, nil
}
