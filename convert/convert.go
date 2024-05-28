package convert

import (
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/llm"
)

type Parameters struct {
	Architectures []string `json:"architectures"`
}

type Converter interface {
	KV(*Vocabulary) map[string]any
	Tensors([]Tensor) []llm.Tensor

	tensorName(string) (string, error)
}

func Convert(p string, ws io.WriteSeeker) error {
	f, err := os.Open(filepath.Join(p, "config.json"))
	if err != nil {
		return err
	}
	defer f.Close()

	var ps Parameters
	if err := json.NewDecoder(f).Decode(&ps); err != nil {
		return err
	}

	if len(ps.Architectures) < 1 {
		return errors.New("unknown architecture")
	}

	var c Converter
	switch ps.Architectures[0] {
	case "LlamaForCausalLM", "MistralForCausalLM", "MixtralForCausalLM":
		c = &llama{}
	case "GemmaForCausalLM":
		c = &gemma{}
	case "PhiForCausalLM", "Phi3ForCausalLM":
		c = &phi{}
	default:
		return errors.New("unsupported architecture")
	}

	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return err
	}

	if err := json.NewDecoder(f).Decode(&c); err != nil {
		return err
	}

	v, err := parseVocabulary(p)
	if err != nil {
		return err
	}

	ts, err := parseTensors(p)
	if err != nil {
		return err
	}

	return llm.WriteGGUF(ws, c.KV(v), c.Tensors(ts))
}

func cutLast(s, sep string) (before, after string, ok bool) {
	i := strings.LastIndex(s, sep)
	if i >= 0 {
		return s[:i], s[i+len(sep):], true
	}
	return s, "", false
}
