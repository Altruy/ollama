package convert

import (
	"encoding/json"
	"os"
	"path/filepath"
)

type BytePairEncoding struct {
	Version     string     `json:"version"`
	AddedTokens []bpeToken `json:"added_tokens"`
	Model       struct {
		Type   string         `json:"type"`
		Vocab  map[string]int `json:"vocab"`
		Merges []string       `json:"merges"`
	} `json:"model"`

	PreTokenizer struct {
		PreTokenizers []struct {
			Type    string `json:"type"`
			Pattern struct {
				Regex string `json:"Regex"`
			} `json:"pattern"`
		} `json:"pretokenizers"`
	} `json:"pre_tokenizer"`
}

type bpeToken struct {
	ID          int    `json:"id"`
	Content     string `json:"content"`
	Special     bool   `json:"special"`
	UserDefined bool
}

func parseBytePairEncoding(p string) (*Vocabulary, error) {
	f, err := os.Open(filepath.Join(p, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var bpe BytePairEncoding
	if err := json.NewDecoder(f).Decode(&bpe); err != nil {
		return nil, err
	}

	var tokens []bpeToken
	for k, v := range bpe.Model.Vocab {
		tokens = append(tokens, bpeToken{
			ID:      v,
			Content: k,
		})
	}

	for _, t := range bpe.AddedTokens {
		t.UserDefined = true
		tokens = append(tokens, t)
	}

	var vocab Vocabulary
	for _, t := range tokens {
		vocab.Tokens = append(vocab.Tokens, t.Content)
		vocab.Scores = append(vocab.Scores, float32(t.ID))

		switch {
		case t.Special:
			vocab.Types = append(vocab.Types, tokenTypeControl)
		case t.UserDefined:
			vocab.Types = append(vocab.Types, tokenTypeUserDefined)
		default:
			vocab.Types = append(vocab.Types, tokenTypeNormal)
		}
	}

	vocab.Merges = bpe.Model.Merges
	return &vocab, nil
}
