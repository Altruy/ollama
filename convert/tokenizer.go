package convert

import (
	"errors"
	"path/filepath"
)

const (
	_ int32 = iota
	tokenTypeNormal
	tokenTypeUnknown
	tokenTypeControl
	tokenTypeUserDefined
	tokenTypeUnused
	tokenTypeByte
)

type Vocabulary struct {
	Tokens []string
	Scores []float32
	Types  []int32
	Merges []string
}

func parseVocabulary(p string) (*Vocabulary, error) {
	patterns := map[string]func(string) (*Vocabulary, error){
		"tokenizer.model": parseSentencePiece,
		"tokenizer.json":  parseBytePairEncoding,
	}

	for pattern, parseFn := range patterns {
		matches, err := filepath.Glob(filepath.Join(p, pattern))
		if err != nil {
			return nil, err
		}

		if len(matches) > 0 {
			return parseFn(p)
		}
	}

	return nil, errors.New("unknown tensor format")
}
