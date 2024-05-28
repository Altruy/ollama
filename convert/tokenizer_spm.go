package convert

import (
	"os"
	"path/filepath"

	"google.golang.org/protobuf/proto"

	"github.com/ollama/ollama/convert/sentencepiece"
)

type SentencePiece struct {
	proto *sentencepiece.ModelProto
}

func parseSentencePiece(p string) (*Vocabulary, error) {
	bts, err := os.ReadFile(filepath.Join(p, "tokenizer.model"))
	if err != nil {
		return nil, err
	}

	var spm sentencepiece.ModelProto
	if err := proto.Unmarshal(bts, &spm); err != nil {
		return nil, err
	}

	var vocab Vocabulary
	for _, piece := range spm.GetPieces() {
		vocab.Tokens = append(vocab.Tokens, piece.GetPiece())
		vocab.Scores = append(vocab.Scores, piece.GetScore())

		switch t := piece.GetType(); t {
		case sentencepiece.ModelProto_SentencePiece_UNKNOWN,
			sentencepiece.ModelProto_SentencePiece_CONTROL,
			sentencepiece.ModelProto_SentencePiece_UNUSED,
			sentencepiece.ModelProto_SentencePiece_BYTE:
			vocab.Types = append(vocab.Types, int32(t))
		default:
			vocab.Types = append(vocab.Types, int32(sentencepiece.ModelProto_SentencePiece_NORMAL))
		}
	}

	return &vocab, nil
}
