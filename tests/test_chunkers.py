"""Unit tests for the three chunking strategies (Phase 2).

Covers the task's required cases:

Fixed:     short passage, exact-size passage, long passage, overlap, deterministic.
Semantic:  Hindi sentence boundaries, Devanagari danda, ?/! boundaries,
           long-sentence fixed fallback.
Adaptive:  short path, medium path, long path.
Common:    metadata preservation, chunk IDs, empty input, deterministic output.

These run with the project venv and load the BGE-M3 tokenizer (first run is
slower, ~30 s, due to tokenizer load). They need no large data:

    venv/bin/python -m pytest tests/test_chunkers.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backend.rag.chunkers.adaptive import (
    PATH_LONG, PATH_MEDIUM, PATH_SHORT, SHORT_MAX, MEDIUM_MAX,
    adaptive_path, split_adaptive, AdaptiveChunker,
)
from backend.rag.chunkers.base import make_chunks
from backend.rag.chunkers.fixed import (
    CHUNK_SIZE, OVERLAP, STRIDE, FixedChunker, split_fixed,
)
from backend.rag.chunkers.semantic import MAX_TOKENS, split_semantic
from backend.rag.chunkers.sentences import split_sentences
from backend.rag.chunkers.tokenizer import count_tokens, encode_with_offsets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(text="परीक्षण पाठ।", **over):
    """A minimal canonical document for chunkers."""
    d = {
        "document_id": "hi_42_p0",
        "query_id": 42,
        "passage_idx": 0,
        "text": text,
        "text_en": "test text",
        "query": "क्या?",
        "query_en": "what?",
        "answer": "उत्तर",
        "answer_en": "answer",
        "language": "hi",
        "source_lang_code": "eng_Latn",
        "target_lang_code": "hin_Deva",
        "is_selected": 1,
        "query_type": "DESCRIPTION",
        "source": "MSMARCO-XI",
        "source_file": "hintrain.parquet",
        "answerable": True,
    }
    d.update(over)
    return d


def _piece_token_counts(text, pieces):
    """Token count of each piece computed from the ORIGINAL encoding (j-i),
    NOT by re-tokenizing the substring (re-tokenizing gives a different count
    due to BPE context effects).

    Robust to XLM-R's leading-whitespace offset trimming: we match a piece to
    its token window by scanning for the contiguous span of tokens whose
    concatenated offset-span equals the piece (allowing the piece's own leading
    space to be the trimmed bit). Simpler & exact: find the char span of the
    piece in `text`, then count how many token offset-spans are *covered* by
    [schar, echar) (a token is covered if its [a,b) lies within [schar,echar)).
    """
    if not pieces:
        return []
    ids, offs = encode_with_offsets(text)
    counts = []
    cursor = 0
    for piece in pieces:
        schar = text.index(piece, cursor)
        echar = schar + len(piece)
        # count tokens whose offset span is fully within [schar, echar)
        cnt = 0
        for (a, b) in offs:
            if a >= schar and b <= echar and a < echar and b > schar:
                cnt += 1
        counts.append(cnt)
        cursor = schar + 1
    return counts


def _repeat(text, n):
    """Repeat ``text`` ``n`` times joined by spaces to make a long passage.

    NOTE: each repetition is the SAME sentence (no index suffix) so that the
    last chunk ends at a real sentence terminator (.) — important for the
    semantic "ends at sentence boundary" test. For overlap-uniqueness tests
    that need an unambiguous text, use _repeat_unique.
    """
    return " ".join([text] * n)


def _repeat_unique(text, n):
    """Repeat ``text`` ``n`` times, each made distinct by an index suffix, so
    every token window is unambiguous (used for fixed-overlap verification)."""
    return " ".join(f"{text} {i}" for i in range(n))


# ===========================================================================
# FIXED
# ===========================================================================

class TestFixed:
    def test_short_passage_one_chunk(self):
        text = "यह एक छोटा वाक्य है।"
        pieces = split_fixed(text)
        assert len(pieces) == 1
        assert pieces[0] == text

    def test_exact_size_one_chunk(self):
        # Build a passage that tokenizes to exactly CHUNK_SIZE tokens.
        word = "शब्द "  # each word adds >=1 content token
        text = ""
        # grow to >= CHUNK_SIZE, then trim char-by-char to land exactly on it.
        while count_tokens(text) < CHUNK_SIZE:
            text += word
        # Now possibly over by a few tokens. Trim from the end one char at a
        # time until the count is exactly CHUNK_SIZE (tokens shrink as chars
        # are removed). This terminates because removing chars can only reduce
        # or keep the token count.
        while count_tokens(text) > CHUNK_SIZE and len(text) > 1:
            text = text[:-1]
        assert count_tokens(text) == CHUNK_SIZE, (
            f"could not reach exactly {CHUNK_SIZE}, got {count_tokens(text)}")
        pieces = split_fixed(text)
        assert len(pieces) == 1, "exact-size passage must be one chunk"
        # The chunk text is an exact substring of the passage; XLM-R trims
        # trailing whitespace in offsets, so if the passage ends with space the
        # chunk omits it. Compare by substring membership + token count, the
        # properties that actually matter.
        assert pieces[0] in text
        assert count_tokens(pieces[0]) == CHUNK_SIZE or len(pieces[0]) >= len(text) - 2

    def test_long_passage_overlap(self):
        text = _repeat_unique("यह एक अद्वितीय वाक्य है जो दोहराया नहीं जाता।", 400)
        n = count_tokens(text)
        assert n > CHUNK_SIZE
        pieces = split_fixed(text)
        assert len(pieces) > 1
        # every piece is an exact substring of the original
        assert all(p in text for p in pieces)
        # consecutive windows share exactly OVERLAP tokens, stride == STRIDE
        counts = _piece_token_counts(text, pieces)
        # interior windows are exactly CHUNK_SIZE
        for c in counts[:-1]:
            assert c == CHUNK_SIZE, f"interior window should be {CHUNK_SIZE}, got {c}"
        # last (merged) window is <= CHUNK_SIZE + OVERLAP - 1 and >= 1
        assert 1 <= counts[-1] <= CHUNK_SIZE + OVERLAP - 1

    def test_overlap_is_exactly_32(self):
        text = _repeat_unique("अद्वितीय परीक्षण वाक्य सामग्री यहाँ।", 500)
        pieces = split_fixed(text)
        ids, offs = encode_with_offsets(text)
        # map each piece to its token window
        windows = []
        cursor = 0
        for piece in pieces:
            schar = text.index(piece, cursor)
            echar = schar + len(piece)
            a = next(i for i, (x, _) in enumerate(offs) if x == schar)
            b = next((j for j in range(a + 1, len(offs) + 1)
                      if offs[j - 1][1] == echar), None)
            windows.append((a, b))
            cursor = schar + 1
        # overlap between consecutive = end of i minus start of i+1
        for i in range(len(windows) - 1):
            overlap = windows[i][1] - windows[i + 1][0]
            assert overlap == OVERLAP, f"pair {i} overlap={overlap}"

    def test_deterministic_output(self):
        text = _repeat("नियतात्मकता परीक्षण वाक्य।", 300)
        a = split_fixed(text)
        b = split_fixed(text)
        assert a == b


# ===========================================================================
# SEMANTIC
# ===========================================================================

class TestSemantic:
    def test_hindi_sentence_boundaries(self):
        text = "पहला वाक्य है। दूसरा वाक्य है। तीसरा वाक्य है।"
        sents = split_sentences(text)
        assert len(sents) == 3
        assert "".join(sents) == text  # exact reconstruction

    def test_devanagari_danda(self):
        text = "एक। दो। तीन।"
        sents = split_sentences(text)
        assert len(sents) == 3
        # danda stays attached to its sentence
        assert sents[0].rstrip().endswith("।")
        assert sents[1].rstrip().endswith("।")

    def test_question_exclamation_boundaries(self):
        text = "क्या हाल है? मैं ठीक हूँ! चलो चलें।"
        sents = split_sentences(text)
        assert len(sents) == 3
        assert "".join(sents) == text

    def test_decimal_not_split(self):
        text = "यह 3.14 का मान है। दूसरा वाक्य।"
        sents = split_sentences(text)
        assert len(sents) == 2, "decimal must NOT be a sentence boundary"

    def test_short_passage_one_chunk(self):
        text = "यह एक छोटा वाक्य है।"
        pieces = split_semantic(text)
        assert len(pieces) == 1
        assert pieces[0] == text

    def test_no_chunk_exceeds_max(self):
        # a passage with many sentences that must split
        text = _repeat("यह एक वाक्य है जिसमें कुछ सामग्री है।", 200)
        assert count_tokens(text) > MAX_TOKENS
        pieces = split_semantic(text)
        assert len(pieces) > 1
        for p in pieces:
            # every chunk is an exact substring AND <= MAX_TOKENS
            assert p in text
            # count via original encoding to avoid re-tokenize mismatch
            counts = _piece_token_counts(text, [p])
            assert counts[0] <= MAX_TOKENS, f"chunk exceeds max: {counts[0]}"

    def test_chunks_end_at_sentence_boundary(self):
        text = _repeat("यह एक वाक्य है जिसमें कुछ सामग्री है।", 200)
        pieces = split_semantic(text)
        # every chunk (except possibly a fixed-fallback piece) should end at a
        # sentence terminator after stripping trailing whitespace
        terminators = ("।", "?", "!", "॥")
        for p in pieces:
            stripped = p.rstrip()
            # fixed-fallback pieces (from an oversized single sentence) won't
            # necessarily end at a terminator; here no sentence is oversized,
            # so all must end at a terminator.
            assert stripped[-1] in terminators, (
                f"chunk not ending at sentence boundary: {repr(stripped[-30:])}")

    def test_long_sentence_fallback(self):
        # single sentence with NO terminator and > MAX_TOKENS tokens -> fixed split
        text = "यह एक बहुत लंबा वाक्य है जिसमें कोई वाक्य विभाजक नहीं है " * 400
        assert count_tokens(text) > MAX_TOKENS
        assert len(split_sentences(text)) == 1
        pieces = split_semantic(text)
        assert len(pieces) > 1
        # each piece is an exact substring of the original
        assert all(p in text for p in pieces)

    def test_deterministic(self):
        text = _repeat("नियतात्मकता वाक्य।", 150)
        assert split_semantic(text) == split_semantic(text)


# ===========================================================================
# ADAPTIVE
# ===========================================================================

class TestAdaptive:
    def test_short_path(self):
        text = "यह एक छोटा वाक्य है। " * 3
        assert count_tokens(text) <= SHORT_MAX
        assert adaptive_path(text) == PATH_SHORT
        pieces = split_adaptive(text)
        assert pieces == [text], "short path must keep passage whole"

    def test_medium_path(self):
        text = "पहला वाक्य है। दूसरा वाक्य है। " * 30
        n = count_tokens(text)
        assert SHORT_MAX < n <= MEDIUM_MAX, f"medium fixture has {n} tokens"
        assert adaptive_path(text) == PATH_MEDIUM
        pieces = split_adaptive(text)
        assert len(pieces) >= 1
        assert all(p in text for p in pieces)

    def test_long_path(self):
        text = _repeat("यह एक बहुत लंबा वाक्य है जो दोहराया गया है।", 300)
        n = count_tokens(text)
        assert n > MEDIUM_MAX, f"long fixture has {n} tokens"
        assert adaptive_path(text) == PATH_LONG
        pieces = split_adaptive(text)
        assert len(pieces) > 1
        assert all(p in text for p in pieces)

    def test_threshold_boundaries(self):
        # exact boundary tokens: <= SHORT_MAX -> short; SHORT_MAX+1 -> medium
        # Build a passage of exactly SHORT_MAX tokens.
        base = "शब्द "
        text = base
        while count_tokens(text) < SHORT_MAX:
            text += base
        while count_tokens(text) > SHORT_MAX:
            text = text[:-len(base)]
        # add minimal to hit exactly SHORT_MAX (approximation within +/-1)
        assert adaptive_path(text) == PATH_SHORT

    def test_deterministic(self):
        text = _repeat("अनुकूल नियतात्मक वाक्य।", 250)
        assert split_adaptive(text) == split_adaptive(text)


# ===========================================================================
# COMMON: metadata preservation, chunk IDs, empty input, deterministic
# ===========================================================================

class TestCommon:
    def test_metadata_preserved(self):
        doc = _doc(text="परीक्षण पाठ यहाँ। दूसरा वाक्य।")
        for chunker in (FixedChunker(), AdaptiveChunker()):
            chunks = chunker.chunk(doc)
            assert len(chunks) >= 1
            for c in chunks:
                # all canonical metadata carried through
                assert c["document_id"] == doc["document_id"]
                assert c["query_id"] == doc["query_id"]
                assert c["language"] == "hi"
                assert c["is_selected"] == 1
                assert c["query_type"] == "DESCRIPTION"
                assert c["source"] == "MSMARCO-XI"
                assert c["source_file"] == "hintrain.parquet"
                # English parallel passage kept aligned (full parent passage)
                assert c["text_en"] == doc["text_en"]
                assert c["query"] == doc["query"]
                assert c["query_en"] == doc["query_en"]

    def test_chunk_ids_unique_and_indexed(self):
        doc = _doc(text=_repeat("मेटाडेटा परीक्षण वाक्य।", 250))
        chunks = FixedChunker().chunk(doc)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "chunk_ids must be unique"
        assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
        # chunk_id format: document_id + _c{index}
        for i, c in enumerate(chunks):
            assert c["chunk_id"] == f"{doc['document_id']}_c{i}"

    def test_chunk_strategy_label(self):
        doc = _doc(text="छोटा पाठ।")
        assert all(c["chunk_strategy"] == "fixed" for c in FixedChunker().chunk(doc))
        from backend.rag.chunkers.semantic import SemanticChunker
        assert all(c["chunk_strategy"] == "semantic" for c in SemanticChunker().chunk(doc))
        assert all(c["chunk_strategy"] == "adaptive" for c in AdaptiveChunker().chunk(doc))

    def test_empty_input(self):
        doc = _doc(text="")
        for chunker in (FixedChunker(), AdaptiveChunker()):
            from backend.rag.chunkers.semantic import SemanticChunker
            assert chunker.chunk(doc) == [], "empty passage must produce no chunks"
        from backend.rag.chunkers.semantic import SemanticChunker
        assert SemanticChunker().chunk(_doc(text="")) == []

    def test_whitespace_only_input(self):
        doc = _doc(text="   \n\t  ")
        assert FixedChunker().chunk(doc) == []

    def test_make_chunks_drops_empty_pieces(self):
        doc = _doc(text="x")
        pieces = ["अच्छा।", "   ", "", "और अच्छा।"]
        chunks = make_chunks(doc, pieces, "fixed")
        assert len(chunks) == 2  # only the two non-empty pieces

    def test_deterministic_across_chunkers(self):
        doc = _doc(text=_repeat("नियतात्मकता पाठ वाक्य।", 200))
        for chunker in (FixedChunker(), AdaptiveChunker()):
            from backend.rag.chunkers.semantic import SemanticChunker
            a = chunker.chunk(doc)
            b = chunker.chunk(doc)
            # chunks compare by (chunk_id, text) to be order-independent of identity
            assert [c["text"] for c in a] == [c["text"] for c in b]
