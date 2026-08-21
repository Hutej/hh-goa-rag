"""Phase 8: speech-to-text via Sarvam.

audio -> Sarvam STT -> transcribed text (JSON). The text can be piped directly
into the existing retrieve_hybrid.py / answer_query.py (no retrieval/generation
duplication).

Usage:
    venv/bin/python scripts/transcribe.py --audio path/to/audio.wav
    venv/bin/python scripts/transcribe.py --audio audio.wav --language hi-IN
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.rag.stt import (  # noqa: E402
    DEFAULT_LANGUAGE, STTError, get_provider, transcribe,
)

_log = lambda *m: print(*m, file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", required=True, help="path to audio file")
    ap.add_argument("--language", default=DEFAULT_LANGUAGE,
                    help="language_code, default hi-IN")
    args = ap.parse_args()

    _log("transcribing via Sarvam STT...")
    try:
        result = transcribe(args.audio, language=args.language)
    except STTError as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False), flush=True)
        sys.exit(1)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
