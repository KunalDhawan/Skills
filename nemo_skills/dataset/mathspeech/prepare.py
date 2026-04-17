# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare the MathSpeech benchmark for NeMo Skills evaluation.

Downloads the MathSpeech dataset from HuggingFace (AAAI2025/MathSpeech, ~107 MB,
1,101 samples) and emits two per-mode ``test.jsonl`` files that share the same
audio:

- ``asr/test.jsonl``    \u2014 prompt asks for spoken English transcription; scored with WER.
- ``latex/test.jsonl``  \u2014 prompt asks for LaTeX formula; scored with CER/BLEU/ROUGE.

The paper uses all 1,101 samples as the evaluation set (training data for
MathSpeech's pipeline is a separately scraped SE\u2192LaTeX corpus, not this data).

Dataset: https://huggingface.co/datasets/AAAI2025/MathSpeech
Paper:   https://arxiv.org/abs/2412.15655

Usage:
    # Auto-download to a ``data/`` subdirectory next to this script.
    ns prepare_data mathspeech

    # Use pre-downloaded data
    ns prepare_data mathspeech --data_dir=/path/to/mathspeech-data

    # Skip saving .wav files (JSONL still includes audio paths \u2014 not recommended).
    ns prepare_data mathspeech --no-audio
"""

import argparse
import json
from pathlib import Path

HF_REPO_ID = "AAAI2025/MathSpeech"
HF_SPLIT = "train"  # HF dataset only exposes 'train'; paper evaluates all 1,101 rows.
AUDIO_SUBDIR = "audio"
MIN_AUDIO_DURATION = 0.1

PROMPT_ASR = "Transcribe the spoken mathematical expression in this audio into plain English text."
PROMPT_LATEX = (
    "Transcribe the spoken mathematical expression in this audio into a LaTeX formula. "
    "Output only the LaTeX formula between $ ... $ delimiters, with no additional commentary."
)


def _sample_id(idx, source):
    """Build a stable, filename-safe sample id from the row index and YouTube source URL."""
    tail = ""
    if isinstance(source, str) and "v=" in source:
        tail = source.split("v=", 1)[1].split("&", 1)[0]
    tail = "".join(c for c in tail if c.isalnum() or c in ("-", "_"))
    return f"mathspeech_{idx:04d}_{tail}" if tail else f"mathspeech_{idx:04d}"


def _decode_duration(audio_bytes):
    """Return audio duration in seconds from raw bytes, or None if undecodable."""
    import io

    import soundfile as sf

    try:
        with sf.SoundFile(io.BytesIO(audio_bytes)) as snd:
            if snd.samplerate <= 0:
                return None
            return len(snd) / snd.samplerate
    except Exception:
        return None


def download_and_save_audio(output_audio_dir, with_audio=True):
    """Stream the HF dataset and write per-sample MP3 files and a metadata JSONL.

    Audio arrives as raw mp3 bytes on the HF dataset, so we write them out directly
    without decode/re-encode. Returns the path to ``metadata.jsonl`` with one row per
    sample: ``{"id", "audio_filename", "duration", "transcription", "latex", "source"}``.
    Skips download if the manifest already exists and all audio files are present.
    """
    from datasets import Audio, load_dataset

    output_audio_dir = Path(output_audio_dir)
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_audio_dir.parent / "metadata.jsonl"

    if manifest_path.exists():
        manifest_rows = [json.loads(line) for line in manifest_path.open(encoding="utf-8") if line.strip()]
        if not with_audio or all((output_audio_dir / row["audio_filename"]).exists() for row in manifest_rows):
            print(f"Dataset already prepared at {output_audio_dir.parent} ({len(manifest_rows)} rows). Skipping download.")
            return manifest_path

    print("=" * 70)
    print(f"DOWNLOADING {HF_REPO_ID} from HuggingFace")
    print(f"Destination: {output_audio_dir.parent}")
    print("Total size: ~107 MB, 1,101 samples. Typically completes in a few minutes.")
    print("=" * 70)

    ds = load_dataset(HF_REPO_ID, split=HF_SPLIT).cast_column("audio", Audio(decode=False))

    rows = []
    skipped = 0
    for idx, entry in enumerate(ds):
        source = entry.get("Source", "")
        audio_info = entry.get("audio") or {}
        audio_bytes = audio_info.get("bytes")
        if not audio_bytes:
            skipped += 1
            continue

        duration = _decode_duration(audio_bytes)
        if duration is None or duration < MIN_AUDIO_DURATION:
            skipped += 1
            continue

        sid = _sample_id(idx, source)
        orig_name = audio_info.get("path") or ""
        suffix = Path(orig_name).suffix.lower() or ".mp3"
        audio_filename = f"{sid}{suffix}"
        if with_audio:
            (output_audio_dir / audio_filename).write_bytes(audio_bytes)

        rows.append(
            {
                "id": sid,
                "audio_filename": audio_filename,
                "duration": float(duration),
                "transcription": entry.get("transcription", "").strip(),
                "latex": entry.get("LaTeX", "").strip(),
                "source": source,
            }
        )

    with manifest_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    if skipped:
        print(f"Skipped {skipped} rows (missing audio or <{MIN_AUDIO_DURATION}s).")
    print(f"Wrote {len(rows)} samples to {manifest_path}")
    return manifest_path


def build_messages(prompt_text, audio_path, duration):
    """Build OpenAI-format messages with audio metadata."""
    return [
        {
            "role": "user",
            "content": prompt_text,
            "audio": {
                "path": audio_path,
                "duration": float(duration),
            },
        }
    ]


def write_mode_jsonl(rows, mode, output_path, audio_prefix):
    """Write a per-mode test.jsonl with audio paths prefixed by ``audio_prefix``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            audio_path = f"{audio_prefix.rstrip('/')}/{AUDIO_SUBDIR}/{row['audio_filename']}"
            if mode == "asr":
                prompt = PROMPT_ASR
                expected_answer = row["transcription"]
            elif mode == "latex":
                prompt = PROMPT_LATEX
                expected_answer = row["latex"]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if not expected_answer:
                continue

            record = {
                "messages": build_messages(prompt, audio_path, row["duration"]),
                "expected_answer": expected_answer,
                "transcription": row["transcription"],
                "latex": row["latex"],
                "source": row["source"],
                "id": row["id"],
                "duration": row["duration"],
                "audio_filepath": audio_path,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"  {mode}: wrote {count} samples to {output_path}")
    return count


def main():
    """Parse arguments, download data if needed, and write per-mode JSONL splits."""
    parser = argparse.ArgumentParser(description="Prepare MathSpeech for NeMo Skills")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Path to the MathSpeech dataset root. If the directory already contains "
            "metadata.jsonl and audio/, that data is used directly. If missing, the "
            "dataset is downloaded there. Defaults to a 'data/' subdirectory next to "
            "this script."
        ),
    )
    parser.add_argument(
        "--audio-prefix",
        type=str,
        default=None,
        help=(
            "Override the audio path prefix written into JSONL files. Defaults to "
            "data_dir. Useful for container mount points (e.g. /data/mathspeech)."
        ),
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip saving .flac files; JSONL still references audio paths (not recommended).",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else output_dir / "data"
    audio_dir = data_dir / AUDIO_SUBDIR
    manifest_path = data_dir / "metadata.jsonl"

    if manifest_path.exists():
        print(f"Using pre-downloaded manifest at {manifest_path}")
    elif args.no_audio:
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Cannot use --no-audio before data is "
            "downloaded. Run once without --no-audio first."
        )
    else:
        download_and_save_audio(audio_dir, with_audio=not args.no_audio)

    with manifest_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(rows)} samples from manifest.")

    audio_prefix = args.audio_prefix if args.audio_prefix else str(data_dir)
    if not args.no_audio:
        sample_wav = Path(audio_prefix) / AUDIO_SUBDIR / rows[0]["audio_filename"]
        if not sample_wav.exists():
            print(
                f"WARNING: Sample audio file not found at {sample_wav}. Audio paths may "
                "need adjustment via --audio-prefix."
            )

    modes = {
        "asr": output_dir / "asr" / "test.jsonl",
        "latex": output_dir / "latex" / "test.jsonl",
    }

    print("\nWriting per-mode JSONL splits...")
    total = 0
    for mode, path in modes.items():
        total += write_mode_jsonl(rows, mode, path, audio_prefix)

    print(f"\nDone. Wrote {total} records across {len(modes)} modes.")


if __name__ == "__main__":
    main()
