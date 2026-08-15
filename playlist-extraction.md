# General playlist extraction skill

Use this workflow when converting an authorized playlist into a matching media and transcript tree. The workflow is portable across machines: never assume a particular Windows username, drive letter, or `/mnt/...` mount. Use the directory where the user runs the commands, or an explicit `PLAYLIST_ROOT`.

## Output layout

```text
project/
├── playlist_mp4/
├── audio_mp3/
└── transcripts/qwen3-asr/
```

Keep the playlist index in every filename so MP4, MP3, and transcript files match.

## Operating rules

- Only process media the user is authorized to download and retain.
- Download all MP4 files and convert all MP3 files before starting GPU transcription.
- Make every stage resumable: skip existing non-empty outputs.
- If Python is used for multiple files, use bounded `multiprocessing` or `concurrent.futures`; never spawn an unbounded process per file.
- If Python is not used, use native tool parallelism such as `yt-dlp --concurrent-fragments N` and `ffmpeg -threads 0`.
- For vLLM, keep one server/model loaded and send bounded concurrent requests. Tune server batching instead of launching one model per file.

## RTX 30/40/50-series notes

The comments below are guidance, not hard-coded device requirements:

```text
# RTX 30-series: use a current CUDA-enabled PyTorch build; reduce concurrency if VRAM is 8 GB.
# RTX 40-series: use a current CUDA-enabled PyTorch build; FP16 is a practical default.
# RTX 50-series: use a recent NVIDIA Windows driver and a current PyTorch wheel with support
#               for the selected CUDA runtime; prefer the official PyTorch selector if cu128
#               is not the best available option for the installed release.
# Any RTX generation: verify WSL GPU access with nvidia-smi and then torch.cuda.is_available().
# Any VRAM size: tune batch/request concurrency from actual free VRAM, not GPU name alone.
```

Do not install a separate Linux display driver inside WSL. Use WSL 2 with the Windows NVIDIA driver, then install the CUDA-enabled PyTorch wheel inside the `uv` environment.

## One-time setup in any directory

```bash
mkdir -p playlist_extraction
cd playlist_extraction
sudo apt update && sudo apt install -y ffmpeg curl
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

cat > pyproject.toml <<'EOF'
[project]
name = "playlist-extraction"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["yt-dlp>=2025.1.0", "httpx>=0.27.0", "qwen-asr>=0.0.6"]
[tool.uv]
package = false
EOF

uv sync
# Use the official PyTorch selector if a newer CUDA wheel is recommended.
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Download and convert

```bash
export PLAYLIST_URL='https://www.youtube.com/playlist?list=REPLACE_ME'
export YTDLP_CONCURRENT_FRAGMENTS=8
uv run python download_playlist.py "$PLAYLIST_URL"
uv run python convert_audio.py
```

Increase `YTDLP_CONCURRENT_FRAGMENTS` only when network and disk can sustain it. The audio conversion script uses FFmpeg's available CPU threads. For multiple independent files, add a bounded process pool rather than relying on a single serial loop.

## Transcription throughput

Start vLLM separately with the model's supported audio-serving command. Keep the model resident and tune the version-appropriate batching options, commonly `--max-num-seqs` and `--max-num-batched-tokens`. Start conservatively, then raise concurrency while watching VRAM, latency, and error rate.

```bash
export PLAYLIST_ROOT="$PWD"
export VLLM_BASE_URL='http://127.0.0.1:8000'
export VLLM_MODEL='Qwen/Qwen3-ASR-1.7B'
uv run python transcribe.py --backend auto --language Chinese
```

`auto` uses vLLM when `VLLM_BASE_URL` exists and falls back to local Transformers/Qwen inference when a request fails. Use `--backend transformers` to bypass vLLM. Keep Transformers workers bounded because each worker can consume substantial VRAM.

## Embedded files

Save each block with the shown filename in the same working directory as this Markdown file.

### `download_playlist.py`

```python
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(os.getenv("PLAYLIST_ROOT", Path.cwd())).expanduser().resolve()
OUT = ROOT / "playlist_mp4"
OUT.mkdir(parents=True, exist_ok=True)

if len(sys.argv) != 2:
    raise SystemExit("Usage: uv run python download_playlist.py PLAYLIST_URL")

subprocess.run([
    "yt-dlp", "--yes-playlist", "--ignore-errors", "--continue",
    "--no-overwrites", "--write-info-json",
    "--concurrent-fragments", os.getenv("YTDLP_CONCURRENT_FRAGMENTS", "8"),
    "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
    "--merge-output-format", "mp4",
    "-o", str(OUT / "%(playlist_index)03d - %(title)s.%(ext)s"),
    sys.argv[1],
], check=True)
```

### `convert_audio.py`

```python
from pathlib import Path
import os
import subprocess

ROOT = Path(os.getenv("PLAYLIST_ROOT", Path.cwd())).expanduser().resolve()
VIDEO_DIR = ROOT / "playlist_mp4"
AUDIO_DIR = ROOT / "audio_mp3"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

for video in sorted(VIDEO_DIR.glob("*.mp4")):
    output = AUDIO_DIR / f"{video.stem}.mp3"
    if output.exists() and output.stat().st_size:
        continue
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-threads", "0", "-y",
        "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
        "-codec:a", "libmp3lame", "-q:a", "2", str(output),
    ], check=True)
```

### `transcribe.py`

```python
import argparse
import os
from pathlib import Path

ROOT = Path(os.getenv("PLAYLIST_ROOT", Path.cwd())).expanduser().resolve()
AUDIOS = ROOT / "audio_mp3"
OUT = ROOT / "transcripts" / "qwen3-asr"

def transformers_text(audio: Path, language: str) -> str:
    from qwen_asr import Qwen3ASRModel
    model = Qwen3ASRModel.from_pretrained(
        os.getenv("ASR_MODEL", "Qwen/Qwen3-ASR-1.7B"),
        device_map=os.getenv("ASR_DEVICE", "auto"),
        dtype=os.getenv("ASR_DTYPE", "float16"),
        max_inference_batch_size=1,
        max_new_tokens=4096,
    )
    result = model.transcribe(str(audio), language=language)[0]
    return getattr(result, "text", str(result)).strip()

def vllm_text(audio: Path, language: str) -> str:
    import httpx
    base = os.environ["VLLM_BASE_URL"].rstrip("/")
    data = {"model": os.getenv("VLLM_MODEL", "Qwen/Qwen3-ASR-1.7B"), "language": language}
    headers = {"Authorization": f"Bearer {os.getenv('VLLM_API_KEY', 'EMPTY')}"}
    with audio.open("rb") as handle:
        response = httpx.post(
            f"{base}/v1/audio/transcriptions", data=data,
            files={"file": (audio.name, handle, "audio/mpeg")},
            headers=headers, timeout=None,
        )
    response.raise_for_status()
    return response.json()["text"].strip()

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["auto", "vllm", "transformers"], default="auto")
parser.add_argument("--language", default="Chinese")
args = parser.parse_args()
OUT.mkdir(parents=True, exist_ok=True)

for audio in sorted(AUDIOS.glob("*.mp3")):
    output = OUT / f"{audio.stem}.txt"
    if output.exists() and output.stat().st_size:
        continue
    try:
        use_vllm = args.backend == "vllm" or (args.backend == "auto" and os.getenv("VLLM_BASE_URL"))
        text = vllm_text(audio, args.language) if use_vllm else transformers_text(audio, args.language)
    except Exception as exc:
        if args.backend == "auto" and os.getenv("VLLM_BASE_URL"):
            print(f"vLLM failed for {audio.name}: {exc}; using Transformers", flush=True)
            text = transformers_text(audio, args.language)
        else:
            raise
    output.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote {output}", flush=True)
```

## Validation

```bash
find playlist_mp4 -type f -name '*.mp4' | wc -l
find audio_mp3 -type f -name '*.mp3' | wc -l
find transcripts/qwen3-asr -type f -name '*.txt' | wc -l
```

Repair only missing or empty outputs and rerun the relevant stage. Do not delete the full tree unless the source playlist changed.
