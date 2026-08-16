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
dependencies = ["yt-dlp>=2025.1.0"]
[tool.uv]
package = false
EOF

uv sync
```

## Download and convert

```bash
export PLAYLIST_URL='https://www.youtube.com/playlist?list=REPLACE_ME'
export YTDLP_CONCURRENT_FRAGMENTS=8
uv run python download_playlist.py "$PLAYLIST_URL"
uv run python convert_audio.py
```

Increase `YTDLP_CONCURRENT_FRAGMENTS` only when network and disk can sustain it. The audio conversion script uses FFmpeg's available CPU threads. For multiple independent files, add a bounded process pool rather than relying on a single serial loop.

## WSL 2 + Qwen3-ASR vLLM transcription

Use a separate Python 3.12 environment for Qwen/vLLM. Do not add vLLM or a
second PyTorch build to the media-download environment above: vLLM is compiled
against a specific Torch/CUDA stack.

The following pairing was verified on 2026-08-15 with WSL 2 and an RTX 5070 Ti
(16 GB): `qwen-asr==0.0.6`, `vllm==0.14.0`, `transformers==4.57.6`, and
CUDA 12.8. Treat the pin as a known-good baseline; rerun the preflight and
canary before changing it.

```bash
cd playlist_extraction
uv venv --python 3.12 .venv-qwen-asr-vllm
uv pip install --python .venv-qwen-asr-vllm/bin/python \
  'qwen-asr[vllm]==0.0.6' httpx

.venv-qwen-asr-vllm/bin/python -c "import vllm, torch; from qwen_asr.cli.serve import main; print(vllm.__version__); print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))"
nvidia-smi
```

Run vLLM processes, caches, logs, and IPC sockets on the Linux filesystem.
Source media may remain under `/mnt/c`, but vLLM's IPC path must be somewhere
such as `/tmp`, not a Windows-mounted directory. Start with two in-flight
requests on a 16 GB GPU; tune upward only after the canary succeeds.

```bash
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/tmp
export VLLM_RPC_BASE_PATH=/tmp
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=64
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_V2_MODEL_RUNNER=0

.venv-qwen-asr-vllm/bin/qwen-asr-serve Qwen/Qwen3-ASR-1.7B \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 8192 \
  --enforce-eager \
  --max-num-seqs 2
```

`qwen-asr-serve` registers the Qwen ASR model before invoking vLLM. Do not use
the realtime API for a playlist batch; use OpenAI-compatible
`POST /v1/audio/transcriptions`.

### Required readiness and one-file canary

Wait for the server to report its model, then transcribe one representative MP3
before submitting the playlist. The API accepts ISO language codes, so use
`zh`, not `Chinese`.

```bash
curl --fail http://127.0.0.1:8000/v1/models

curl --fail -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F 'model=Qwen/Qwen3-ASR-1.7B' \
  -F 'language=zh' \
  -F 'file=@audio_mp3/001 - example.mp3'
```

The server default is version-dependent and may reject large multipart uploads.
Set `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` above the largest MP3. For files larger
than the verified upload limit, split audio before transcription and retain a
manifest that records the segment order.

### Bounded asynchronous batch

Run the embedded `transcribe_vllm_async.py` client only after the readiness and
canary checks succeed:

```bash
export VLLM_BASE_URL='http://127.0.0.1:8000'
.venv-qwen-asr-vllm/bin/python transcribe_vllm_async.py --concurrency 2
```

The client writes each successful transcript atomically, records failures in a
JSONL file, and skips existing non-empty transcripts on rerun. Do not load a
Transformers fallback in the same process or GPU while vLLM is serving. If the
vLLM canary fails, stop vLLM and run an explicitly separate Transformers
recovery pass.

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

### `transcribe_vllm_async.py`

```python
import argparse
import asyncio
import json
import os
from pathlib import Path

ROOT = Path(os.getenv("PLAYLIST_ROOT", Path.cwd())).expanduser().resolve()
AUDIOS = ROOT / "audio_mp3"
OUT = ROOT / "transcripts" / "qwen3-asr"
FAILURES = OUT / "failures.jsonl"

async def transcribe_one(client, audio, output, args, semaphore):
    if output.exists() and output.stat().st_size:
        print(f"Skipping {audio.name}", flush=True)
        return

    async with semaphore:
        try:
            with audio.open("rb") as handle:
                response = await client.post(
                    f"{args.base_url}/v1/audio/transcriptions",
                    data={"model": args.model, "language": args.language},
                    files={"file": (audio.name, handle, "audio/mpeg")},
                )
            response.raise_for_status()
            temporary = output.with_suffix(".txt.part")
            temporary.write_text(response.json()["text"].strip() + "\n", encoding="utf-8")
            temporary.replace(output)
            print(f"Wrote {output}", flush=True)
        except Exception as exc:
            with FAILURES.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"audio": audio.name, "error": str(exc)}, ensure_ascii=False) + "\n")
            print(f"Failed {audio.name}: {exc}", flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000"))
parser.add_argument("--model", default=os.getenv("VLLM_MODEL", "Qwen/Qwen3-ASR-1.7B"))
parser.add_argument("--language", default="zh")
parser.add_argument("--concurrency", type=int, default=2)
args = parser.parse_args()
OUT.mkdir(parents=True, exist_ok=True)

if args.concurrency < 1:
    raise SystemExit("--concurrency must be at least 1")

import httpx
timeout = httpx.Timeout(connect=30, read=None, write=120, pool=None)
async def main():
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=timeout) as client:
        await asyncio.gather(*(
            transcribe_one(client, audio, OUT / f"{audio.stem}.txt", args, semaphore)
            for audio in sorted(AUDIOS.glob("*.mp3"))
        ))

asyncio.run(main())
```

## Validation

```bash
find playlist_mp4 -type f -name '*.mp4' | wc -l
find audio_mp3 -type f -name '*.mp3' | wc -l
find transcripts/qwen3-asr -type f -name '*.txt' -size +1c | wc -l
find transcripts/qwen3-asr -type f -name '*.part' | wc -l
```

Repair only missing or empty outputs and rerun the relevant stage. Do not delete the full tree unless the source playlist changed.
