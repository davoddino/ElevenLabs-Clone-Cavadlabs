# Qwen TTS Local API

FastAPI wrapper around local `Qwen/Qwen3-TTS-0.6B` inference.

## Quick start (no Docker)

1. Create `qwen-tts/.env`:

```env
DISABLE_API_KEY_AUTH=true
STORAGE_BACKEND=local
LOCAL_STORAGE_ROOT=/absolute/path/to/local-storage
QWEN_TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Instruct
QWEN_TTS_PORT=8003
```

2. Start the API:

```bash
python main.py
```

`api.py` auto-loads `qwen-tts/.env`. You can still override values from shell env vars.

## Endpoints

- `POST /generate`
  - body: `{ "text": "...", "target_voice": "Cherry" }`
  - returns: `{ "audio_url": "...", "s3_key": "..." }`
- `GET /voices`
- `GET /health`

## Required env vars

- `STORAGE_BACKEND` (`local` or `s3`)
- `API_KEY` (optional if `DISABLE_API_KEY_AUTH=true`)

When `STORAGE_BACKEND=local`:
- `LOCAL_STORAGE_ROOT` (default `/data/storage`)

When `STORAGE_BACKEND=s3`:
- `S3_BUCKET`
- `AWS_REGION`
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (or IAM role)

## Optional env vars

- `QWEN_TTS_MODEL_ID` (default `Qwen/Qwen3-TTS-0.6B`)
- `QWEN_TTS_VOICES` (comma-separated, default `Cherry,Chelsie,Ethan,Serena,Dylan,Jada`)
- `QWEN_TTS_MAX_TEXT_LENGTH` (default `3000`)
- `QWEN_TTS_MAX_NEW_TOKENS` (default `2048`)
- `QWEN_TTS_SAMPLE_RATE` (default `24000`)
- `QWEN_TTS_HOST` (default `0.0.0.0`)
- `QWEN_TTS_PORT` (default `8003`)
- `DISABLE_API_KEY_AUTH=true` to disable Authorization checks in local dev

## Model notes

- If `QWEN_TTS_MODEL_ID` ends with `-Base`, the API will automatically try an `-Instruct` fallback.
- If you see `missing spk_dict.pt`, the selected checkpoint is not compatible with this runtime.
