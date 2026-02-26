# Qwen TTS Local API

FastAPI wrapper around local `Qwen/Qwen3-TTS-0.6B` inference.

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
- `DISABLE_API_KEY_AUTH=true` to disable Authorization checks in local dev
