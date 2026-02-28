# ElevenLabs Clone

Practical local development guide for this monorepo.

## Services in this repo

- `elevenlabs-clone-frontend`: Next.js app (UI + API routes + Inngest functions)
- `qwen-tts`: local Qwen3-TTS FastAPI backend
- `StyleTTS2`, `seed-vc`, `Make-An-Audio`: optional extra backends

## Quick Start (local, no Docker)

Use 3 terminals.

### 1. Start `qwen-tts` API (terminal 1)

```bash
cd qwen-tts
python -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

Minimal `qwen-tts/.env`:

```env
DISABLE_API_KEY_AUTH=true
STORAGE_BACKEND=local
LOCAL_STORAGE_ROOT=/absolute/path/to/ElevenLabs-Clone-Cavadlabs/local-storage
QWEN_TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
QWEN_TTS_TOKENIZER_ID=Qwen/Qwen3-TTS-Tokenizer-12Hz
QWEN_TTS_MODEL_MODE=custom_voice
QWEN_TTS_HOST=0.0.0.0
QWEN_TTS_PORT=8003
```

Notes:

- First boot downloads model weights from Hugging Face.
- `QWEN_TTS_MODEL_MODE=base` requires reference audio (`ref_audio`/`ref_text`) by design.
- If you do not want voice cloning, use `custom_voice` or `voice_design`.

### 2. Start frontend (terminal 2)

```bash
cd elevenlabs-clone-frontend
npm install
cp .env.example .env
npm run dev
```

Minimal `elevenlabs-clone-frontend/.env`:

```env
AUTH_SECRET=replace-with-npx-auth-secret
DATABASE_URL=file:./db.sqlite
STORAGE_BACKEND=local
LOCAL_STORAGE_ROOT=/absolute/path/to/ElevenLabs-Clone-Cavadlabs/local-storage
QWEN_TTS_API_ROUTE=http://localhost:8003
TTS_DEFAULT_SERVICE=qwen-tts
BACKEND_API_KEY=
```

Critical rule:

- `LOCAL_STORAGE_ROOT` must be the same absolute path in frontend and `qwen-tts`.

### 3. Start Inngest dev server (terminal 3)

```bash
cd elevenlabs-clone-frontend
npm run inngest-dev
```

If this process is not running, generation jobs are created but not processed.

### 4. Health checks

```bash
curl -s http://localhost:8003/health
curl -s http://localhost:8003/voices
```

Frontend should run on `http://localhost:3000`.

## Docker (qwen-tts only)

```bash
docker compose build --no-cache qwen-tts-api
docker compose up -d qwen-tts-api
curl -s http://localhost:8003/health
```

`docker-compose.yml` maps host `8003` to container `8000` for `qwen-tts-api`.

## Troubleshooting

### `400 Base model requires a reference voice`

You are using a Base model without `ref_audio`/`ref_text`. Switch to `CustomVoice`/`VoiceDesign`, or configure reference voice env vars for Base mode.

### `400 Target voice not supported`

The voice sent by frontend is not in the model speaker list. Check `GET /voices` and choose one of those names.

### `404 /api/storage/...`

Frontend is looking in a different local folder than the backend wrote to. Align `LOCAL_STORAGE_ROOT` exactly in both `.env` files.

### `Inngest API Error: 401 Event key not found` or stuck jobs

Run `npm run inngest-dev` in `elevenlabs-clone-frontend`.
