# ElevenLabs Clone Frontend

Next.js app used for UI, API routes, and Inngest job execution.

## Local startup

### 1. Install dependencies

```bash
npm install
```

### 2. Configure `.env`

```bash
cp .env.example .env
```

Minimum required values:

```env
AUTH_SECRET=replace-with-npx-auth-secret
DATABASE_URL=file:./db.sqlite
STORAGE_BACKEND=local
LOCAL_STORAGE_ROOT=/absolute/path/to/ElevenLabs-Clone-Cavadlabs/local-storage
QWEN_TTS_API_ROUTE=http://localhost:8003
TTS_DEFAULT_SERVICE=qwen-tts
BACKEND_API_KEY=
```

Important:

- `LOCAL_STORAGE_ROOT` must match the same absolute path used by `qwen-tts`.
- If backend auth is enabled, set `BACKEND_API_KEY` to the same value expected by backend.

### 3. Run frontend

```bash
npm run dev
```

### 4. Run Inngest worker (separate terminal)

```bash
npm run inngest-dev
```

Without this command, TTS generation jobs stay pending or fail in background flow.

## Useful commands

```bash
npm run typecheck
npm run lint
npm run db:push
npm run db:studio
```

## Known warnings in dev

- `GET /x/inngest 404`, `/.netlify/functions/inngest 404`, `/.redwood/functions/inngest 404` are expected during Inngest discovery in local dev.
- Cross-origin warning for `allowedDevOrigins` is a Next.js dev warning and does not block generation.
