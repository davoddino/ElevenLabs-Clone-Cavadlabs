# Qwen TTS Local API

FastAPI wrapper around local Qwen3-TTS inference using the official `qwen_tts.Qwen3TTSModel` runtime.

## Quick Start (No Docker)

1. Create `qwen-tts/.env` from the example.

```bash
cp .env.example .env
```

2. Edit `.env` and set at least:

```env
DISABLE_API_KEY_AUTH=true
STORAGE_BACKEND=local
LOCAL_STORAGE_ROOT=/absolute/path/to/local-storage
QWEN_TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
QWEN_TTS_MODEL_MODE=custom_voice
QWEN_TTS_PORT=8003
```

3. Start the API:

```bash
python main.py
```

`api.py` auto-loads `qwen-tts/.env`, so no `export` is required.
The service uses Hugging Face default cache paths automatically (no manual HF path config needed).

## Endpoints

- `POST /generate`
- `GET /voices`
- `GET /health`

### `POST /generate` request body

```json
{
  "text": "Hello world",
  "target_voice": "Cherry",
  "language": "Auto",
  "instruct": "Warm and expressive",
  "ref_audio": "/path/to/ref.wav",
  "ref_text": "Reference transcript"
}
```

Fields used depend on model mode:
- `custom_voice`: uses `target_voice` (speaker id)
- `voice_design`: uses `instruct` (or falls back to `target_voice` as description)
- `base`: uses `ref_audio/ref_text` or preset/default reference from env

If `base` is selected and no reference voice is configured, requests fail with `400` by design.

## Model Modes

The API auto-detects mode from `QWEN_TTS_MODEL_ID` and can be overridden with `QWEN_TTS_MODEL_MODE`.

- `...CustomVoice` -> `custom_voice`
- `...VoiceDesign` -> `voice_design`
- `...Base` -> `base`

## Tokenizer Auto-Fix

On startup, the API now:
- downloads the selected model snapshot
- downloads `Qwen/Qwen3-TTS-Tokenizer-12Hz` (or `QWEN_TTS_TOKENIZER_ID`)
- ensures `<model_path>/speech_tokenizer` points to a tokenizer folder containing `preprocessor_config.json`

This prevents the common Base-model crash:
`Can't load feature extractor .../speech_tokenizer ... preprocessor_config.json`.

Example released IDs:
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign`

## Base Model Voice Mapping

For frontend compatibility (`target_voice` like `Cherry`), configure one or both:

- Global default reference:
  - `QWEN_TTS_BASE_REF_AUDIO`
  - `QWEN_TTS_BASE_REF_TEXT`
- Per-voice presets:
  - `QWEN_TTS_VOICE_CLONE_PRESETS` as JSON map

Example:

```env
QWEN_TTS_VOICE_CLONE_PRESETS={"Cherry":{"ref_audio":"/path/cherry.wav","ref_text":"Hello from Cherry"},"Chelsie":{"ref_audio":"/path/chelsie.wav","ref_text":"Hello from Chelsie"}}
```

## Key Env Vars

- `QWEN_TTS_MODEL_ID`
- `QWEN_TTS_MODEL_MODE` (`base`, `custom_voice`, `voice_design`)
- `QWEN_TTS_MAX_TEXT_LENGTH` (default `3000`)
- `QWEN_TTS_MAX_NEW_TOKENS` (default `2048`)
- `QWEN_TTS_LANGUAGE` (default `Auto`)
- `QWEN_TTS_HOST` (default `0.0.0.0`)
- `QWEN_TTS_PORT` (default `8003`)
- `QWEN_TTS_DEVICE_MAP` (default `cuda:0` when CUDA is available)
- `QWEN_TTS_ATTN_IMPLEMENTATION` (optional)
- `QWEN_TTS_X_VECTOR_ONLY_MODE` (Base mode only)
- `QWEN_TTS_TOKENIZER_ID` (default `Qwen/Qwen3-TTS-Tokenizer-12Hz`)

## Storage

- `STORAGE_BACKEND=local` -> saves `.wav` under `LOCAL_STORAGE_ROOT/qwen-tts-output`
- `STORAGE_BACKEND=s3` -> uploads to `S3_BUCKET` with prefix `S3_PREFIX`
