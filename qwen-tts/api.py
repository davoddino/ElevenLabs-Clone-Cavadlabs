import json
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import soundfile as sf
import torch
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.security import APIKeyHeader
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_local_env_file() -> None:
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def _as_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_voice_clone_presets(raw: str | None) -> dict[str, dict[str, str]]:
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        logger.warning("Invalid QWEN_TTS_VOICE_CLONE_PRESETS JSON: %s", error)
        return {}

    if not isinstance(parsed, dict):
        logger.warning("QWEN_TTS_VOICE_CLONE_PRESETS must be a JSON object")
        return {}

    presets: dict[str, dict[str, str]] = {}
    for key, value in parsed.items():
        if not isinstance(value, dict):
            continue

        ref_audio = value.get("ref_audio")
        ref_text = value.get("ref_text")
        language = value.get("language")

        if isinstance(ref_audio, str) and ref_audio.strip():
            preset = {"ref_audio": ref_audio.strip()}
            if isinstance(ref_text, str) and ref_text.strip():
                preset["ref_text"] = ref_text.strip()
            if isinstance(language, str) and language.strip():
                preset["language"] = language.strip()
            presets[str(key)] = preset

    return presets


def _detect_model_mode(model_id: str, override: str | None) -> str:
    if override:
        normalized = override.strip().lower()
        alias_map = {
            "base": "base",
            "voice_clone": "base",
            "clone": "base",
            "custom": "custom_voice",
            "customvoice": "custom_voice",
            "custom_voice": "custom_voice",
            "design": "voice_design",
            "voicedesign": "voice_design",
            "voice_design": "voice_design",
        }
        if normalized in alias_map:
            return alias_map[normalized]

    lower_id = model_id.lower()
    if "customvoice" in lower_id:
        return "custom_voice"
    if "voicedesign" in lower_id:
        return "voice_design"
    if "base" in lower_id:
        return "base"

    return "custom_voice"


def _sibling_model_id(model_id: str, target_mode: str) -> str | None:
    lower_id = model_id.lower()
    if lower_id.endswith("-base"):
        root = model_id[:-5]
    elif lower_id.endswith("-customvoice"):
        root = model_id[:-12]
    elif lower_id.endswith("-voicedesign"):
        root = model_id[:-12]
    else:
        return None

    if target_mode == "base":
        return f"{root}-Base"
    if target_mode == "custom_voice":
        return f"{root}-CustomVoice"
    if target_mode == "voice_design":
        return f"{root}-VoiceDesign"
    return None


def _resolve_model_path(model_id: str) -> str:
    local_candidate = Path(model_id)
    if local_candidate.exists():
        return str(local_candidate.resolve())

    # Uses Hugging Face default cache location unless HF_HOME/TRANSFORMERS_CACHE is set.
    return snapshot_download(repo_id=model_id)


def _ensure_speech_tokenizer(model_path: str, tokenizer_path: str) -> str:
    model_dir = Path(model_path)
    tokenizer_dir = Path(tokenizer_path)
    speech_tokenizer_dir = model_dir / "speech_tokenizer"
    required_config = speech_tokenizer_dir / "preprocessor_config.json"

    if speech_tokenizer_dir.is_symlink():
        if required_config.exists():
            return str(speech_tokenizer_dir)
        speech_tokenizer_dir.unlink()
    elif speech_tokenizer_dir.is_dir():
        if required_config.exists():
            return str(speech_tokenizer_dir)
        shutil.rmtree(speech_tokenizer_dir)
    elif speech_tokenizer_dir.exists():
        speech_tokenizer_dir.unlink()

    try:
        os.symlink(tokenizer_dir, speech_tokenizer_dir, target_is_directory=True)
    except OSError:
        shutil.copytree(tokenizer_dir, speech_tokenizer_dir)

    if not required_config.exists():
        raise RuntimeError(
            "speech_tokenizer is missing preprocessor_config.json after tokenizer sync"
        )

    return str(speech_tokenizer_dir)


def _load_qwen_model(model_id: str):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer_repo = os.getenv(
        "QWEN_TTS_TOKENIZER_ID", "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    )

    model_path = _resolve_model_path(model_id)
    tokenizer_path = snapshot_download(repo_id=tokenizer_repo)
    speech_tokenizer_path = _ensure_speech_tokenizer(model_path, tokenizer_path)

    logger.info("Resolved model path: %s", model_path)
    logger.info("Resolved tokenizer path: %s", tokenizer_path)
    logger.info("Using speech_tokenizer path: %s", speech_tokenizer_path)

    model_kwargs: dict[str, Any] = {"dtype": dtype, "trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = os.getenv("QWEN_TTS_DEVICE_MAP", "cuda:0")
        attn_implementation = os.getenv("QWEN_TTS_ATTN_IMPLEMENTATION")
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

    return Qwen3TTSModel.from_pretrained(model_path, **model_kwargs)


def _extract_first_audio(wavs: Any) -> np.ndarray:
    if isinstance(wavs, torch.Tensor):
        arr = wavs.detach().cpu().numpy()
        if arr.ndim > 1:
            arr = arr[0]
        return np.asarray(arr, dtype=np.float32)

    if isinstance(wavs, np.ndarray):
        arr = wavs
        if arr.ndim > 1:
            arr = arr[0]
        return np.asarray(arr, dtype=np.float32)

    if isinstance(wavs, (list, tuple)) and len(wavs) > 0:
        first = wavs[0]
        if isinstance(first, torch.Tensor):
            first = first.detach().cpu().numpy()
        return np.asarray(first, dtype=np.float32)

    raise RuntimeError("Unexpected audio output from Qwen3TTSModel")


def _parse_voices(raw: str | None) -> list[str]:
    if not raw:
        return []

    voices = [segment.strip() for segment in raw.split(",") if segment.strip()]
    return list(dict.fromkeys(voices))


_load_local_env_file()

API_KEY = os.getenv("API_KEY")
AUTH_DISABLED = os.getenv("DISABLE_API_KEY_AUTH", "false").lower() == "true"
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()
LOCAL_STORAGE_ROOT = os.getenv("LOCAL_STORAGE_ROOT", "/data/storage")
MODEL_ID = os.getenv("QWEN_TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
DEFAULT_LANGUAGE = os.getenv("QWEN_TTS_LANGUAGE", "Auto")
MAX_TEXT_LENGTH = int(os.getenv("QWEN_TTS_MAX_TEXT_LENGTH", "3000"))
MAX_NEW_TOKENS = int(os.getenv("QWEN_TTS_MAX_NEW_TOKENS", "2048"))

# Applies only to Base model generation.
X_VECTOR_ONLY_MODE = _as_bool(os.getenv("QWEN_TTS_X_VECTOR_ONLY_MODE"), False)
BASE_REF_AUDIO = os.getenv("QWEN_TTS_BASE_REF_AUDIO", "").strip()
BASE_REF_TEXT = os.getenv("QWEN_TTS_BASE_REF_TEXT", "").strip()
BASE_REF_LANGUAGE = os.getenv("QWEN_TTS_BASE_REF_LANGUAGE", DEFAULT_LANGUAGE)

# JSON map: {"VoiceName": {"ref_audio": "...", "ref_text": "...", "language": "English"}}
VOICE_CLONE_PRESETS = _parse_voice_clone_presets(
    os.getenv("QWEN_TTS_VOICE_CLONE_PRESETS")
)
AUTO_FALLBACK_FROM_BASE = _as_bool(
    os.getenv("QWEN_TTS_AUTO_FALLBACK_FROM_BASE"), True
)
ALLOW_UNKNOWN_SPEAKER_FALLBACK = _as_bool(
    os.getenv("QWEN_TTS_ALLOW_UNKNOWN_SPEAKER_FALLBACK"), True
)

MODEL_MODE = _detect_model_mode(MODEL_ID, os.getenv("QWEN_TTS_MODEL_MODE"))

DEFAULT_VOICES = ["Cherry", "Chelsie", "Ethan", "Serena", "Dylan", "Jada"]
ENV_VOICES = _parse_voices(os.getenv("QWEN_TTS_VOICES"))

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
qwen_model = None
loaded_model_id = None
loaded_model_mode = None
loaded_supported_speakers: list[str] = []


async def verify_api_key(authorization: str = Header(None)):
    if AUTH_DISABLED or not API_KEY:
        return "auth-disabled"

    if not authorization:
        logger.warning("No API key provided")
        raise HTTPException(status_code=401, detail="API key is missing")

    if authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
    else:
        token = authorization

    if token != API_KEY:
        logger.warning("Invalid API key provided")
        raise HTTPException(status_code=401, detail="Invalid API key")

    return token


def get_s3_client():
    client_kwargs = {"region_name": os.getenv("AWS_REGION", "us-east-1")}

    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        client_kwargs.update(
            {
                "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            }
        )

    return boto3.client("s3", **client_kwargs)


s3_client = get_s3_client() if STORAGE_BACKEND == "s3" else None

S3_PREFIX = os.getenv("S3_PREFIX", "qwen-tts-output")
S3_BUCKET = os.getenv("S3_BUCKET", "elevenlabs-clone")


def _resolve_supported_voices() -> list[str]:
    if loaded_model_mode == "custom_voice" and loaded_supported_speakers:
        return loaded_supported_speakers

    if loaded_model_mode == "base" and VOICE_CLONE_PRESETS:
        return list(VOICE_CLONE_PRESETS.keys())

    if ENV_VOICES:
        return ENV_VOICES

    if loaded_model_mode == "voice_design":
        return ["VoiceDesign"]

    return DEFAULT_VOICES


def _base_reference_configured() -> bool:
    return bool(BASE_REF_AUDIO) or bool(VOICE_CLONE_PRESETS)


def _resolve_base_prompt(request: "TextOnlyRequest") -> tuple[str, str | None, str]:
    language = request.language or BASE_REF_LANGUAGE or DEFAULT_LANGUAGE

    if request.ref_audio:
        ref_text = request.ref_text.strip() if request.ref_text else None
        if not ref_text and not X_VECTOR_ONLY_MODE:
            raise HTTPException(
                status_code=400,
                detail="ref_text is required for Base model unless QWEN_TTS_X_VECTOR_ONLY_MODE=true",
            )
        return request.ref_audio, ref_text, language

    if request.target_voice and request.target_voice in VOICE_CLONE_PRESETS:
        preset = VOICE_CLONE_PRESETS[request.target_voice]
        ref_audio = preset["ref_audio"]
        ref_text = preset.get("ref_text")
        preset_language = preset.get("language") or language

        if not ref_text and not X_VECTOR_ONLY_MODE:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Preset '{request.target_voice}' is missing ref_text. "
                    "Add ref_text or enable QWEN_TTS_X_VECTOR_ONLY_MODE=true"
                ),
            )

        return ref_audio, ref_text, preset_language

    if BASE_REF_AUDIO:
        ref_text = request.ref_text.strip() if request.ref_text else (BASE_REF_TEXT or None)
        if not ref_text and not X_VECTOR_ONLY_MODE:
            raise HTTPException(
                status_code=400,
                detail=(
                    "QWEN_TTS_BASE_REF_TEXT is required for Base model unless "
                    "QWEN_TTS_X_VECTOR_ONLY_MODE=true"
                ),
            )
        return BASE_REF_AUDIO, ref_text, language

    raise HTTPException(
        status_code=400,
        detail=(
            "Base model requires a reference voice. "
            "Provide ref_audio/ref_text in request, or configure "
            "QWEN_TTS_VOICE_CLONE_PRESETS / QWEN_TTS_BASE_REF_AUDIO(+REF_TEXT)."
        ),
    )


def _generate_audio(request: "TextOnlyRequest") -> tuple[np.ndarray, int]:
    language = request.language or DEFAULT_LANGUAGE

    if loaded_model_mode == "custom_voice":
        speaker = request.target_voice or (_resolve_supported_voices()[0] if _resolve_supported_voices() else None)
        if not speaker:
            raise HTTPException(status_code=400, detail="No speaker available for CustomVoice model")

        if loaded_supported_speakers and speaker not in loaded_supported_speakers:
            if ALLOW_UNKNOWN_SPEAKER_FALLBACK and loaded_supported_speakers:
                fallback_speaker = loaded_supported_speakers[0]
                logger.warning(
                    "Requested voice '%s' not found; falling back to '%s'",
                    speaker,
                    fallback_speaker,
                )
                speaker = fallback_speaker
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Target voice not supported for this model. Choose from: "
                        + ", ".join(loaded_supported_speakers)
                    ),
                )

        kwargs: dict[str, Any] = {
            "text": request.text,
            "language": language,
            "speaker": speaker,
            "max_new_tokens": MAX_NEW_TOKENS,
        }
        if request.instruct:
            kwargs["instruct"] = request.instruct

        wavs, sample_rate = qwen_model.generate_custom_voice(**kwargs)
        return _extract_first_audio(wavs), int(sample_rate)

    if loaded_model_mode == "voice_design":
        instruct = request.instruct or request.target_voice
        if not instruct:
            raise HTTPException(
                status_code=400,
                detail="VoiceDesign model requires 'instruct' (or target_voice as description)",
            )

        wavs, sample_rate = qwen_model.generate_voice_design(
            text=request.text,
            language=language,
            instruct=instruct,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        return _extract_first_audio(wavs), int(sample_rate)

    if loaded_model_mode == "base":
        ref_audio, ref_text, resolved_language = _resolve_base_prompt(request)
        kwargs = {
            "text": request.text,
            "language": resolved_language,
            "ref_audio": ref_audio,
            "max_new_tokens": MAX_NEW_TOKENS,
        }

        if ref_text:
            kwargs["ref_text"] = ref_text
        if X_VECTOR_ONLY_MODE:
            kwargs["x_vector_only_mode"] = True

        wavs, sample_rate = qwen_model.generate_voice_clone(**kwargs)
        return _extract_first_audio(wavs), int(sample_rate)

    raise HTTPException(status_code=500, detail=f"Unsupported model mode: {loaded_model_mode}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qwen_model, loaded_model_id, loaded_model_mode, loaded_supported_speakers

    selected_model_id = MODEL_ID
    selected_mode = MODEL_MODE

    if selected_mode == "base" and not _base_reference_configured():
        if AUTO_FALLBACK_FROM_BASE:
            fallback_model_id = _sibling_model_id(selected_model_id, "custom_voice")
            if fallback_model_id:
                logger.warning(
                    "Base model selected but no reference voice configured. "
                    "Falling back to CustomVoice model: %s",
                    fallback_model_id,
                )
                selected_model_id = fallback_model_id
                selected_mode = "custom_voice"
            else:
                logger.warning(
                    "Base model selected without reference voice. Could not derive a CustomVoice sibling model id."
                )
        else:
            logger.warning(
                "Base model selected without reference voice and auto fallback is disabled."
            )

    logger.info("Loading Qwen TTS model: %s", selected_model_id)
    logger.info("Detected model mode: %s", selected_mode)

    try:
        qwen_model = _load_qwen_model(selected_model_id)
        loaded_model_id = selected_model_id
        loaded_model_mode = selected_mode

        if loaded_model_mode == "custom_voice" and hasattr(qwen_model, "get_supported_speakers"):
            speakers = qwen_model.get_supported_speakers()
            if isinstance(speakers, (list, tuple)):
                loaded_supported_speakers = [str(item) for item in speakers]
                logger.info("Loaded %s supported speakers", len(loaded_supported_speakers))

        logger.info("Qwen TTS model loaded successfully")
    except Exception as error:
        logger.exception("Failed to load Qwen TTS model: %s", error)
        raise

    yield

    logger.info("Shutting down Qwen TTS API")


app = FastAPI(title="Qwen TTS API", lifespan=lifespan)


class TextOnlyRequest(BaseModel):
    text: str
    target_voice: str | None = None
    language: str | None = None
    instruct: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None


@app.post("/generate", dependencies=[Depends(verify_api_key)])
async def generate_speech(
    request: TextOnlyRequest,
    background_tasks: BackgroundTasks,
):
    if qwen_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if len(request.text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text length exceeds the limit of {MAX_TEXT_LENGTH} characters",
        )

    try:
        audio, sample_rate = _generate_audio(request)

        audio_id = str(uuid.uuid4())
        output_filename = f"{audio_id}.wav"
        local_path = f"/tmp/{output_filename}"
        sf.write(local_path, audio, samplerate=sample_rate)

        s3_key = f"{S3_PREFIX}/{output_filename}"
        presigned_url = ""

        if STORAGE_BACKEND == "s3":
            if not s3_client:
                raise RuntimeError("S3 client not initialized")

            s3_client.upload_file(local_path, S3_BUCKET, s3_key)
            presigned_url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": S3_BUCKET, "Key": s3_key},
                ExpiresIn=3600,
            )
        else:
            output_dir = os.path.join(LOCAL_STORAGE_ROOT, S3_PREFIX)
            os.makedirs(output_dir, exist_ok=True)
            final_path = os.path.join(output_dir, output_filename)
            shutil.copyfile(local_path, final_path)
            presigned_url = f"/api/storage/{s3_key}"

        background_tasks.add_task(os.remove, local_path)

        return {
            "audio_url": presigned_url,
            "s3_key": s3_key,
        }
    except HTTPException as error:
        logger.warning(
            "Request rejected: status=%s detail=%s target_voice=%s mode=%s",
            error.status_code,
            getattr(error, "detail", ""),
            request.target_voice,
            loaded_model_mode or MODEL_MODE,
        )
        raise
    except Exception as error:
        logger.exception("Failed to generate Qwen TTS audio: %s", error)
        raise HTTPException(status_code=500, detail="Failed to generate speech")


@app.get("/voices", dependencies=[Depends(verify_api_key)])
async def list_voices():
    return {"voices": _resolve_supported_voices()}


@app.get("/health", dependencies=[Depends(verify_api_key)])
async def health_check():
    if qwen_model is not None:
        return {
            "status": "healthy",
            "model": loaded_model_id or MODEL_ID,
            "mode": loaded_model_mode or MODEL_MODE,
        }
    return {"status": "unhealthy", "model": "not loaded", "mode": MODEL_MODE}
