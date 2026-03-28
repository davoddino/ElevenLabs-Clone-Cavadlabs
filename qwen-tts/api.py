import base64
import io
import json
import logging
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
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
VOXTRAL_TTS_API_ROUTE = os.getenv(
    "VOXTRAL_TTS_API_ROUTE", "http://127.0.0.1:8000/v1/audio/speech"
).strip()
VOXTRAL_TTS_API_KEY = os.getenv("VOXTRAL_TTS_API_KEY", "").strip()
VOXTRAL_RESPONSE_FORMAT = os.getenv("VOXTRAL_TTS_RESPONSE_FORMAT", "wav").strip() or "wav"
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

MODEL_MODE = _detect_model_mode(MODEL_ID, os.getenv("QWEN_TTS_MODEL_MODE"))

DEFAULT_VOICES = ["Cherry", "Chelsie", "Ethan", "Serena", "Dylan", "Jada"]
ENV_VOICES = _parse_voices(os.getenv("QWEN_TTS_VOICES"))

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
qwen_models: dict[str, Any] = {}
qwen_model_modes: dict[str, str] = {}
qwen_supported_speakers: dict[str, list[str]] = {}
voxtral_server_process: subprocess.Popen[str] | None = None


def _is_voxtral_model(model_id: str) -> bool:
    return model_id.strip().lower().startswith("mistralai/voxtral")


def _resolve_model_id(request_model_id: str | None) -> str:
    if request_model_id and request_model_id.strip():
        return request_model_id.strip()
    return MODEL_ID


def _resolve_model_mode_for_id(model_id: str) -> str:
    if model_id in qwen_model_modes:
        return qwen_model_modes[model_id]
    return _detect_model_mode(model_id, os.getenv("QWEN_TTS_MODEL_MODE"))


def _ensure_qwen_model_loaded(model_id: str) -> tuple[Any, str, list[str]]:
    if model_id in qwen_models:
        return (
            qwen_models[model_id],
            qwen_model_modes[model_id],
            qwen_supported_speakers.get(model_id, []),
        )

    model_mode = _detect_model_mode(model_id, os.getenv("QWEN_TTS_MODEL_MODE"))
    model = _load_qwen_model(model_id)
    speakers: list[str] = []

    if model_mode == "custom_voice" and hasattr(model, "get_supported_speakers"):
        raw_speakers = model.get_supported_speakers()
        if isinstance(raw_speakers, (list, tuple)):
            speakers = [str(item) for item in raw_speakers]

    qwen_models[model_id] = model
    qwen_model_modes[model_id] = model_mode
    qwen_supported_speakers[model_id] = speakers
    logger.info("Loaded Qwen model on demand: %s (mode=%s)", model_id, model_mode)
    return model, model_mode, speakers


def _build_vllm_serve_command(model_id: str, route: str) -> list[str]:
    parsed = urllib.parse.urlparse(route)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 8000)
    return ["vllm", "serve", model_id, "--omni", "--host", host, "--port", str(port)]


def _voxtral_models_url(route: str) -> str:
    parsed = urllib.parse.urlparse(route)
    base_path = parsed.path or "/v1/audio/speech"
    if "/audio/speech" in base_path:
        path = base_path.replace("/audio/speech", "/models")
    else:
        path = "/v1/models"
    return urllib.parse.urlunparse(
        (
            parsed.scheme or "http",
            parsed.netloc,
            path,
            "",
            "",
            "",
        )
    )


def _is_voxtral_server_reachable(route: str, timeout: float = 2.0) -> bool:
    probe_url = _voxtral_models_url(route)
    req = urllib.request.Request(probe_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def _ensure_voxtral_server_running(model_id: str) -> None:
    global voxtral_server_process

    if _is_voxtral_server_reachable(VOXTRAL_TTS_API_ROUTE):
        return

    if voxtral_server_process is None or voxtral_server_process.poll() is not None:
        if not shutil.which("vllm"):
            raise HTTPException(
                status_code=500,
                detail=(
                    "Voxtral richiede `vllm` installato. Installa con `pip install -U vllm` "
                    "e `pip install git+https://github.com/vllm-project/vllm-omni.git --upgrade`."
                ),
            )

        cmd = _build_vllm_serve_command(model_id, VOXTRAL_TTS_API_ROUTE)
        logger.info("Starting vLLM Omni for Voxtral: %s", " ".join(cmd))
        voxtral_server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )

    deadline = time.time() + 180
    while time.time() < deadline:
        if _is_voxtral_server_reachable(VOXTRAL_TTS_API_ROUTE):
            logger.info("Voxtral server is reachable")
            return
        if voxtral_server_process is not None and voxtral_server_process.poll() is not None:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Il processo vLLM Omni si e' chiuso durante l'avvio. "
                    "Controlla GPU/CUDA e dipendenze vllm-omni."
                ),
            )
        time.sleep(1.5)

    raise HTTPException(
        status_code=504,
        detail="Timeout mentre avviavo vLLM Omni per Voxtral.",
    )


def _parse_voxtral_audio_response(
    raw_bytes: bytes,
    content_type: str | None,
) -> tuple[np.ndarray, int]:
    if content_type and "application/json" in content_type.lower():
        payload = json.loads(raw_bytes.decode("utf-8"))
        audio_b64: str | None = None

        if isinstance(payload, dict):
            if isinstance(payload.get("audio"), str):
                audio_b64 = payload["audio"]
            elif isinstance(payload.get("data"), list) and payload["data"]:
                first_item = payload["data"][0]
                if isinstance(first_item, dict):
                    if isinstance(first_item.get("b64_json"), str):
                        audio_b64 = first_item["b64_json"]
                    elif isinstance(first_item.get("audio"), str):
                        audio_b64 = first_item["audio"]

        if not audio_b64:
            raise RuntimeError("Voxtral API returned JSON without audio payload")

        raw_bytes = base64.b64decode(audio_b64)

    audio_arr, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    if isinstance(audio_arr, np.ndarray) and audio_arr.ndim > 1:
        audio_arr = audio_arr[:, 0]
    return np.asarray(audio_arr, dtype=np.float32), int(sample_rate)


def _generate_voxtral_audio(
    request: "TextOnlyRequest",
    model_id: str,
) -> tuple[np.ndarray, int]:
    _ensure_voxtral_server_running(model_id)

    payload: dict[str, Any] = {
        "model": model_id,
        "input": request.text,
        "response_format": VOXTRAL_RESPONSE_FORMAT,
    }
    if request.target_voice:
        payload["voice"] = request.target_voice

    if request.instruct:
        payload["instructions"] = request.instruct

    headers = {"Content-Type": "application/json"}
    if VOXTRAL_TTS_API_KEY:
        headers["Authorization"] = f"Bearer {VOXTRAL_TTS_API_KEY}"

    req = urllib.request.Request(
        VOXTRAL_TTS_API_ROUTE,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type")
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="ignore")
        raise HTTPException(
            status_code=502,
            detail=f"Voxtral API error: {error.code} {error_body}",
        ) from error
    except urllib.error.URLError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Voxtral API not reachable: {error}",
        ) from error

    try:
        return _parse_voxtral_audio_response(raw, content_type)
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to parse Voxtral audio response: {error}",
        ) from error


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


def _ensure_local_storage_root_writable() -> None:
    try:
        os.makedirs(LOCAL_STORAGE_ROOT, exist_ok=True)
        probe_path = os.path.join(LOCAL_STORAGE_ROOT, ".qwen_write_probe")
        with open(probe_path, "wb") as probe:
            probe.write(b"ok")
        os.remove(probe_path)
    except Exception as error:
        raise RuntimeError(
            f"LOCAL_STORAGE_ROOT is not writable: {LOCAL_STORAGE_ROOT}. "
            "Fix directory ownership/permissions."
        ) from error


def _resolve_supported_voices(
    model_mode: str,
    supported_speakers: list[str] | None = None,
) -> list[str]:
    if model_mode == "custom_voice" and supported_speakers:
        return supported_speakers

    if model_mode == "base" and VOICE_CLONE_PRESETS:
        return list(VOICE_CLONE_PRESETS.keys())

    if ENV_VOICES:
        return ENV_VOICES

    if model_mode == "voice_design":
        return ["VoiceDesign"]

    if model_mode == "voxtral_tts":
        return ENV_VOICES or ["casual_male"]

    return DEFAULT_VOICES


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


def _generate_audio(
    request: "TextOnlyRequest",
    model_id: str,
) -> tuple[np.ndarray, int]:
    language = request.language or DEFAULT_LANGUAGE

    if _is_voxtral_model(model_id):
        return _generate_voxtral_audio(request, model_id)

    qwen_model, model_mode, supported_speakers = _ensure_qwen_model_loaded(model_id)

    if model_mode == "custom_voice":
        available_voices = _resolve_supported_voices(model_mode, supported_speakers)
        speaker = request.target_voice or (available_voices[0] if available_voices else None)
        if not speaker:
            raise HTTPException(status_code=400, detail="No speaker available for CustomVoice model")

        if supported_speakers and speaker not in supported_speakers:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Target voice not supported for this model. Choose from: "
                    + ", ".join(supported_speakers)
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

    if model_mode == "voice_design":
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

    if model_mode == "base":
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

    raise HTTPException(status_code=500, detail=f"Unsupported model mode: {model_mode}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Default TTS model: %s", MODEL_ID)
    logger.info("Default detected mode: %s", MODEL_MODE)

    try:
        if STORAGE_BACKEND == "local":
            _ensure_local_storage_root_writable()
            logger.info("Using local storage root: %s", LOCAL_STORAGE_ROOT)
    except Exception as error:
        logger.exception("Failed during API startup: %s", error)
        raise

    yield

    if voxtral_server_process is not None and voxtral_server_process.poll() is None:
        logger.info("Shutting down managed Voxtral server process")
        voxtral_server_process.terminate()

    logger.info("Shutting down Qwen TTS API")


app = FastAPI(title="Qwen TTS API", lifespan=lifespan)


class TextOnlyRequest(BaseModel):
    text: str
    model_id: str | None = None
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
    if len(request.text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text length exceeds the limit of {MAX_TEXT_LENGTH} characters",
        )

    resolved_model_id = _resolve_model_id(request.model_id)

    try:
        audio, sample_rate = _generate_audio(request, resolved_model_id)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise RuntimeError("Generated empty audio buffer")
        if not np.isfinite(audio).all():
            raise RuntimeError("Generated audio contains non-finite values")
        audio = np.clip(audio, -1.0, 1.0)

        audio_id = str(uuid.uuid4())
        output_filename = f"{audio_id}.wav"
        local_path = f"/tmp/{output_filename}"
        # Force browser-friendly WAV output.
        sf.write(local_path, audio, samplerate=sample_rate, format="WAV", subtype="PCM_16")

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
            logger.info(
                "Saved local audio file: key=%s path=%s exists=%s",
                s3_key,
                final_path,
                os.path.exists(final_path),
            )
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
            _resolve_model_mode_for_id(resolved_model_id),
        )
        raise
    except Exception as error:
        logger.exception("Failed to generate Qwen TTS audio: %s", error)
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {error}")


@app.get("/voices", dependencies=[Depends(verify_api_key)])
async def list_voices(model_id: str | None = None):
    resolved_model_id = _resolve_model_id(model_id)
    if _is_voxtral_model(resolved_model_id):
        return {"voices": _resolve_supported_voices("voxtral_tts")}

    _, model_mode, speakers = _ensure_qwen_model_loaded(resolved_model_id)
    return {"voices": _resolve_supported_voices(model_mode, speakers)}


@app.get("/health", dependencies=[Depends(verify_api_key)])
async def health_check():
    return {
        "status": "healthy",
        "default_model": MODEL_ID,
        "default_mode": _resolve_model_mode_for_id(MODEL_ID),
        "loaded_qwen_models": list(qwen_models.keys()),
        "voxtral_route": VOXTRAL_TTS_API_ROUTE,
        "local_storage_root": LOCAL_STORAGE_ROOT if STORAGE_BACKEND == "local" else None,
    }


@app.get("/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    return {
        "models": [
            {
                "id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                "mode": "custom_voice",
                "provider": "qwen",
            },
            {
                "id": "mistralai/Voxtral-4B-TTS-2603",
                "mode": "voxtral_tts",
                "provider": "mistral",
            },
        ]
    }
