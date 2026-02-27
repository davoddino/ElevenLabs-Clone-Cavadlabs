import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager

import boto3
import numpy as np
import soundfile as sf
import torch

# Compatibility shim for newer transformers with older torch builds.
if (
    hasattr(torch.utils, "_pytree")
    and not hasattr(torch.utils._pytree, "register_pytree_node")
    and hasattr(torch.utils._pytree, "_register_pytree_node")
):
    def _compat_register_pytree_node(*args, **kwargs):
        try:
            return torch.utils._pytree._register_pytree_node(*args, **kwargs)
        except TypeError:
            kwargs.pop("serialized_type_name", None)
            kwargs.pop("flatten_with_keys_fn", None)
            return torch.utils._pytree._register_pytree_node(*args, **kwargs)

    torch.utils._pytree.register_pytree_node = _compat_register_pytree_node

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from qwen_omni_utils import process_mm_info
from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_KEY")
AUTH_DISABLED = os.getenv("DISABLE_API_KEY_AUTH", "false").lower() == "true"
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()
LOCAL_STORAGE_ROOT = os.getenv("LOCAL_STORAGE_ROOT", "/data/storage")
MODEL_ID = os.getenv("QWEN_TTS_MODEL_ID", "Qwen/Qwen3-TTS-0.6B")
SAMPLE_RATE = int(os.getenv("QWEN_TTS_SAMPLE_RATE", "24000"))
MAX_TEXT_LENGTH = int(os.getenv("QWEN_TTS_MAX_TEXT_LENGTH", "3000"))
MAX_NEW_TOKENS = int(os.getenv("QWEN_TTS_MAX_NEW_TOKENS", "2048"))

DEFAULT_VOICES = ["Cherry", "Chelsie", "Ethan", "Serena", "Dylan", "Jada"]
SUPPORTED_VOICES = [
    voice.strip()
    for voice in os.getenv("QWEN_TTS_VOICES", ",".join(DEFAULT_VOICES)).split(",")
    if voice.strip()
]

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
qwen_model = None
qwen_processor = None


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


def _build_messages(text: str, voice: str):
    instruction = (
        "Generate the audio according to the following instruction: "
        f"{text} Voice: {voice}"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "audio", "audio_url": "<|AUDIO|>"},
            ],
        }
    ]


def _generate_audio(text: str, voice: str) -> np.ndarray:
    messages = _build_messages(text, voice)
    input_text = qwen_processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

    inputs = qwen_processor(
        text=input_text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )

    model_device = next(qwen_model.parameters()).device
    inputs = inputs.to(model_device)

    with torch.inference_mode():
        output = qwen_model.generate(
            **inputs,
            use_audio_in_video=False,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    output_ids = output[0][inputs.input_ids.shape[1] :]
    decoded_audio = qwen_model.audio_tokenizer.decode(
        output_ids,
        output.audio_codes[0],
        output.audio_scales[0],
        output.audio_logits[0],
    )

    if isinstance(decoded_audio, torch.Tensor):
        decoded_audio = decoded_audio.detach().cpu().numpy()

    return np.asarray(decoded_audio, dtype=np.float32)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qwen_model, qwen_processor

    logger.info("Loading Qwen TTS model: %s", MODEL_ID)
    try:
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

        qwen_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_ID,
            **model_kwargs,
        )
        if not torch.cuda.is_available():
            qwen_model = qwen_model.to("cpu")
        qwen_model.eval()

        qwen_processor = AutoProcessor.from_pretrained(MODEL_ID)
        logger.info("Qwen TTS model loaded successfully")
    except Exception as error:
        logger.exception("Failed to load Qwen TTS model: %s", error)
        raise

    yield

    logger.info("Shutting down Qwen TTS API")


app = FastAPI(title="Qwen TTS API", lifespan=lifespan)


class TextOnlyRequest(BaseModel):
    text: str
    target_voice: str


@app.post("/generate", dependencies=[Depends(verify_api_key)])
async def generate_speech(
    request: TextOnlyRequest,
    background_tasks: BackgroundTasks,
):
    if not qwen_model or not qwen_processor:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if len(request.text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text length exceeds the limit of {MAX_TEXT_LENGTH} characters",
        )

    if request.target_voice not in SUPPORTED_VOICES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Target voice not supported. Choose from: "
                + ", ".join(SUPPORTED_VOICES)
            ),
        )

    try:
        audio = _generate_audio(request.text, request.target_voice)

        audio_id = str(uuid.uuid4())
        output_filename = f"{audio_id}.wav"
        local_path = f"/tmp/{output_filename}"
        sf.write(local_path, audio, samplerate=SAMPLE_RATE)

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
    except Exception as error:
        logger.exception("Failed to generate Qwen TTS audio: %s", error)
        raise HTTPException(status_code=500, detail="Failed to generate speech")


@app.get("/voices", dependencies=[Depends(verify_api_key)])
async def list_voices():
    return {"voices": SUPPORTED_VOICES}


@app.get("/health", dependencies=[Depends(verify_api_key)])
async def health_check():
    if qwen_model and qwen_processor:
        return {"status": "healthy", "model": MODEL_ID}
    return {"status": "unhealthy", "model": "not loaded"}
