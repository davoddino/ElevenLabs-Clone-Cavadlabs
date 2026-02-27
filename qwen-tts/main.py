import os

import uvicorn


def _int_env(*names: str, default: int) -> int:
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return default


if __name__ == "__main__":
    host = os.getenv("QWEN_TTS_HOST") or os.getenv("HOST") or "0.0.0.0"
    port = _int_env("QWEN_TTS_PORT", "PORT", default=8003)
    reload_enabled = (
        os.getenv("QWEN_TTS_RELOAD", "false").strip().lower() == "true"
    )
    log_level = os.getenv("QWEN_TTS_LOG_LEVEL", "info")

    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload_enabled,
        log_level=log_level,
    )
