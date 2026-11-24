"""Hugging Face Space entry point (free Gradio SDK slot, no Docker needed).

HF runs `python app.py` and routes the Space URL to port 7860; this starts the DCF platform's
FastAPI analysis service there. The engine itself is installed from GitHub via requirements.txt
(Settings -> "Factory rebuild" picks up new engine commits).
Secret required: DCF_SEC_USER_AGENT = "<your name> <your email>" (SEC EDGAR contact).
"""
import os

os.environ.setdefault("DCF_REPORTS_DIR", "/tmp/dcf/reports")
os.environ.setdefault("DCF_CACHE_DIR", "/tmp/dcf/cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dcf/mpl")
os.environ.setdefault("DCF_API_ORIGINS", "https://ps11242026.github.io")
os.environ.setdefault("DCF_API_WORKERS", "1")
os.environ.setdefault("DCF_API_RATE_PER_HOUR", "15")
for d in ("DCF_REPORTS_DIR", "DCF_CACHE_DIR", "MPLCONFIGDIR"):
    os.makedirs(os.environ[d], exist_ok=True)

import uvicorn  # noqa: E402

from dcf_project.api.app import app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")),
                proxy_headers=True, forwarded_allow_ips="*")
