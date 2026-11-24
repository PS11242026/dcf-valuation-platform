# Hugging Face Space (Docker SDK) for the DCF Valuation Platform API.
# Upload this file and README.md to a new Docker Space; set the Space secret
# DCF_SEC_USER_AGENT to "<your name> <your email>" (SEC EDGAR requires a contact).
# The engine is installed from the public GitHub repo at build time: after pushing engine
# changes, use Settings -> "Factory rebuild" on the Space to pick them up.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

ARG ENGINE_REPO=https://github.com/PS11242026/dcf-valuation-platform.git
ARG ENGINE_REF=main
RUN git clone --depth 1 --branch ${ENGINE_REF} ${ENGINE_REPO} engine \
    && pip install --no-cache-dir --user -e "./engine[api]"

ENV DCF_REPORTS_DIR=/tmp/dcf/reports \
    DCF_CACHE_DIR=/tmp/dcf/cache \
    MPLCONFIGDIR=/tmp/dcf/mpl \
    DCF_API_ORIGINS=https://ps11242026.github.io \
    DCF_API_WORKERS=1 \
    DCF_API_RATE_PER_HOUR=15

EXPOSE 7860
CMD ["uvicorn", "dcf_project.api.app:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
