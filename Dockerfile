# Reproducible environment for the curvature experiments (Sections 6.1-6.4).
#   docker build -t curvature-transfer .
#   docker run --rm -v "$PWD/results:/repo/results" curvature-transfer            # full reproduction
#   docker run --rm -v "$PWD/results:/repo/results" curvature-transfer --quick    # smoke run
FROM python:3.12-slim
ENV PYTHONHASHSEED=0 MPLBACKEND=Agg PIP_NO_CACHE_DIR=1
WORKDIR /repo
# git: orcut (the exact solver, pinned to 1.0.0) is installed from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY . /repo
RUN pip install -e ".[experiments,dev]"
ENTRYPOINT ["python", "main.py"]
