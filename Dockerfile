# The environment of the experiments (Section 6).
#   docker build -t curvature-transfer .
#   docker run --rm -v "$PWD/results:/repo/results" -v "$PWD/logs:/repo/logs" curvature-transfer            # full reproduction
#   docker run --rm -v "$PWD/results:/repo/results" -v "$PWD/logs:/repo/logs" curvature-transfer --quick    # smoke run
FROM python:3.12-slim
ENV PYTHONHASHSEED=0 MPLBACKEND=Agg PIP_NO_CACHE_DIR=1
WORKDIR /repo
COPY . /repo
RUN pip install -e ".[experiments,dev]"
ENTRYPOINT ["python", "main.py"]
