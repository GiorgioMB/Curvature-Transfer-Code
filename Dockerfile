# The environment of the experiments (Section 6).
#   docker build -t tqbound .
#   docker run --rm -v "$PWD/results:/repo/results" -v "$PWD/logs:/repo/logs" tqbound            # full reproduction
#   docker run --rm -v "$PWD/results:/repo/results" -v "$PWD/logs:/repo/logs" tqbound --quick    # smoke run
FROM python:3.12-slim
ENV PYTHONHASHSEED=0 MPLBACKEND=Agg PIP_NO_CACHE_DIR=1
WORKDIR /repo
COPY . /repo
RUN pip install -e ".[experiments,dev]"
ENTRYPOINT ["python", "main.py"]
