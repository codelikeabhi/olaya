# Sandbox for OlayaBench Track C runs: one Exercism exercise per container, graded by pytest.
FROM python:3.12-slim
RUN pip install --no-cache-dir pytest==8.3.3 && useradd -m agent
USER agent
WORKDIR /work
