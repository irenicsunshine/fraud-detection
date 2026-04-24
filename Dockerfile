# =============================================================================
# 🎓 DOCKERFILE — Packaging the App into a Container
# =============================================================================
#
# WHAT IS DOCKER?
# ----------------
# Docker lets you bundle your app + its dependencies into a single "image"
# that runs identically everywhere — your laptop, a server, or the cloud.
#
# Without Docker:
#   "It works on my machine" → fails on the server because Python version
#   differs, a package is missing, or the OS behaves differently.
#
# With Docker:
#   The image contains Python 3.11, all packages, your code — everything.
#   The server just runs the image. No setup, no surprises.
#
# HOW TO BUILD AND RUN LOCALLY:
# ------------------------------
#   docker build -t fraud-detection .
#   docker run -p 8000:8000 fraud-detection
#   → API available at http://localhost:8000
#
# WHAT EACH INSTRUCTION DOES:
# ----------------------------
# FROM    → base image (like choosing an OS + Python pre-installed)
# WORKDIR → set working directory inside the container
# COPY    → copy files from your machine INTO the container
# RUN     → execute a command during the build (install packages, etc.)
# EXPOSE  → document which port the app listens on
# CMD     → the command that runs when the container starts
# =============================================================================

# 🎓 BASE IMAGE
# We use python:3.11-slim — a minimal Debian Linux with Python 3.11.
# "slim" means it strips out unnecessary tools, keeping the image small.
# Smaller image = faster deploys, less storage, smaller attack surface.
FROM python:3.11-slim

# 🎓 WORKING DIRECTORY
# All subsequent commands run inside /app inside the container.
# Think of it like "cd /app" — it's created automatically if it doesn't exist.
WORKDIR /app

# 🎓 SYSTEM DEPENDENCIES
# Some Python packages need C libraries to compile.
# - gcc: C compiler (needed by some packages like numpy on some platforms)
# - libgomp1: OpenMP library (needed by XGBoost for parallelism)
# We clean up apt cache afterward to keep the image small.
RUN apt-get update && apt-get install -y \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 🎓 COPY REQUIREMENTS FIRST (layer caching trick)
# Docker builds in layers. If requirements.txt hasn't changed,
# Docker reuses the cached layer and skips re-installing packages.
# This makes rebuilds MUCH faster when you only change your code.
#
# Pattern: copy requirements → install → copy code
# NOT:     copy everything → install (invalidates cache on every code change)
COPY requirements.txt .

# 🎓 INSTALL PYTHON DEPENDENCIES
# --no-cache-dir: don't cache downloaded packages (saves space in the image)
RUN pip install --no-cache-dir -r requirements.txt

# 🎓 COPY APPLICATION CODE
# Now copy everything else. This layer changes often (every code edit),
# but since packages are already installed above, it's fast.
COPY src/ ./src/
COPY api.py .
COPY main.py .

# 🎓 COPY PRE-TRAINED MODELS
# The models/ directory contains trained_models.pkl and feature_names.pkl.
# These are checked in so the API can start immediately without training.
COPY models/ ./models/

# 🎓 CREATE NON-ROOT USER (security best practice)
# Running as root inside a container is a security risk.
# If an attacker exploits your app, they'd have root access to the container.
# Running as a non-root user limits the blast radius.
RUN useradd --create-home appuser
USER appuser

# 🎓 EXPOSE PORT
# Documents that this container listens on port 8000.
# This is documentation only — you still need -p 8000:8000 in docker run.
EXPOSE 8000

# 🎓 HEALTHCHECK
# Docker periodically runs this command to check if the container is healthy.
# If it fails repeatedly, Docker can restart the container automatically.
# --interval=30s: check every 30 seconds
# --timeout=10s: fail if no response in 10 seconds
# --retries=3: mark unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# 🎓 START COMMAND
# This runs when the container starts.
# uvicorn: the ASGI server that serves our FastAPI app
# api:app: "in the file api.py, find the variable named app"
# --host 0.0.0.0: listen on all network interfaces (required in Docker)
# --port 8000: the port to listen on
# (no --reload in production — reload watches files for changes, wastes CPU)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
