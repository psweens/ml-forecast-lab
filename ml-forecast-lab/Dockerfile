ARG BUILD_FROM
ARG BUILD_VERSION

# ===== Stage 1: Builder =====
FROM ${BUILD_FROM} AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        build-essential \
        gcc \
        g++ \
        gfortran \
        make \
        cmake \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a virtual environment
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# ===== Stage 2: Runtime =====
FROM ${BUILD_FROM}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install only runtime dependencies (no compilers)
# Hailo RT Python bindings are installed system-wide via apt when available;
# the venv uses --system-site-packages so it can see them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        libgomp1 \
        ca-certificates \
    && ( apt-get install -y --no-install-recommends python3-hailort 2>/dev/null \
         && echo "HailoRT Python bindings installed" \
         || echo "HailoRT not available in apt — Hailo acceleration will be disabled" ) \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built virtualenv from builder and allow system site-packages
# (needed for python3-hailort which is apt-installed, not pip-installed)
COPY --from=builder /opt/venv /opt/venv
RUN if [ -f /opt/venv/pyvenv.cfg ]; then \
        sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' /opt/venv/pyvenv.cfg; \
    fi
ENV PATH="/opt/venv/bin:$PATH"

# Copy the ML Forecast Lab package and default config
COPY ml_forecast_lab /app/ml_forecast_lab
COPY mlfl.yaml /app/mlfl.yaml

# Copy rootfs (includes s6-overlay configurations and startup scripts)
COPY rootfs /

# Make s6-overlay init script executable
RUN chmod a+x /etc/s6-overlay/s6-rc.d/init-mlforecastlab/run

# Set working directory
WORKDIR /app

# Set Python path to include the app directory
ENV PYTHONPATH=/app:$PYTHONPATH

ARG BUILD_ARCH
ARG BUILD_DATE
ARG BUILD_DESCRIPTION
ARG BUILD_NAME
ARG BUILD_REF
ARG BUILD_REPOSITORY
LABEL \
    io.hass.name="${BUILD_NAME}" \
    io.hass.description="${BUILD_DESCRIPTION}" \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="addon" \
    io.hass.version=${BUILD_VERSION}
