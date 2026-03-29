ARG BUILD_FROM
ARG BUILD_VERSION
FROM ${BUILD_FROM}

# Set shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install Python and system dependencies for ML library compilation
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
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

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the ML Forecast Lab package
COPY ml_forecast_lab /app/ml_forecast_lab

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
