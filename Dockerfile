ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install system dependencies required for ML libraries and compilation
RUN apk add --no-cache \
    build-essential \
    gcc \
    g++ \
    gfortran \
    make \
    cmake \
    git \
    libc-dev \
    musl-dev \
    linux-headers \
    curl \
    ca-certificates

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the ML Forecast Lab package
COPY ml_forecast_lab /app/ml_forecast_lab

# Copy rootfs (includes s6-overlay configurations and startup scripts)
COPY rootfs /

# Set working directory
WORKDIR /app

# Set Python path to include the app directory
ENV PYTHONPATH=/app:$PYTHONPATH

# Run the application via s6-overlay
CMD [ "/init" ]
