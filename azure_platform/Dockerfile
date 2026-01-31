# =============================================================================
# Vineyard Pipeline Docker Image
# =============================================================================
# Single image that can run any of the 3 pipeline jobs:
#   - job1_ground_removal.sh
#   - job2_clustering.sh
#   - job3_features.sh
#
# Build:
#   docker build -t vineyard-pipeline:dev -f azure_platform/Dockerfile .
#
# Run examples:
#   docker run --rm -v "$(pwd)":/data -e RUN_ID="test_001" ... vineyard-pipeline:dev bash /app/azure_platform/job1_ground_removal.sh
# =============================================================================

# ---------- Build stage for C++ ----------
FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    make \
    libpcl-dev \
    libpdal-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy C++ source and CMakeLists.txt from scripts/
COPY ./scripts/CMakeLists.txt ./scripts/CMakeLists.txt
COPY ./scripts/clustering_only.cpp ./scripts/clustering_only.cpp
COPY ./scripts/alternatives ./scripts/alternatives

# Build C++ binaries
RUN cd scripts && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j$(nproc)

# ---------- Runtime stage ----------
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    pdal \
    libpdal-plugins \
    libpcl-io1.12 \
    libpcl-filters1.12 \
    libpcl-segmentation1.12 \
    libpcl-kdtree1.12 \
    libpcl-common1.12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pipeline job scripts
COPY ./azure_platform ./azure_platform

# Copy reusable Python code from scripts/
COPY ./scripts/*.py ./scripts/
COPY ./scripts/*.sh ./scripts/

# Copy built C++ binary from builder stage
COPY --from=builder /src/scripts/build/clustering_only /app/bin/clustering_only

# Install Python dependencies
COPY ./azure_platform/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Make job scripts executable
RUN chmod +x /app/azure_platform/*.sh

# Set PATH to include our bin directory
ENV PATH="/app/bin:${PATH}"

# Default working directory
WORKDIR /app

# No default CMD - user specifies which job to run
