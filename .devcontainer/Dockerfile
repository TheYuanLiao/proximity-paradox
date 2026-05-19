# === Base: NVIDIA CUDA 12.1 on Ubuntu 22.04 (GPU support for PyTorch) ===
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# === Prevent interactive prompts ===
ENV DEBIAN_FRONTEND=noninteractive

# === Install system dependencies ===
USER root
RUN apt-get update && apt-get install -y \
    curl \
    bzip2 \
    git \
    vim \
    htop \
    build-essential \
    cmake \
    pkg-config \
    libgeos-dev \
    libproj-dev \
    libgdal-dev \
    gdal-bin \
    # Java for PySpark
    openjdk-17-jdk \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Note: Java 17 installed above for PySpark compatibility
# Java 21 will be installed later and set as default for R/r5r

# === Install Micromamba ===
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba && \
    mv bin/micromamba /usr/local/bin/micromamba

# === Setup Micromamba environment ===
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

# === Copy environment.yml ===
COPY .devcontainer/environment.yml /tmp/env.yml

# === Create Conda environment ===
RUN micromamba create -y -f /tmp/env.yml -n geoenv && \
    micromamba clean --all --yes

# === Use this environment for all following RUN/CMD ===
SHELL ["micromamba", "run", "-n", "geoenv", "/bin/bash", "-c"]

# === Create working directory ===
WORKDIR /workspace

# === Copy and install infostop ===
RUN apt-get update && apt-get install -y build-essential g++ cmake
COPY infostop /workspace/infostop
# infomap uses legacy setup.py that needs pkg_resources (removed in setuptools 71+)
RUN pip install "setuptools<70" wheel && \
    pip install infomap==1.0.6 --no-build-isolation && \
    pip install -e /workspace/infostop

# === Install PyTorch with CUDA support (pip is more reliable than conda for this) ===
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# === Default command ===
CMD ["micromamba", "run", "-n", "geoenv", "python"]

# === Install R from CRAN repository ===
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    dirmngr \
    gnupg \
 && curl -fsSL https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc \
 && add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" \
 && apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# === Install Java 21 and configure alternatives ===
# Both Java 17 (PySpark) and Java 21 (R/r5r) are available
# Java 21 is set as default
RUN apt-get update && apt-get install -y \
    openjdk-21-jdk \
    locales \
 && locale-gen C.UTF-8 \
 && update-alternatives --install /usr/bin/java java /usr/lib/jvm/java-21-openjdk-amd64/bin/java 2 \
 && update-alternatives --install /usr/bin/java java /usr/lib/jvm/java-17-openjdk-amd64/bin/java 1 \
 && update-alternatives --set java /usr/lib/jvm/java-21-openjdk-amd64/bin/java

# Java 21 as default (for R/r5r), PySpark can use SPARK_HOME or explicit path if needed
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$JAVA_HOME/lib/server:$LD_LIBRARY_PATH
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN R CMD javareconf

# Install R packages
RUN R -q -e "install.packages(c('rjava','r5r','data.table'), repos='https://cloud.r-project.org')"