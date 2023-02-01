FROM mcr.microsoft.com/devcontainers/miniconda:0-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# [Optional] Uncomment to install a different version of Python than the default
# RUN conda install -y python=3.6 \
#     && pip install --no-cache-dir pipx \
#     && pipx reinstall-all

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# Install Soplex

ENV SCIPOPTDIR=/opt/scip

# install build-essentials, cmake, make, etc.
RUN apt-get update && \
    apt-get install -y build-essential cmake

RUN wget https://soplex.zib.de/download/release/soplex-4.0.1.tgz && \
    tar -xzf soplex-4.0.1.tgz && \
    cd soplex-4.0.1/ && \
    mkdir build && \
    cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$SCIPOPTDIR && \
    make -C ./build -j 4 && \
    sudo make -C ./build install && \
    cd ..

# Install SCIP
COPY ./scip_patch/vanillafullstrong.patch /tmp/vanillafullstrong.patch
RUN wget https://www.scipopt.org/download/release/scip-6.0.1.tgz && \
    tar -xzf scip-6.0.1.tgz && \
    cd scip-6.0.1/ && \
    patch -p1 < /tmp/vanillafullstrong.patch && \
    mkdir build && \
    cmake -S . -B build -DSOPLEX_DIR=$SCIPOPTDIR -DCMAKE_INSTALL_PREFIX=$SCIPOPTDIR && \
    make -C ./build -j 4 && \
    sudo make -C ./build install && \
    cd ..

# Intialize conda environment
RUN conda init bash

COPY .devcontainer/environment.yml environment.yml
RUN conda env update -n base -f environment.yml