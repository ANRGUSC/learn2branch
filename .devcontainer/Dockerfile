FROM mcr.microsoft.com/devcontainers/miniconda:0-3

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
RUN conda update conda && \
    conda create -n l2b python=3.6 && \
    conda init
SHELL ["/bin/bash", "--login", "-c"]
RUN conda activate l2b && \
    conda install cython numpy scikit-learn=0.20.2 tensorflow-gpu=1.12.0 && \
    pip install git+https://github.com/ds4dm/PySCIPOpt.git@ml-branching && \
    pip install git+https://github.com/jma127/pyltr@78fa0ebfef67d6594b8415aa5c6136e30a5e3395 && \
    git clone https://github.com/ANRGUSC/PySVMRank.git && \
    cd PySVMRank && \
    git checkout bugfix/verbosity && \
    wget http://download.joachims.org/svm_rank/current/svm_rank.tar.gz && \
    mkdir src/c && \
    tar -xzf svm_rank.tar.gz -C src/c && \
    bash patch.sh && \ 
    pip install .


# activate the conda environment "l2b" by default
RUN echo "conda activate l2b" >> ~/.bashrc