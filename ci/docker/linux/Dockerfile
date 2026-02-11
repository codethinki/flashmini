FROM cachyos/cachyos:latest

RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
    base-devel \
    cmake \
    ninja \
    git \
    openmpi \
    cuda \
    cudnn \
    onednn \
    intel-oneapi-mkl \
    python \
    python-pip \
    vcpkg \
    wget \
    cloc

ENV VCPKG_ROOT=/opt/vcpkg

# Install ArrayFire from script
RUN wget https://arrayfire.s3.amazonaws.com/3.10.0/ArrayFire-v3.10.0_Linux_x86_64.sh -O af_installer.sh && \
    chmod +x af_installer.sh && \
    ./af_installer.sh --include-subdir --prefix=/opt --skip-license && \
    rm af_installer.sh
ENV ArrayFire_DIR=/opt/arrayfire
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/arrayfire/lib64
