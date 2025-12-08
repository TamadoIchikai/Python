FROM cupy/cupy:v13.0.0

# Install sudo and create devuser with UID/GID from build args
ARG UID=1000
ARG GID=1000

RUN apt-get update -y && \
    apt-get install -y sudo && \
    groupadd -g ${GID} devuser || true && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash devuser && \
    usermod -aG sudo devuser && \
    printf "devuser ALL=(ALL) NOPASSWD:ALL\n" > /etc/sudoers.d/devuser && \
    chmod 0440 /etc/sudoers.d/devuser
