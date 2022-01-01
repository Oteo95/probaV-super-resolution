FROM tensorflow/tensorflow:latest-gpu


COPY dockerfiles/requirements.txt /tmp/
RUN set -ex; \
    pip3 --no-cache-dir install --requirement /tmp/requirements.txt

ARG USERNAME=research
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

USER $USERNAME
ENV PYTHONPATH "/workspaces/probaV-super-resolution/src"