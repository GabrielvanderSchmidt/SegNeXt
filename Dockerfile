ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ARG MMCV="1.4.8"
ARG MMSEG="0.24.1"

ENV PYTHONUNBUFFERED TRUE

# -- Run to avoid GPG error --
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# ----------------------------
# (from https://github.com/NVIDIA/nvidia-docker/issues/1632)

WORKDIR /home
# Must build container with --no-cache option, otherwise you will be stuck with older commits.

# -- Clone modified repository and model files --
RUN apt-get update \
    && apt-get install wget -y \
    && apt-get install git -y \
    && git clone https://github.com/GabrielvanderSchmidt/SegNeXt.git
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    openjdk-11-jre-headless \
    # MMDet Requirements
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    # Download pretrained model checkpoint
    && mkdir SegNeXt/checkpoints \
    && mkdir -p videos/{original/{todo, done}, segmented} \
    && wget https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth -P SegNeXt/checkpoints

    

ENV PATH="/opt/conda/bin:$PATH"
RUN export FORCE_CUDA=1

# TORCHSEVER
#RUN pip install torchserve torch-model-archiver

# MMLAB
ARG PYTORCH
ARG CUDA
RUN ["/bin/bash", "-c", "pip install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]
RUN pip install mmsegmentation==${MMSEG}

# -- Run only on CPU --
RUN export FORCE_CUDA=0
RUN export CUDA_VISIBLE_DEVICES=""

CMD python3 /home/SegNeXt/entrypoint.py
