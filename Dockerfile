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
    && git clone https://github.com/GabrielvanderSchmidt/SegNeXt.git \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    openjdk-11-jre-headless \
    # MMDet Requirements
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir SegNeXt/checkpoints \
    && mkdir -p videos/{original/{todo,done},segmented} \
    # Download pretrained model checkpoint
    && wget -nv https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth -P SegNeXt/checkpoints \
    && wget -nv https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x1024_40k_cityscapes/pspnet_r101-d8_512x1024_40k_cityscapes_20200604_232751-467e7cf4.pth -P SegNeXt/checkpoints \
    && wget -nv https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r18-d8_512x1024_80k_cityscapes/pspnet_r18-d8_512x1024_80k_cityscapes_20201225_021458-09ffa746.pth -P SegNeXt/checkpoints \
    && wget -nv https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r18b-d8_512x1024_80k_cityscapes/pspnet_r18b-d8_512x1024_80k_cityscapes_20201226_063116-26928a60.pth -P SegNeXt/checkpoints \
    && wget -nv https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d32_512x1024_80k_cityscapes/pspnet_r50-d32_512x1024_80k_cityscapes_20220316_224840-9092b254.pth -P SegNeXt/checkpoints \
    && wget -nv https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d32_rsb-pretrain_512x1024_adamw_80k_cityscapes/pspnet_r50-d32_rsb-pretrain_512x1024_adamw_80k_cityscapes_20220316_141229-dd9c9610.pth -P SegNeXt/checkpoints \
    && wget -nv https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50b-d32_512x1024_80k_cityscapes/pspnet_r50b-d32_512x1024_80k_cityscapes_20220311_152152-23bcaf8c.pth -P SegNeXt/checkpoints

    

ENV PATH="/opt/conda/bin:$PATH"
ENV FORCE_CUDA="1"

# MMLAB
RUN ["/bin/bash", "-c", "pip install mmcv-full==${MMCV}+torch${PYTORCH}+cu${CUDA//./} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]
RUN pip install mmsegmentation==${MMSEG}

CMD python3 /home/SegNeXt/entrypoint.py
