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

RUN apt-get update \
    && apt-get install wget -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    openjdk-11-jre-headless \
    # MMDet Requirements
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:$PATH"
RUN export FORCE_CUDA=1

# TORCHSEVER
#RUN pip install torchserve torch-model-archiver

# MMLAB
ARG PYTORCH
ARG CUDA
RUN ["/bin/bash", "-c", "pip install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]
RUN pip install mmsegmentation==${MMSEG}

#RUN useradd -m model-server \
#    && mkdir -p /home/model-server/tmp

#COPY entrypoint.sh /usr/local/bin/entrypoint.sh

#RUN chmod +x /usr/local/bin/entrypoint.sh \
#    && chown -R model-server /home/model-server

#COPY config.properties /home/model-server/config.properties
#RUN mkdir /home/model-server/model-store 

# -- Clone modified repository and model files --
WORKDIR /home
RUN git clone https://github.com/GabrielvanderSchmidt/SegNeXt.git
# Must build container with --no-cache option, otherwise you will be stuck with older commits.
# -----------------------------------------------


# To use Cityscapes:
#RUN wget --no-verbose https://cloud.tsinghua.edu.cn/seafhttp/files/3eefc3ea-708f-4032-a236-6668ea9bb3d2/segnext_large_1024x1024_city_160k.pth \
#    && mv SegNeXt/local_configs/segnext/large/segnext.large.1024x1024.city.160k.py /home/model-server/model-store/city-configs.py
# Or, to use ADE20K:
#RUN apt-get install wget \
#    && wget https://cloud.tsinghua.edu.cn/f/d4f8e1020643414fbf7f/?dl=1 \ # <- wrong link, this will download index.html
#    && mv segnext_large_512x512_ade_160k.pth /home/model-server/model-store/ade-model.pth \
#    && mv SegNeXt/local_configs/segnext/large/segnext.large.512x512.ade.160k.py /home/model-server/model-store/ade-configs.py
# -----------------------------------------------

# -- Move SegNeXt/configs/_base_/ to /home/_base_ --
#RUN mv SegNeXt/configs/_base_/ /home/_base_/ # <- Error had nothing to do with this, it was a relative path local_configs/segnext/large/segnext...160k.py leading to a different folder
# Anything else that needs to be moved?
# --------------------------------------------------

# -- Create .mar file -- (WIP) 
# docker exec -it segnext-serve bash
#RUN python /home/SegNeXt/tools/torchserve/mmseg2torchserve.py \
#    --config "/home/model-server/model-store/city-configs.py" \
#    --checkpoint "/home/model-server/model-store/city-model.pth" \
#    --output-folder "/home/model-server/model-store/" \
#    --model-name segnext-city
# Do we still need to register the model, or does the config.properties -> load_models=all do that already?
# ----------------------

#RUN chown -R model-server /home/model-server/model-store

#EXPOSE 8080 8081 8082

#USER model-server
#WORKDIR /home/model-server
#ENV TEMP=/home/model-server/tmp

# -- Added to (hopefully) run only on CPU --
RUN export FORCE_CUDA=0
RUN export CUDA_VISIBLE_DEVICES=""
# When working with SegNeXt through Docker/Torchserve, use CUDA_VISIBLE_DEVICES="" (according to StackOverflow) or CUDA_VISIBLE_DEVICES=-1 (according to SegNeXt's GitHub page).
# When running it directly, run with --device cpu option.
# ------------------------------------------
 
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]
