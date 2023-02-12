print("RUNNING ENTRYPOINT.PY\n\n\n")

import os
import time
import subprocess
from datetime import datetime

import warnings

import torch

print(f"Cuda is available? {torch.cuda.is_available()}")
print(f"Devices found: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print("All devices:")
for index, name in [(index, torch.cuda.get_device_name(index)) for index in range(torch.cuda.device_count())]:
    print(f"\t{index}: {name}")

root = r"/home/videos"
segnext = r"/home/SegNeXt"
device = os.environ["CUDA_VISIBLE_DEVICES"]
device = f"cuda:{device}" if (isinstance(device, str) and len(device) == 1) else "cpu"
print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t- Device set is {device}.\n\n\n")

models = [
    ("pspnet_r50-d8", "pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth", "pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py"), # fps=4.07, mIoU=77.85
    ("pspnet_r101-d8", "pspnet_r101-d8_512x1024_40k_cityscapes_20200604_232751-467e7cf4.pth", "pspnet/pspnet_r101-d8_512x1024_40k_cityscapes.py"), # fps=2.68, mIoU=78.34
    ("pspnet_r18-d8", "pspnet_r18-d8_512x1024_80k_cityscapes_20201225_021458-09ffa746.pth", "pspnet/pspnet_r18-d8_512x1024_80k_cityscapes.py"), # fps=15.71, mIoU=74.87
    ("pspnet_r18b-d8", "pspnet_r18b-d8_512x1024_80k_cityscapes_20201226_063116-26928a60.pth", "pspnet/pspnet_r18b-d8_512x1024_80k_cityscapes.py"), # fps=16.28, mIoU=74.23
    ("pspnet_r50-d32", "pspnet_r50-d32_512x1024_80k_cityscapes_20220316_224840-9092b254.pth", "pspnet/pspnet_r50-d32_512x1024_80k_cityscapes.py"), # fps=15.21, mIoU=73.88
    ("pspnet_r50b-d32_rsb", "pspnet_r50-d32_rsb-pretrain_512x1024_adamw_80k_cityscapes_20220316_141229-dd9c9610.pth", "pspnet/pspnet_r50-d32_rsb-pretrain_512x1024_adamw_80k_cityscapes.py"), # fps=16.08, mIoU=74.09
    ("pspnet_r50b-d32", "pspnet_r50b-d32_512x1024_80k_cityscapes_20220311_152152-23bcaf8c.pth", "pspnet/pspnet_r50b-d32_512x1024_80k_cityscapes.py"), # fps=15.41, mIoU=72.61
    ]

while True:
    for file in os.listdir(os.path.join(root, "original/todo")):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t- Checking file {file}")
        
        for model_name, model_check, model_conf in models:
            if file.endswith(".avi") and os.path.isfile(os.path.join(root, "original/todo", file)):
                start = datetime.now()
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t- Segmenting {os.path.join(root, 'original/todo', file)} with {model_name}...")

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s1 = subprocess.run(["python3", os.path.join(segnext, "demo/video_demo.py"), os.path.join(root, "original/todo", file), os.path.join(segnext, "configs", model_conf), os.path.join(segnext, "checkpoints", model_check), f'--device={device}', f"--output-file={os.path.join(root, 'segmented', file)}"])

                end = datetime.now()
                delta = end - start
                filesize = None if not os.path.isfile(os.path.join(root, "segmented", file)) else os.path.getsize(os.path.join(root, "segmented", file))
                inference_file = (file.split(".")[0] if "." in file else file) + ".inferences"
                if filesize is None:
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t- ERROR: Model {model_name} did not produce the expected output file {os.path.join(root, 'segmented', file)}!")
                    continue # Replace with exit() in the future?
                elif filesize == 0:
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t- ERROR: Output file {os.path.join(root, 'segmented', file)} is size 0!")
                    continue # Replace with exit() in the future?
                else:
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t- Output file {os.path.join(root, 'segmented', file)} is size {filesize} bytes. Total time: {str(delta)}")
                filename_avi = file.split(".")[0] + f"-{model_name}-{str(delta)}." + file.split(".")[1]
                filename_pkl = filename_avi.split(".")[0] + ".inferences"
                s2 = subprocess.run(["mv", os.path.join(root, "segmented", file), os.path.join(root, "segmented", filename_avi)])
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t- Moved output file {os.path.join(root, 'segmented', file)} to {os.path.join(root, 'segmented', filename_avi)}.")
                if os.path.isfile(inference_file) is False:
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t- ERROR: Model {model_name} did not produce the expected output file {os.path.join(root, 'segmented', inference_file)}!")
                    continue
                s3 = subprocess.run(["mv", os.path.join(root, "segmented", inference_file), os.path.join(root, "segmented", filename_pkl)])
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t- Moved input file {os.path.join(root, 'segmented', inference_file)} to {os.path.join(root, 'segmented', filename_pkl)}.")
                
        s4 = subprocess.run(["mv", os.path.join(root, "original/todo", file), os.path.join(root, "original/done", file)])
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t- Moved input file {os.path.join(root, 'original/todo', file)} to {os.path.join(root, 'original/done', file)}")
    time.sleep(30)
