print("RUNNING ENTRYPOINT.PY\n\n\n")

import os
import time
import subprocess
from datetime import datetime

import warnings

root = r"/home/videos"
segnext = r"/home/SegNeXt"
device = "cpu"

models = [
    ("pspnet_r50-d8", "pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth", "pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py"), # Slower model, higher accuracy
    ("pspnet_r18b-d8", "pspnet_r18b-d8_512x1024_80k_cityscapes_20201226_063116-26928a60.pth", "pspnet/pspnet_r18b-d8_512x1024_80k_cityscapes.py"), # Faster model, lower accuracy
    # No model for night time, apparently?
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
                filesize = os.path.getsize(os.path.join(root, "segmented", file))
                if filesize == 0:
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t- ERROR: Output file {os.path.join(root, 'segmented', file)} is size 0!")
                else:
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t- Output file {os.path.join(root, 'segmented', file)} is size {filesize} bytes. Total time: {str(delta)}")
                s2 = subprocess.run(["mv", os.path.join(root, "original/todo", file), os.path.join(root, "original/done", file)])
                inference_file = (file.split(".")[0] if "." in file else file) + ".inferences"
                s3 = subprocess.run(["mv", os.path.join(root, "original/todo", inference_file), os.path.join(root, "original/done", inference_file)])
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t- Moved input file {os.path.join(root, 'original/todo', file)} to {os.path.join(root, 'original/done', file)}.")
    time.sleep(30)
