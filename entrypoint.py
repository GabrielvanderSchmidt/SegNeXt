import os
import time
import subprocess

root = r"/home/videos"
segnext = r"/home/SegNeXt"
conf_model = "pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py"
check_model = "pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth"

while True:
   for file in os.listdir(os.path.join(root, "original/todo")):
       print(f"{now.strftime('%Y-%m-%d %H:%M:%S')}\t- Checking file {file}")
       if file.endswith(".avi") and os.isfile(os.path.join(root, "original/todo", file)):
           print(f"{now.strftime('%Y-%m-%d %H:%M:%S')}\t- Segmenting {os.path.join(root, 'original/todo', file)}...")
           s1 = subprocess.run("python3", os.path.join(segnext, "demo/video_demo.py"), os.path.join(root, "original/todo", file), os.path.join(segnext, "configs", model), os.path.join(segnext, "checkpoints", check_model), '--device="cpu"', f"--output-file={os.path.join(root, 'segmented', file)}")
           filesize = os.path.getsize(os.path.join(root, "segmented", file))
           if filesize == 0:
               print("{now.strftime('%Y-%m-%d %H:%M:%S')}\t- ERROR: Output file {os.path.join(root, 'segmented', file)} is size 0!")
           else:
               print("{now.strftime('%Y-%m-%d %H:%M:%S')}\t- Output file {os.path.join(root, 'segmented', file)} is size {filesize} bytes.")
           s2 = subprocess.run("mv", os.path.join(root, "original/todo", file), os.path.join(root, "original/done", file))
           print("{now.strftime('%Y-%m-%d %H:%M:%S')}\t- Moved input file {os.path.join(root, 'original/todo', file)} to {os.path.join(root, 'original/done', file)}.")
   time.sleep(30)
