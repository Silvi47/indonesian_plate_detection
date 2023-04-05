import argparse
from ultralytics import YOLO
import cv2
import pandas as pd
from ultralytics.yolo.utils.files import increment_path
from datetime import datetime
from PIL import Image
from pathlib import Path

now = datetime.now()

def imgwrite(img, file=Path('image.jpg')):
    file.parent.mkdir(parents=True, exist_ok=True)
    f = str(increment_path(file).with_suffix('.jpg'))
    Image.fromarray(img[..., ::-1]).save(f, quality=95, subsampling=0)

model = YOLO("model_best2.pt")

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True, help="Path to the file")
args = vars(ap.parse_args())

results = model.predict(source=args["file"], save=False, show=False, save_crop=False) # show_crop, save_txt
# Source: cv2.imread('im.jpg')[:,:,::-1], np.zeros((640,1280,3)), torch.zeros(16,3,320,640)

class_list = ["plate"]

for result in results:
    bbox_idx = result.boxes.xyxy.tolist()
    img = result.orig_img
    print("bbox: ", bbox_idx)

    for bbox in bbox_idx:
        x3, y3, x4, y4 = bbox
        cropped_img = img[int(round(y3)):int(round(y4)), int(round(x3)):int(round(x4))]
        # Save cropped image
        # imgwrite(cropped_image,"img.jpg")
