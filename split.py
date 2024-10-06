import os
import random
import shutil
from glob import glob

random.seed(3)
train_rate = 0.7

os.system("rm -rf data/split/images")
os.makedirs("data/split/images/train")
os.makedirs("data/split/images/val")

os.system("rm -rf data/split/labels")
os.makedirs("data/split/labels/train")
os.makedirs("data/split/labels/val")

paths = glob("data/labels/*")
random.shuffle(paths)
len_paths = len(paths)
for i, label_path in enumerate(paths):
    if label_path.endswith("classes.txt"):
        continue

    image_path = label_path.replace("labels", "images").replace(".txt", ".png")

    if (i + 1) / len_paths <= train_rate:
        image_save_path = image_path.replace("images", "split/images/train")
        label_save_path = label_path.replace("labels", "split/labels/train")
    else:
        image_save_path = image_path.replace("images", "split/images/val")
        label_save_path = label_path.replace("labels", "split/labels/val")

    shutil.copy(image_path, image_save_path)
    shutil.copy(label_path, label_save_path)
