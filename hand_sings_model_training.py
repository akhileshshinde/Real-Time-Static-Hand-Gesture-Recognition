import cv2
import os


train_image_dir = "/home/extra_space/akhilesh/Bhatiyaniai/dataset_split/train/images"
val_image_dir = "/home/extra_space/akhilesh/Bhatiyaniai/dataset_split/valid/images"


# Create data.yaml
data_yaml = """train: /home/extra_space/akhilesh/Bhatiyaniai/dataset_split/train/images
val: /home/extra_space/akhilesh/Bhatiyaniai/dataset_split/valid/images


nc: 4  # Number of classes (1 for person)
names: ['open_palm', 'fist', 'peace_sign', 'thumbs_up']  # Class names
"""

# Save the data.yaml file
with open("data.yaml", "w") as f:
    f.write(data_yaml)

print("data.yaml file created")


train_command = f"""
python train.py --img 640 --batch 16 --epochs 100 --data data.yaml \
--weights /home/akhilesh/yolov5_new/yolov5s.pt --project runs/train --name hand_sign \
--exist-ok --device 0 --workers 4
"""

print("To train the model, run the following command:")
print(train_command)