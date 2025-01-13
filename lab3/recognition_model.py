import os
import random
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from torchvision import models
# Датасет без разметки берём отсюда - https://www.kaggle.com/datasets/devdgohil/the-oxfordiiit-pet-dataset/code
# Используем Faster R-CNN
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

image_folder = "test_images"

image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

random_image_name = random.choice(image_files)

image_path = os.path.join(image_folder, random_image_name)
image = Image.open(image_path)

transform = transforms.Compose([
    transforms.ToTensor()
])

input_image = transform(image).unsqueeze(0)

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

with torch.no_grad():
    prediction = model(input_image)

boxes = prediction[0]['boxes'].numpy()
scores = prediction[0]['scores'].numpy()

plt.imshow(image)

threshold = 0.5
for i in range(len(scores)):
    if scores[i] > threshold:
        box = boxes[i]
        x_min, y_min, x_max, y_max = box
        plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, edgecolor='r', facecolor='none', lw=3))

plt.show()

