import torch
from model import *
from config import *
from safetensors.torch import load_model
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np

model = StyleTransfer().to(device)
load_model(model, model_weights)
model.eval()

input_transform = transform = transforms.Compose([
    transforms.ToTensor()
])

output_transform = transform = transforms.Compose([
    transforms.ToPILImage()
])

cam = cv2.VideoCapture(0)

while(True):
    ret, frame = cam.read()
    frame = Image.fromarray(frame)
    frame = input_transform(frame).unsqueeze(0).to(device)

    result = output_transform(model(frame).squeeze(0))
    cv2.imshow('Neural Style Transfer', np.array(result))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
