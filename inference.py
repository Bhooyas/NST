import torch
from model import *
from config import *
from safetensors.torch import load_model
import torchvision.transforms as transforms
from PIL import Image

model = StyleTransfer().to(device)
load_model(model, model_weights)
model.eval()

input_transform = transform = transforms.Compose([
    transforms.Resize(1024),
    transforms.ToTensor()
])

output_transform = transform = transforms.Compose([
    transforms.ToPILImage()
])

img = Image.open(test_image_path)
img = input_transform(img).unsqueeze(0).to(device)
output = torch.clamp(model(img), 0, 1)
output = output_transform(output.squeeze(0))
output.show()
