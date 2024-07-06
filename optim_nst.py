import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 0.03
content_weigth = 1
style_weight = 1_00_00_000
epochs = 500
content_path = "content.jpg"

class NST(nn.Module):

    def __init__(self):
        super(NST, self).__init__()
        self.layers = [0, 5, 10, 19, 28]
        self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:self.layers[-1]+1]

    def forward(self, x):
        features = []
        for idx, layer in enumerate(self.vgg):
            x = layer(x)

            if idx in self.layers:
                features.append(x)
        return features

def get_image(image_path):
    image = Image.open(image_path)
    tranform = transforms.Compose([transforms.Resize(500), transforms.ToTensor()])
    image = tranform(image).unsqueeze(0)
    return image

nst = NST().to(device).eval()

style = get_image("Starry_night_4k.jpg").to(device)
content = get_image(content_path).to(device)
generated = content.clone().requires_grad_(True)
optim = optim.Adam([generated], lr=lr)

for epoch in tqdm(range(epochs)):
    generated_features = nst(generated)
    content_features = nst(content)
    style_features = nst(style)

    stlye_loss = 0
    content_loss = 0

    for generated_feature, content_feature, style_feature in zip(generated_features, content_features, style_features):
        content_loss += torch.mean((generated_feature - content_feature) ** 2)

        b, c, h, w = generated_feature.shape
        generated_gram = generated_feature.view(c, h*w).mm(generated_feature.view(c, h*w).t()) / ( c * h * w)

        b, c, h, w = style_feature.shape
        style_gram = style_feature.view(c, h*w).mm(style_feature.view(c, h*w).t()) / (c * h * w)

        stlye_loss += torch.mean((generated_gram - style_gram) ** 2)

    total_loss = (content_weigth * content_loss) + (style_weight * stlye_loss)

    optim.zero_grad()
    total_loss.backward()
    optim.step()


save_image(generated, "generated.jpg")

generated = cv2.imread("generated.jpg")
dest_gray = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)
b, c, h, w = content.shape
content = cv2.imread(content_path)
content = cv2.resize(content, (w, h), interpolation=cv2.INTER_AREA)
src_yiq = cv2.cvtColor(content, cv2.COLOR_BGR2YCrCb)
src_yiq[...,0] = dest_gray
generated = cv2.cvtColor(src_yiq, cv2.COLOR_YCrCb2BGR)
cv2.imwrite("generated_color.jpg", generated)
# save_image(generated, "generated.jpg")
