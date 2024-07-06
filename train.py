import torch
from model import *
from config import *
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image
from safetensors.torch import save_model, load_model

model = StyleTransfer().to(device)
# load_model(model, model_weights)
vgg = VGG().eval().to(device)

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomCrop(image_size),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(data_dir, transform)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=lr)
style_image = Image.open(style_image_path)
style_image = transform(style_image)

test_image = Image.open(test_image_path)
test_image = test_transform(test_image).unsqueeze(0).to(device)

for epoch in range(epochs):
    pbar = tqdm(dataloader)
    pbar.set_description(f"Epoch {epoch+1}/{epochs}")
    count = 0
    idx = 0
    for (img, _) in pbar:
        batch_len = len(img)
        count += batch_len
        idx += 1

        style_img = style_image.repeat(batch_len, 1, 1, 1).to(device)
        img = img.to(device)
        gen = model(img)
        img_features = vgg(img)
        gen_features = vgg(gen)
        stlye_features = vgg(style_img)

        stlye_loss = 0
        content_loss = 0

        for gen_feature, img_feature, style_feature in zip(gen_features, img_features, stlye_features):
            content_loss += torch.mean((gen_feature - img_feature) ** 2)

            b, c, h, w = gen_feature.shape
            gen_gram = gen_feature.view(b, c, h*w).bmm(gen_feature.view(b, c, h*w).transpose(1, 2)) / (b * c * h * w)

            b, c, h, w = style_feature.shape
            style_gram = style_feature.view(b, c, h*w).bmm(style_feature.view(b, c, h*w).transpose(1, 2)) / (b * c * h * w)

            stlye_loss += torch.mean((gen_gram - style_gram) ** 2)

        total_loss = (content_weigth * content_loss) + (style_weight * stlye_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (idx % save_image_interval) == 0:
            temp = model(test_image)
            save_image(temp,  f"{image_checkpoints}/{epoch+1}-{idx}.jpg")
            save_model(model, model_weights)

temp = model(test_image)
save_image(temp,  f"{image_checkpoints}/trained_model.jpg")
save_model(model, model_weights)
