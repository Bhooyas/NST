import torch

data_dir = "./data"
style_image_path = "./images/style/rain_princess.jpg"
test_image_path = "./images/content.jpg"
image_checkpoints = "./styled_images"
# model_weights = "./models/starry_nigth.safetensors"
model_weights = "./models/rain_princess.safetensors"

image_size = 256
batch_size = 4
lr = 0.001
content_weigth = 1
style_weight = 10_00_00_000
save_image_interval = 100
epochs = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

# Onnx conversion
model_path = "./models/rain_princess.safetensors"
onnx_path = "./UI/models/rain_princess.onnx"
