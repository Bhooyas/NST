import torch

data_dir = "./data" # The directory of training images
style_image_path = "./images/style/rain_princess.jpg" # The path of style image
test_image_path = "./images/content.jpg" # The path for test image
image_checkpoints = "./styled_images" # Directory to store images while training
model_weights = "./models/starry_night.safetensors" # Path to save the model weigths
# model_weights = "./models/rain_princess.safetensors"

image_size = 256 # The image size for training (Image size doesn't matter for inference)
batch_size = 4 # Batch size
lr = 0.001 # learning rate
content_weigth = 1 # The loss multiple for content loss. Tweak this if orignal image is not being created
style_weight = 10_00_00_000 # The loss multiple for stlye loss. Tweak this if the style is not being transfered properly.
save_image_interval = 100 # After how many steps to save the test image and model weigths
epochs = 2 # Epochs
device = "cuda" if torch.cuda.is_available() else "cpu" # Device to train on

# Onnx conversion
# model_path = "./models/starry_night.safetensors"
# onnx_path = "./UI/models/starry_night.onnx"
model_path = "./models/rain_princess.safetensors" # Path of the model to convert
onnx_path = "./UI/models/rain_princess.onnx" # Path where the converted model has to be saved
