from model import *
from config import *
from safetensors.torch import load_model
import torch.onnx
import onnx
import onnxruntime
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def infer_onnx(onnx_path, test_image):
    session = onnxruntime.InferenceSession(onnx_path)
    input_transform = transform = transforms.Compose([
        transforms.Resize(1024),
        transforms.ToTensor()
    ])

    img = Image.open(test_image_path)
    img = input_transform(img).unsqueeze(0).numpy()
    output = session.run(None, {'input': img})[0][0]
    result = np.clip(output, 0, 1)*255
    result = result.transpose(1,2,0).astype(np.uint8)
    print(result.min(), result.max(), result.shape)
    Image.fromarray(result).show()

model = StyleTransfer()
load_model(model, model_path)
model.eval()

dummy_input = torch.randn(1, 3, 500, 500)
dynamic_axes = {"input": {0: "batch_size", 2: "imgx", 3: "imgy"},
                "output": {0: "batch_size", 2: "imgx", 3: "imgy"}}

torch.onnx.export(model, dummy_input, onnx_path, opset_version=15, verbose=False, input_names=["input"], output_names=["output"], dynamic_axes=dynamic_axes)

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

infer_onnx(onnx_path, test_image_path)
