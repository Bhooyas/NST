# Neural Style Transformer

![NST](https://socialify.git.ci/Bhooyas/NST/image?font=KoHo&language=1&name=1&owner=1&pattern=Circuit%20Board&stargazers=1&theme=Auto)

Pytorch implementation for neural style transfer.

## Inference the models

### Using Webcam

The first step would be to clone the project using the following command: -
```
git clone https://github.com/Bhooyas/NST.git
```

The next step is to navigate inside the directory and install requirements: -
```
cd NST/UI
pip install -r requirements.txt
```

Thereafter we will run the following command for webcam inference: -
```
python inference_webcam.py
```

Press `q` key to exit from the prview.

### Static Images

The first step would be to clone the project using the following command: -
```
git clone https://github.com/Bhooyas/NST.git
```

The next step is to navigate inside the UI folder: -
```
cd NST/UI
```

In the next step we just run python http server at this location to host the html application. The command is as follows: -
```
python -m http.server
```

Then open [localhost:8000](http://localhost:8000) and try out the model. You can check the model loading and infernce time inside the console.

You can test the website for any image. You can get random [images](https://picsum.photos/450/300) using [Lorem Picsum](https://picsum.photos) website for testing.

## Training the NST Model

The first step would be to clone the project using the following command: -
```
git clone https://github.com/Bhooyas/NST.git
```

The next step would be to install the requirements: -
```
cd NST
pip install -r requirements.txt
```

The next step will be to get the data. You can run the follwing shell script to get the data: -
```
./get_dataset.sh
```
You can run it on git bash in windows.

The configuration for the model can be found and edited in the `config.py`.

The next step is to train the model using the following command: -
```
python train.py
```

You can now infer from the model from python using the following command: -
```
python inference.py
```

You can use this just created model for [webcam inference](#using-webcam)

If you want to add the inference of the model using UI. Then we need to do some additional steps. The first is to convert the model from `safetenor` to `onnx`. We achieve this using the following command: -

```
python onnx-conversion.py
```

The above command wil create onnx file in the `UI/models` directory. The step is to create the html file for the model. Just make a copy `rain_princess.html` and rename it to the `model_name.html`. The replace the image name at `line 17` and model name at `line 33`. The next step will be to edit and add the anchor tag in `index.html`. The follow the [Inference Section](#inference-the-models)


## References
https://www.pexels.com/photo/lake-and-mountain-417074/ - Content Image

https://github.com/lengstrom/fast-style-transfer/tree/master/examples/style - Style Image
