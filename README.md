## Usage

The model training was done using `train.py`. The configuration is secified in `config.py`.

Model conversion can be done using `onnx-conversion.py`. The model configuration for this script is specified into `config.py`.

The UI can be run using python http server using the following command inside the UI folder: -
```
python -m http.server
```

Then open [localhost:8000](http://localhost:8000) and try out the model. You can check the model loading and infernce time inside the console.

---
## References
https://www.pexels.com/photo/lake-and-mountain-417074/ - Content Image

https://github.com/lengstrom/fast-style-transfer/tree/master/examples/style - Style Image
