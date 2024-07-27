let session, model_loaded = false;
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const maxSize = 2560;
const loader = document.getElementsByClassName("loader")[0];
const output = document.getElementById("output");
const saveButton = document.getElementById("save");
const a  = document.createElement('a');

async function loadmodel(model_name){
  try{
    console.time("Time taken for loading model");
    path = "./models/" + model_name + ".onnx";
    session = await ort.InferenceSession.create(path);
    console.timeEnd("Time taken for loading model");
    model_loaded = true;
    document.getElementById("model-state").innerHTML = "Model Loaded"
  } catch(e) {
    console.log(e);
  }
}

function handleDragOver(event) {
  event.preventDefault();
  event.dataTransfer.dropEffect = "copy";
  document.getElementById("drop-area").classList.add("highlight");
}

function handleDragEnter(event) {
  event.preventDefault();
  document.getElementById("drop-area").classList.add("highlight");
}

function handleDragLeave(event) {
  document.getElementById("drop-area").classList.remove("highlight");
}

function handleDrop(event) {
  event.preventDefault();
  document.getElementById("drop-area").classList.remove("highlight");
  const files = event.dataTransfer.files;
  handleFiles(files);
}

function handleFiles(files) {
  const file = files[0];
  if (!model_loaded) {
    alert("Please wait till model is loaded");
    return;
  }
  if (!file.type.startsWith("image/")) {
    alert("Please select an image");
    return;
  }
  loader.style.visibility = "visible";
  const reader = new FileReader();
  reader.onload = function () {
    let img = new Image();
    img.onload = function () {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      imgWidth = img.width;
      imgHeight = img.height;
      max = Math.max(imgWidth, imgHeight);
      let ratio = 1.0;
      if (max > maxSize) {
        ratio = maxSize / max;
      }
      width = Math.ceil(imgWidth * ratio);
      height = Math.ceil(imgHeight * ratio);
      canvas.width = width;
      canvas.height = height;
      ctx.drawImage(img, 0, 0, width, height);
      style_img(ctx, width, height);
    }
    img.src = event.target.result;
  }
  reader.readAsDataURL(file);
}

async function style_img(ctx, width, height){
  imageData = ctx.getImageData(0, 0, width, height).data;
  tensorData = new Float32Array(width * height * 3);
  const step = width * height

  for(let y = 0; y < height; y++){
    for(let x = 0; x < width; x++){
      const [di, si] = [y * width + x, (y * width + x) * 4];
      tensorData[di] = imageData[si + 0] / 255;
      tensorData[di + step] = imageData[si + 1] / 255;
      tensorData[di + step * 2] = imageData[si + 2] / 255;
    }
  }
  console.log(width, height)
  imgTensor = new ort.Tensor("float32", tensorData, [1, 3, height, width]);
  const feed = {input: imgTensor};
  console.time("Time taken for Inference");
  let result = await session.run(feed);
  console.timeEnd("Time taken for Inference");
  output.src = result.output.toDataURL();
  output.style.display = "block";
  saveButton.style.display = "block";
  loader.style.visibility = "hidden";
}

function downloadImage(event) {
  a.href = output.src;
  a.download = "styled_image.jpg";
  a.click()
}
