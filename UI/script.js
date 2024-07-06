// load model

let session;
var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');
let imgTensor, imageData, tensorData, result;

async function loadmodel(){
  try{
    console.time("loading model")
    session = await ort.InferenceSession.create('./models/rain_princess.onnx')
    console.timeEnd("loading model")
  } catch(e) {
    console.log(e)
  }
}

function inputImage(input){
  var reader  = new FileReader()
  var file = input.files[0]
  reader.onload = function () {
    var img = new Image()
    img.onload = function(event){
      var maxWidth = canvas.width;
      var maxHeight = canvas.height;
      var width = img.width;
      var height = img.height;
      var ratio = Math.min(maxWidth / width, maxHeight / height);
      var width = Math.ceil(width * ratio)
      var height = Math.ceil(height * ratio)
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, width, height);
      style_img(ctx, width, height);
    }
    img.src = event.target.result
  }
  reader.readAsDataURL(file)
}

async function style_img(ctx, width, height){
  imageData = ctx.getImageData(0, 0, width, height).data;
  tensorData = new Float32Array(width * height * 3);
  const step = width * height

  for(let y = 0; y < height; y++){
    for(let x = 0; x < width; x++){
      const [di, si] = [y * width + x, (y * width + x) * 4];
      tensorData[di] = imageData[si + 0] / 255
      tensorData[di + step] = imageData[si + 1] / 255
      tensorData[di + step * 2] = imageData[si + 2] / 255
    }
  }
  console.log(width, height)
  imgTensor = new ort.Tensor("float32", tensorData, [1, 3, height, width])
  const feed = {input: imgTensor}
  console.time("Inference Started")
  result = await session.run(feed)
  console.timeEnd("Inference Started")
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.putImageData(result.output.toImageData(), 0, 0)
}

function resizecanvas(){
  canvas.width = window.innerWidth-50;
  canvas.height = window.innerHeight-50;
  console.log(canvas.width, canvas.height)
}

loadmodel()
resizecanvas()
