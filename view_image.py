from flask import Flask, render_template
from glob import glob
import base64

app = Flask(__name__)

html_code = """
<html>
<head>
    <title>Styled Image</title>
</head>
<body>
    <center>
    <img id="img" src="data:image/jpg;base64, " alt="Image">
    <center>
</body>
<script>
const image = document.getElementById("img");

async function updateImage(){
    const response = await fetch('/image');
    const base64 = await response.text();
    image.src = "data:image/jpg;base64, "+base64
}

setInterval(updateImage, 1000);
</script>
</html>
"""

@app.route("/")
def index():
    return html_code

@app.route("/image")
def image():
    images = glob("styled_images/2-*.jpg")
    image = max(images, key=lambda x: int(x.split("-")[-1].split(".")[0]))
    print(image)
    with open(image, "rb") as file:
        encoded_string = base64.b64encode(file.read())
        base64_image = encoded_string.decode("utf-8")
        return base64_image

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
