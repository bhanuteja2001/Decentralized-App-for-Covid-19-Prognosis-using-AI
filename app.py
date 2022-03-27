from flask import Flask, render_template, request,jsonify
from urllib.request import urlopen as uReq
import io

import numpy as np
from Ensemble import Ensemble
from PIL import Image
import cv2

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')

    if not file:
        return

    img_bytes = file.read()
    img = prepare_image(img_bytes)

    return jsonify(prediction=img)

def prepare_image(img):
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    #img = Image.open(io.BytesIO(img))
    obj = Ensemble()
    pred = obj.Result(img)
    return pred








if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8000, debug=True)
	app.run(debug=True)