from flask import Flask, request, jsonify
import onnxruntime
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# ort_session = onnxruntime.InferenceSession("model.onnx")

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/vangoghify', methods=['POST'])
def vangoghify():
    print("here")
    input_data = np.array(request.json['input_data'], dtype=np.float32)

    # Perform inference
    # output = ort_session.run(None, {'input.1': input_data})
    print("here also")

    # Process the output data (e.g., scaling, normalization)
    # output_image_data = output[0].squeeze() 
    # output_image_data = output_image_data.transpose(1, 2, 0)
    # output_image_data = (output_image_data * 0.5 + 0.5).clip(0.0, 1.0)
    # output_image_data = (output_image_data * 255.0).astype(np.uint8)

    # print(output_image_data.shape)

    # Prepare the output image for sending to the frontend
    # output_image = output_image_data.tobytes()
    # output_image_base64 = base64.b64encode(output_image).decode('utf-8')

    # return jsonify({'output_image': output_image_base64})

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)
