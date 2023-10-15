import onnxruntime
import numpy as np
from PIL import Image

model_path = "model.onnx"
ort_session = onnxruntime.InferenceSession(model_path)

input_image = Image.open("input_image.jpg")
input_data = np.array(input_image, dtype=np.float32)
input_data = (input_data / 255.0 - 0.5) / 0.5  # Normalize
input_data = input_data.transpose(2, 0, 1)  # Transpose to (channels, height, width)
input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

# Perform inference
output = ort_session.run(None, {'input.1': input_data})

# Process the output data (e.g., scaling, normalization)
output_image = output[0]  # Assuming a single output
output_image = output_image.squeeze()  # Remove batch dimension

# Save or display the output image
output_image = output_image.transpose(1, 2, 0)  # Transpose to (height, width, channels)
output_image = output_image*0.5 + 0.5
output_image = np.uint8(output_image*255)  # Convert to uint8 if needed
output_image = Image.fromarray(output_image)
output_image.save("output10.jpg")
output_image.show()
