from flask import Flask, request, send_file
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import io

app = Flask(__name__)

# Load mô hình YOLOv8
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

def process_image(image):
    image_np = np.array(image)
    h, w, _ = image_np.shape
    
    # Tiền xử lý ảnh cho YOLOv8
    image_resized = cv2.resize(image_np, (960, 960))
    image_tensor = np.transpose(image_resized[:, :, ::-1].astype(np.float32) / 255.0, (2, 0, 1))[None]
    
    # Dự đoán bounding boxes
    outputs = session.run(None, {session.get_inputs()[0].name: image_tensor})[0].squeeze()
    x_c, y_c, w_b, h_b, scores = outputs
    
    # Lọc box theo threshold
    mask = scores > 0.05
    x1, y1 = (x_c - w_b / 2)[mask] * (w / 960), (y_c - h_b / 2)[mask] * (h / 960)
    x2, y2 = (x_c + w_b / 2)[mask] * (w / 960), (y_c + h_b / 2)[mask] * (h / 960)
    filtered_scores = scores[mask]
    
    # Vẽ bounding box nếu có ít nhất một vùng được phát hiện
    if len(filtered_scores) > 0:
        for i in range(len(filtered_scores)):
            x1_i, y1_i, x2_i, y2_i = map(int, [x1[i], y1[i], x2[i], y2[i]])
            cv2.rectangle(image_np, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
    
    return image_np

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return {"error": "No image uploaded"}, 400
    
    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")
    
    processed_image = process_image(image)
    
    # Lưu ảnh đã xử lý vào buffer
    _, img_encoded = cv2.imencode('.jpg', processed_image)
    img_bytes = io.BytesIO(img_encoded.tobytes())
    
    return send_file(img_bytes, mimetype='image/jpeg', as_attachment=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
