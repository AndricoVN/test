import os
import cv2
import numpy as np
import onnxruntime as ort
import pytesseract
import re

# Cấu hình đường dẫn Tesseract OCR (cập nhật nếu cần)
pytesseract.pytesseract.tesseract_cmd = r"D:\machine learning\detect_models\g\tesseract.exe"  # Nếu dùng Windows

# Load mô hình YOLOv8
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

# Đọc ảnh
image_path = "dataset/images/val/image4.jpg"
image = cv2.imread(image_path)
h, w, _ = image.shape

# Tiền xử lý ảnh cho YOLOv8
image_resized = cv2.resize(image, (960, 960))
image_tensor = np.transpose(image_resized[:, :, ::-1].astype(np.float32) / 255.0, (2, 0, 1))[None]

# Dự đoán bounding boxes
outputs = session.run(None, {session.get_inputs()[0].name: image_tensor})[0].squeeze()
x_c, y_c, w_b, h_b, scores = outputs

# Lọc box theo threshold
mask = scores > 0.05
x1, y1 = (x_c - w_b / 2)[mask] * (w / 960), (y_c - h_b / 2)[mask] * (h / 960)
x2, y2 = (x_c + w_b / 2)[mask] * (w / 960), (y_c + h_b / 2)[mask] * (h / 960)
filtered_scores = scores[mask]

if len(filtered_scores) > 0:
    max_idx = np.argmax(filtered_scores)
    x1, y1, x2, y2 = map(int, [x1[max_idx], y1[max_idx], x2[max_idx], y2[max_idx]])

    # Cắt vùng bounding box
    roi = image[y1:y2, x1:x2]

    # Tiền xử lý ảnh cho OCR
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Nhận diện số bằng Tesseract OCR
    custom_oem_psm_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'  # Chỉ nhận số
    mssv = pytesseract.image_to_string(roi, config=custom_oem_psm_config)
    
    mssv = "".join(re.findall(r"\d+", mssv))

    mssv = mssv.strip()
    
    mssv = mssv[-7:]
    # Vẽ bounding box lên ảnh gốc
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, mssv, (x1 + 28, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

image = cv2.resize(image, (540, 960))

cv2.imshow("Detected Image", image)
cv2.waitKey()
cv2.destroyAllWindows()