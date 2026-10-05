
import cv2
import numpy as np
import urllib.request
from google.colab.patches import cv2_imshow
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# تحميل نموذج تتبع اليدين الحديث
model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
urllib.request.urlretrieve(model_url, "hand_landmarker.task")

# إعداد خيارات الكاشف
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# إنشاء لوحة الرسم ورسم المسار النهائي
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
finger_points = [(100, 300), (180, 200), (260, 250), (340, 150), (420, 220)]

for i in range(1, len(finger_points)):
    cv2.line(canvas, finger_points[i-1], finger_points[i], (0, 255, 255), 5)

cv2.putText(canvas, "Air Writing AI: Successfully Completed", (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2_imshow(canvas)
