from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO('yolov8n.pt')

# 테스트 이미지 경로
image_path = './datasets/valid/images/img_1000_jpg.rf.a33f78585bd5340c7d146bb6441df603.jpg'

# 이미지 로드
image = cv2.imread(image_path)

# 객체 검출
results = model(image)

# 결과 시각화
results.show()
