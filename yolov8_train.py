from multiprocessing import freeze_support
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

if __name__=='__main__':
    freeze_support()
    model.train(data='data.yaml',epochs=100)