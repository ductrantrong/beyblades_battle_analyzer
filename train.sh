python split.py && \
yolo detect train data=data/config.yaml model=yolo11n.pt epochs=100 imgsz=320 save_period=1 dropout=0.1 batch=32