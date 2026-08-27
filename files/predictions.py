
from ultralytics import YOLO

model = YOLO("D://yolo-pepsi-coke//best.pt")

model.predict(source=0,
  save=True,
    line_width=3,
  save_crop=True,
  show_labels=True,
  show_conf=True,
  conf=0.67,
  show=True)