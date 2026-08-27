from ultralytics import YOLO

def main():
    model = YOLO('yolo11n-seg.pt')

    model.train(
        data="D:\\yolo-pepsi-coke\\data.yaml",
        epochs=60,
        imgsz=480,
        batch=8,
        name='yolo-pepsi-coke-seg'
    )

if __name__ == '__main__':
    main()