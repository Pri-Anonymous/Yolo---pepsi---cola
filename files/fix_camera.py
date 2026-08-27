import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run detection on this frame
    results = model.predict(
        source=frame,
        conf=0.25,       # lowered threshold for webcam conditions
        classes=[0, 1],
        verbose=False    # suppress per-frame text spam in terminal
    )

    # Draw boxes/masks/labels directly onto the frame
    annotated_frame = results[0].plot()

    # Show the live window
    cv2.imshow("Pepsi/Coke Detector", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()