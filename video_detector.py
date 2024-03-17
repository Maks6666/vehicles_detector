
import cv2
from ultralytics import YOLO

# cap = cv2.VideoCapture("videos/tank.mp4")
# cap = cv2.VideoCapture("videos/apc.mp4")
cap = cv2.VideoCapture("videos/afvq.mp4")



ret, frame = cap.read()
object_count = 0

model = YOLO("best.pt")
names = model.names
threshold = 0.5

class_counts = {names[class_id].upper(): 0 for class_id in names.keys()}



while ret:

    results = model(frame)[0]
    count = 0
    object_count = 0
    index = 0

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # ключи для словаря class_counts
        class_name = names[int(class_id)].upper()

        if score > threshold:
            count += 1
            index += 1
            object_count += 1
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)



            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.circle(frame, (x_center, y_center), 3, (0, 0, 255), thickness=cv2.FILLED)
            cv2.putText(frame, f"{index}: {names[int(class_id)].upper()}", (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)



    cv2.rectangle(frame, (8, 15), (350, 25), (0, 0, 0), 20)
    cv2.putText(frame, f"Total object count: {object_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Video", frame)

    # out.write(frame)
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()