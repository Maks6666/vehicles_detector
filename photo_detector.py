import os
import cv2
from ultralytics import YOLO


# image_path = "images/apc.jpg"
# image_path = "images/ifv.jpg"
image_path = "images/tank.jpg"

output_image_path = "{}_out.jpg".format(os.path.splitext(image_path)[0])


image = cv2.imread(image_path)
H, W, _ = image.shape


model_path = "best.pt"

model = YOLO(model_path)


threshold = 0.5

results = model(image)[0]




# Проходимся по обнаруженным объектам и рисуем прямоугольники и подписи
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    if score > threshold:

        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)


        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.circle(image, (x_center, y_center), 5, (0, 255, 0), thickness=cv2.FILLED)
        cv2.putText(image, model.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)



# Сохраняем результат
cv2.imwrite(output_image_path, image)
