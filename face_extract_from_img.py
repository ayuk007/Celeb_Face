import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

prototxt_path = 'model_path/deploy.prototxt'
caffemodel_path = 'model_path/weights.caffemodel'

# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

base_path = "Dataset_Raw"
#update_path = "updated_1_images"
output_path = "DATA"

celebs = os.listdir(base_path)

for celeb in celebs:
    os.mkdir(os.path.join(output_path,celeb))
    images = os.listdir(os.path.join(base_path, celeb))

    counter = 1
    for img in images:
        try:
            image = cv2.imread(os.path.join(base_path, celeb, img))

            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            model.setInput(blob)
            detections = model.forward()
            for i in range(0, detections.shape[2]):
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                confidence = detections[0, 0, i, 2]

                # If confidence > 0.95, show box around face
                if (confidence > 0.95):
                    
                    frame = image[startY:endY, startX:endX]
                    cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 2)
            
            # cv2.imwrite(os.path.join(update_path, img), image)
            cv2.imwrite(os.path.join(output_path, celeb, f"{celeb}.{counter}.jpg"), frame)
            counter+=1
        except Exception as e:
            print(f"Exception raised for {img}")