from ultralytics import YOLO
import cv2

model_path = '../models/canicas/2025_02_24/2025_02_24_canicas_yolo11n_FP16_GPU_batch4.engine'

model = YOLO(model_path, task='detect')

#model(conf=0.5, half=True, batch=4, augment=True)

image_1 = cv2.imread('image_1.jpg')
image_2 = cv2.imread('image_2.jpg')
image_3 = cv2.imread('image_3.jpg')
image_4 = cv2.imread('image_4.jpg')

images = [image_1, image_2, image_3, image_4]

results = model.predict(images, conf=0.5, half=True, augment=True, batch=4)

#preprocessed = model.predictor.preprocess([images])

#output = model.predictor.inference(preprocessed)

#results = model.predictor.postprocess(output, preprocessed, [images])
print(results)
print(type(results))
print(len(results))  # Check if `results` is a list or another type

print(results[0].speed["preprocess"])