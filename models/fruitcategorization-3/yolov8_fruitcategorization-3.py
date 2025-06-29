# -*- coding: utf-8 -*-
"""YOLOV8_usage_example.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16aobjmUwAdk_eE8L9wHYJA-k4uIg8UKB
"""

# Dataset link https://universe.roboflow.com/aleena-khan-kryp8/fruitcategorization/dataset/3

# Step 1: Install the required packages

#!pip install roboflow ultralytics

from roboflow import Roboflow
from ultralytics import YOLO
import torch

rf = Roboflow(api_key="UoqSThv4mOF9lEKcZHRm")
project = rf.workspace("aleena-khan-kryp8").project("fruitcategorization")
version = project.version(3)
dataset = version.download("yolov8")

print("Dataset location: ", dataset.location)

# Step 5: Initialize the YOLOv8 model
model = YOLO('yolov8n.pt')

# Step 6: Train the model using the downloaded dataset path
# Assuming the dataset.yaml is located in /content/Fruit-Ripeness-1/
results = model.train(data=dataset.location + '/data.yaml', epochs=100, imgsz=640)

# Step 7: Evaluate the model
metrics = model.val()

# Step 8: Save the trained model
model.save("best_yolov8_fruitcategorization-3.pt")

# Optional: Print evaluation metrics
print("Evaluation metrics: ", metrics)


model = YOLO('/content/best_yolov8_fruitcategorization-3.pt')  # Path to your trained model
results = model.predict("./fruitcategorization-3/valid/images", save=True)

"""
# Step 4: Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Step 5: Configure hyperparameters
config = {
    "epochs": 40,
    "batch": 16,
    "img": 640,
    "optimizer": "Adam",
    "lr0": 0.001,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "device": "0" if torch.cuda.is_available() else "cpu",
}

# Step 6: Start the training process
results = model.train(
    data="/content/Fruit-Ripeness-1",  # Point to the correct path
    epochs=config["epochs"],
    batch=config["batch"],
    imgsz=config["img"],
    optimizer=config["optimizer"],
    lr0=config["lr0"],
    momentum=config["momentum"],
    weight_decay=config["weight_decay"],
    device=config["device"],
    name="fruit_defect_detection"
)

results = model.train(data="/content/Fruit-Ripeness-1", epochs=100, imgsz=640)


# Step 7: Evaluate the model
metrics = model.val()

# Step 8: Save the trained model
model.save("best_yolov8_fruit_defect.pt")

# Export to TensorRT
# model.export(format="engine")

# Step 9: Test the trained model
results = model.predict("/content/Fruit-Ripeness-1/test/images", save=True)
"""
