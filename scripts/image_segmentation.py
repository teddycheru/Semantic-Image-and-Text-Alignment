import os
import cv2
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import torch.nn.functional as F
from torchvision import models

# Define the input directory
input_dir = '/home/tewodros_cheru/Challenge_Data/Assets/3c35998c3f4a279a008ac3ffd8481fea'

# Function to load images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images

# Load images
images = load_images_from_folder(input_dir)

# Preprocess images for YOLO
preprocess = transforms.Compose([
    transforms.ToTensor(),
])

# Convert images to numpy arrays
images_np = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

# Load pre-trained YOLOv5 model
model = YOLO("yolov5su.pt")

# Perform object detection on the batch of images
detections = [model(img) for img in images_np]

# Process detections
for i, detection in enumerate(detections):
    print(f"Image {i+1} detections:")
    if detection[0].probs is not None:
        for idx, box in enumerate(detection[0].boxes.xyxy):
            score = detection[0].probs[idx].item() if detection[0].probs[idx] is not None else 0.0
            class_name = model.names[int(detection[0].classes[idx])]
            print(f"Object {idx+1}: {box.numpy()}, Score: {score}, Class: {class_name}")
    else:
        print("No detections found.")

output_dir = '../data/segmentation_results'
os.makedirs(output_dir, exist_ok=True)

print(f"Input Directory: {input_dir}")  # Debugging statement
print(f"Output Directory: {output_dir}")  # Debugging statement

# Transformation for input image
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

def segment_image(image_path):
    print(f"Processing image: {image_path}")  # Debugging statement
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")  # Debugging statement
        return None, None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(image_rgb).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

    return image, output_predictions

# Perform segmentation on images in the input directory
for img_name in os.listdir(input_dir):
    if img_name.endswith('.jpg') or img_name.endswith('.png'):
        img_path = os.path.join(input_dir, img_name)
        original_image, segmented_image = segment_image(img_path)
        
        if original_image is not None and segmented_image is not None:
            # Save segmented result
            segmented_image_path = os.path.join(output_dir, f'segmented_{img_name}')
            cv2.imwrite(segmented_image_path, segmented_image * 255)  # Scale mask to 0-255
            
            # Debugging: Print the path of the saved file
            print(f"Segmented image saved at: {segmented_image_path}")
            
            # Visualize results
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(segmented_image, cmap='gray')
            plt.title('Segmentation Mask')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"Failed to process image: {img_path}")  # Debugging statement


