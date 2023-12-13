import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

def detect_birds(image):
    """
    Detect birds in an image.

    :param image: An image file.
    :return: A list of cropped bird images.
    """

    # Load the pretrained model
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    model.eval()

    # convert image to numpy array
    image = read_image(image)

    # Process the image
    transformed_image = process_image(image)

    # Lower the detection threshold
    threshold = 0.3

    # Detect birds
    boxes, labels, scores = detect_objects(transformed_image, model, threshold=threshold)

    # Crop and draw bounding boxes around birds
    cropped_images = crop_image(image, boxes, labels)

    # Placeholder
    return cropped_images


def read_image(uploaded_image):
    """
    Read the uploaded image.

    :param uploaded_image: An uploaded image file.
    :return: The image as a numpy array.
    """

    # Read the image from file object
    image_stream = uploaded_image.read()
    image_as_np_array = np.frombuffer(image_stream, dtype=np.uint8)
    image = cv2.imdecode(image_as_np_array, cv2.IMREAD_COLOR)
    return image


def process_image(image):
    """
    Process the uploaded image (convert color to RGB and image to tensor).

    :param uploaded_image: An uploaded image file.
    :return: Transformed image.
    """

    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Transform the image to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image)

    return image


def detect_objects(image, model, threshold=0.5):
    """
    Detect objects in an image.

    :param image: An image file.
    :param model: A pretrained model.
    :param threshold: The confidence threshold.
    :return: Predicted bounding boxes, labels, and scores.
    """

    with torch.no_grad():
        prediction = model([image])
    # Filter predictions based on the threshold
    pred_boxes = prediction[0]['boxes'][prediction[0]['scores'] > threshold]
    pred_labels = prediction[0]['labels'][prediction[0]['scores'] > threshold]
    pred_scores = prediction[0]['scores'][prediction[0]['scores'] > threshold]
    return pred_boxes, pred_labels, pred_scores


def crop_image(image, boxes, labels):
    """
    Crop the image to images containing a single bird.

    :param original_image: The original image.
    :param boxes: Predicted bounding boxes.
    :param labels: Predicted labels.
    :return: List of cropped bird images.
    """

    cropped_images = []

    for box, label in zip(boxes, labels):
        if label == 16:  # Class label for birds in COCO
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            cropped_images.append(crop)

    return cropped_images
