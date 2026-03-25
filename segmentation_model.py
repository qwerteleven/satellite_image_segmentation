from transformers import pipeline
from PIL import Image

import numpy as np

import cv2
import os


def preprocess_image(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x_splits = int(round(image.shape[0] / input_resolution[1]))
    y_splits = int(round(image.shape[1] / input_resolution[0]))


    x_slice_points = np.linspace(0, image.shape[0], x_splits, dtype=np.int16) 
    y_slice_points = np.linspace(0, image.shape[1], y_splits, dtype=np.int16) 
    
    
    crop_inputs = [
        (image[x_slice_points[x]:x_slice_points[x + 1], y_slice_points[y]:y_slice_points[y + 1], :] * 255).astype(np.uint8)
        for x in range(len(x_slice_points) - 1)
        for y in range(len(y_slice_points) - 1)
    ]

    crop_inputs = [Image.fromarray(image) for image in crop_inputs]

    return crop_inputs, x_slice_points, y_slice_points



def inferences_loop(data, ml_model):

    results = []
    for data_input in data:

        print(data_input.size)
        results.append(ml_model(data_input))

    return results


def save_results(results, index = 0):
    for result in results:
        result[detection_type].save(f"{output_path}/{result['label']}_{index}.png")



def union_results(results, original_shape, x_slice_points, y_slice_points):

    images_by_labels = {}

    for result in results:
        for segmentation_type in result:
            if segmentation_type["label"] not in images_by_labels:
                images_by_labels[segmentation_type["label"]] = np.zeros(original_shape[:2])



    for x in range(len(x_slice_points) - 1):
        for y in range(len(y_slice_points) - 1):
            for segmentation_type in results[(x * (len(y_slice_points) - 1)) + y]:

                mask = np.asarray(segmentation_type[detection_type])
                images_by_labels[segmentation_type["label"]][x_slice_points[x]:x_slice_points[x + 1], y_slice_points[y]:y_slice_points[y + 1]] = mask

    return images_by_labels


if __name__ == "__main__":

    model_path = "nvidia/segformer-b0-finetuned-ade-512-512" # "Intel/dpt-hybrid-midas" # "nielsr/eomt-dinov3-ade-semantic-large-512"
    detection_type = "mask" # "mask"


    model = model_path.split("/")[-1]
    input_resolution = [512, 512]
    output_path = f"results/{model}"
    image = cv2.imread("img.tiff", -1)

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    ml_model = pipeline("image-segmentation", model=model_path)

    data, x_slice_points, y_slice_points = preprocess_image(image)
    
    results = inferences_loop(data, ml_model)


    for index_result, input_result in enumerate(results):
        save_results(input_result, index_result)


    images_by_labels = union_results(results, image.shape, x_slice_points, y_slice_points)

    for label, union_image in images_by_labels.items():
        cv2.imwrite(f"{output_path}/union_results_{label}.png", union_image)


