from transformers import pipeline
from PIL import Image

import numpy as np

import cv2
import os


def preprocess_image(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x_splits = int(round(image.shape[0] / input_resolution[1]))
    y_splits = int(round(image.shape[1] / input_resolution[0]))


    x_splits = 4
    y_splits = 4


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
        results.append(ml_model(data_input))

    return results


def save_results(results, index = 0):
 
    results[detection_type].save(f"{output_path}/{detection_type}_{index}.png")



def union_results(results, original_shape, x_slice_points, y_slice_points):

    images_by_labels = {detection_type: np.zeros(original_shape[:2])}

    for x in range(len(x_slice_points) - 1):
        for y in range(len(y_slice_points) - 1):

            segmentation_type = results[(x * (len(y_slice_points) - 1)) + y]
            mask = np.asarray(segmentation_type[detection_type])

            mask = np.where(mask > 10, mask, 0)
            mask = (mask / np.max(mask)) * 255



            images_by_labels[detection_type][x_slice_points[x]:x_slice_points[x + 1], y_slice_points[y]:y_slice_points[y + 1]] = mask

    return images_by_labels


def aling_crops(results):
    """
    for image_index in range(len(results)):

        img = np.asarray(results[image_index][detection_type])
        img = img.copy()
        fuction_value = 0.01

        for index_row in range(img.shape[0]):
            a = img[index_row, 0]
            b = img[index_row, -1]

            m = (b - a) / (img.shape[1] - 0)
            c = img[index_row, 0]

            perspective_slope = np.linspace(0, img.shape[1], img.shape[1]) * m + c

            img[index_row] = np.where(img[index_row] < perspective_slope / 4, 0, img[index_row])
        
        results[image_index][detection_type] =  Image.fromarray(img)

    """

    for x in range(len(x_slice_points) - 1):
        column_size = len(y_slice_points) - 1

        images = results[x * column_size : (x + 1) * column_size]

        for crop_x_index in range(len(images) - 1):
            a = np.asarray(images[crop_x_index][detection_type])[:, -1] 
            b = np.asarray(images[crop_x_index + 1][detection_type])[:, 0]

            c = np.mean(a) - np.mean(b)

            b = np.asarray(images[crop_x_index + 1][detection_type]).transpose() + c
            b = b.transpose()

            b = b / np.max(b) * 255

            images[crop_x_index + 1][detection_type] = Image.fromarray(b)
            results[x * column_size + crop_x_index + 1][detection_type] = Image.fromarray(b)


    for y in range(len(y_slice_points) - 1):
        row_size = len(x_slice_points) - 1
        
        images = [
            results[index + y] 
            for index in range(0, len(results), row_size)
        ]

        for crop_y_index in range(len(images) - 1):
            a = np.asarray(images[crop_y_index][detection_type])[-1, :] 
            b = np.asarray(images[crop_y_index + 1][detection_type])[0, :]

            c = np.mean(a) - np.mean(b)

            b = np.asarray(images[crop_y_index + 1][detection_type]) + c

            b = b / np.max(b) * 255

            images[crop_y_index + 1][detection_type] = Image.fromarray(b)

            results[(crop_y_index + 1) * (len(images)) + y][detection_type] = Image.fromarray(b)
    

    return results

    







if __name__ == "__main__":

    model_path = "Intel/dpt-hybrid-midas"
    detection_type = "depth"
    model = model_path.split("/")[-1]
    input_resolution = [512, 512]
    output_path = f"results/{model}"
    image = cv2.imread("img_2.tiff", -1)



    if not os.path.exists(output_path):
        os.makedirs(output_path)


    ml_model = pipeline("depth-estimation", model=model_path)

    data, x_slice_points, y_slice_points = preprocess_image(image)
    
    results = inferences_loop(data, ml_model)


    for index_result, input_result in enumerate(results):
        save_results(input_result, index_result)
    
    results = aling_crops(results)

    images_by_labels = union_results(results, image.shape, x_slice_points, y_slice_points)

    for label, union_image in images_by_labels.items():
        # union_image = cv2.bilateralFilter(union_image,9,75,75) #  cv2.GaussianBlur(union_image, (5, 5), 0)

        cv2.imwrite(f"{output_path}/union_results_{label}.png", union_image)


