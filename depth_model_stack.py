from transformers import pipeline
from proyeccion_lateral import border_adjust
from PIL import Image

import numpy as np
import math

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

    print(len(crop_inputs), x_slice_points, y_slice_points)

    return crop_inputs, x_slice_points, y_slice_points



def inferences_loop(data, ml_model):

    results = []
    for data_input in data:
        results.append(ml_model(data_input))

    return results


def save_results(results, index = 0):
 
    results[detection_type].save(f"{output_path}/{detection_type}_{index}.png")



def union_results(results, original_shape, x_slice_points, y_slice_points, previous_depth_estimation):

    images_by_labels = {detection_type: np.zeros(original_shape[:2])}


    for x in range(len(x_slice_points) - 1):
        for y in range(len(y_slice_points) - 1):

            segmentation_type = results[(x * (len(y_slice_points) - 1)) + y]
            mask = np.asarray(segmentation_type[detection_type])
            """
            if previous_depth_estimation is not None:
                alpha = 10e-10
                previous_crop = previous_depth_estimation[x_slice_points[x] : x_slice_points[x + 1], y_slice_points[y] : y_slice_points[y + 1]]
                             
                minimum = np.min(previous_crop)
                maximum = np.max(previous_crop)

                mask = mask - np.min(mask)
                mask = mask / (np.max(mask) + alpha)


                mask = mask * (maximum - minimum)
                mask = mask + minimum

              
                mask = previous_crop * 0 + mask * 1
            """
            images_by_labels[detection_type][x_slice_points[x]:x_slice_points[x + 1], y_slice_points[y]:y_slice_points[y + 1]] = mask

   

    for x in range(len(x_slice_points) - 2):
        for y in range(len(y_slice_points) - 2):

            S1 = images_by_labels[detection_type][x_slice_points[x]:x_slice_points[x + 1], y_slice_points[y]:y_slice_points[y + 1]] 
            S2 = images_by_labels[detection_type][x_slice_points[x + 1]:x_slice_points[x + 2], y_slice_points[y]:y_slice_points[y + 1]] 

            S1, S2 = border_adjust(S1, S2, False)

            images_by_labels[detection_type][x_slice_points[x]:x_slice_points[x + 1], y_slice_points[y]:y_slice_points[y + 1]] = S1
            images_by_labels[detection_type][x_slice_points[x + 1]:x_slice_points[x + 2], y_slice_points[y]:y_slice_points[y + 1]] = S2


            S1 = images_by_labels[detection_type][x_slice_points[x]:x_slice_points[x + 1], y_slice_points[y]:y_slice_points[y + 1]] 
            S2 = images_by_labels[detection_type][x_slice_points[x]:x_slice_points[x + 1], y_slice_points[y + 1]:y_slice_points[y + 2]] 

            S1, S2 = border_adjust(S1, S2)

            images_by_labels[detection_type][x_slice_points[x]:x_slice_points[x + 1], y_slice_points[y]:y_slice_points[y + 1]] = S1
            images_by_labels[detection_type][x_slice_points[x]:x_slice_points[x + 1], y_slice_points[y + 1]:y_slice_points[y + 2]] = S2



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

            print((crop_y_index + 1) * (len(images)) + y, (crop_y_index + 1), y)

            results[int((crop_y_index + 1) * (len(images)) + y - 1)][detection_type] = Image.fromarray(b)
    

    return results

    







if __name__ == "__main__":

    model_path = "Intel/dpt-hybrid-midas"
    detection_type = "depth"
    model = model_path.split("/")[-1]
    vertical_view = True
    kernel_size = 40
    output_path = f"results/{model}-stack"
    image = cv2.imread("img_2.tiff", -1)

    input_resolutions = [
        [2 ** int(l_side), 2 ** int(l_side)]
        for l_side in range(8, int(math.log2(min(image.shape[:2]))))
    ][::-1]


    if not os.path.exists(output_path):
        os.makedirs(output_path)


    ml_model = pipeline("depth-estimation", model=model_path)

    previous_depth_estimation = None
    segmentation_sea_earth = None

    for split_resolution in input_resolutions:

        input_resolution = split_resolution

        data, x_slice_points, y_slice_points = preprocess_image(image)
        
        results = inferences_loop(data, ml_model)

        for depth_index, img in enumerate(results):
            cv2.imwrite(f"{output_path}/{input_resolution}_depth_{depth_index}.png", np.asarray(img[detection_type]))


        images_by_labels = union_results(results, image.shape, x_slice_points, y_slice_points, previous_depth_estimation)

        for label, union_image in images_by_labels.items():

            if previous_depth_estimation is None:

                kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size**2
                union_image = cv2.filter2D(union_image, -1, kernel)
                union_image = cv2.filter2D(union_image, -1, kernel)


                for index in range(union_image.shape[0]):
                    a = union_image[index, :]
                    a = a  - np.min(a)
                    union_image[index, :] = a

                if vertical_view:
             
                    union_image = np.where(union_image < 30, 0, 255) 
                    segmentation_sea_earth = union_image / 255
                

            if vertical_view:
                union_image = union_image  * segmentation_sea_earth

            previous_depth_estimation = union_image


            cv2.imwrite(f"{output_path}/union_results_{label}_{input_resolution}.png", union_image)

