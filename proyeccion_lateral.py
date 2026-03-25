import numpy as np
import matplotlib.pyplot as plt
import cv2




def border_adjust(S1, S2, horizontal = True):

    alpha = 10e-10
    mesh_pixels = 100
    change_pixel_limit = 10
    pixel_max_value = 255
    upper_bound_value = 200
    lower_vound_value = 30
    maximum_pixel_change = 4

    if horizontal:
        axis_length = S1.shape[0]
    else:
        axis_length = S1.shape[1]



    for index in range(axis_length):
        if horizontal:
            a = S1[index,:].astype(np.float64)
            b = S2[index,:].astype(np.float64)
        else:
            a = S1[:, index].astype(np.float64)
            b = S2[:, index].astype(np.float64)


        meeting_point = (float(a[-1]) + float(b[0])) / 2
        a_proportion = meeting_point / (a[-1] + alpha)
        
        mask_a = np.ones((len(a),))

        pixel_mean_direction = np.zeros((20,))
        accumulate_change = 0
        for index_mask in range(len(a) - 1):
            change_pixel = abs(a[(len(a) - 1) - index_mask] - a[(len(a) - 1) - index_mask - 1])
            
            pixel_mean_direction[index_mask % 20] = change_pixel
            
            if abs(np.min(pixel_mean_direction) - np.max(pixel_mean_direction)) > maximum_pixel_change:
                accumulate_change += pixel_max_value

            if change_pixel < change_pixel_limit:
                accumulate_change += change_pixel

            if a[(len(a) - 1) - index_mask] < lower_vound_value or a[(len(a) - 1) - index_mask] > upper_bound_value:
                accumulate_change += pixel_max_value

            if accumulate_change > mesh_pixels:
                accumulate_change = pixel_max_value

            mask_a[(len(a) - 1) - index_mask] = accumulate_change

        mask_a = mask_a - np.min(mask_a) 
        mask_a = mask_a / pixel_max_value
        mask_a = 1 - mask_a


        if a_proportion < 1:
            mask_a = mask_a * -1

        
        mask_a = mask_a * abs(float(a[-1]) - meeting_point)


        mask_a = a + mask_a 

        if horizontal:
            S1[index,:] = mask_a
        else:
            S1[:, index] = mask_a





        b_proportion = meeting_point / (b[0] + alpha)

        mask_b = np.ones((len(b),))
        pixel_mean_direction = np.zeros((20,))

        accumulate_change = 0.
        for index_mask in range(len(b) - 1):
            change_pixel = abs(b[index_mask] - b[index_mask + 1]) 
            
            pixel_mean_direction[index_mask % 20] = change_pixel

        
            if abs(np.min(pixel_mean_direction) - np.max(pixel_mean_direction)) > maximum_pixel_change:
                accumulate_change += pixel_max_value

        
            if change_pixel < change_pixel_limit:
                accumulate_change += change_pixel


            if b[index_mask] < lower_vound_value or b[index_mask] > upper_bound_value:
                accumulate_change += pixel_max_value


            if accumulate_change > mesh_pixels:
                accumulate_change = pixel_max_value
        

            mask_b[index_mask] = accumulate_change

        mask_b = mask_b - np.min(mask_b) 
        mask_b = mask_b / pixel_max_value
        mask_b = 1 - mask_b

        mask_b = mask_b * abs(float(b[0]) - meeting_point)


        if b_proportion < 1:
            mask_b = mask_b * -1

        
        mask_b = b + mask_b 

        if horizontal:
            S2[index, :] = mask_b
        else:
            S2[:, index] = mask_b

    return S1, S2


def border_adjust_mean(S1, S2, horizontal = True):
    
    alpha = 10e-10
    mesh_pixels = 100
    change_pixel_limit = 10
    pixel_max_value = 255
    upper_bound_value = 200
    lower_vound_value = 30
    maximum_pixel_change = 4


    if horizontal:
        a = S1[:, -1].astype(np.float64)
        b = S2[:, 0].astype(np.float64)
    else:
        a = S1[-1, :].astype(np.float64)
        b = S2[0, :].astype(np.float64)


    meeting_point = (a + b) / 2



    mask = np.ones(S1.shape[:2])

    pixel_mean_direction = np.zeros((20,))
    accumulate_change = 0

    """

    change_pixel = abs(a[(len(a) - 1) - index_mask] - a[(len(a) - 1) - index_mask - 1])
    
    pixel_mean_direction[index_mask % 20] = change_pixel
    
    if abs(np.min(pixel_mean_direction) - np.max(pixel_mean_direction)) > maximum_pixel_change:
        accumulate_change += pixel_max_value

    if change_pixel < change_pixel_limit:
        accumulate_change += change_pixel

    if a[(len(a) - 1) - index_mask] < lower_vound_value or a[(len(a) - 1) - index_mask] > upper_bound_value:
        accumulate_change += pixel_max_value

    if accumulate_change > mesh_pixels:
        accumulate_change = pixel_max_value

    mask_a[(len(a) - 1) - index_mask] = accumulate_change


    """

    mask = mask - np.min(mask) 
    mask = mask / pixel_max_value
    mask = 1 - mask


    proportion_a = np.mean(meeting_point / (a + alpha))

    mask_a = mask
    mask_b = mask


    if proportion_a < 1:
        mask_a = mask * -1
    else:
        mask_b = mask * -1

    
    mask_a = mask_a * abs(a - meeting_point)
    mask_b = mask_b * abs(b - meeting_point)


    a = a + mask_a
    b = b + mask_b


    S1 = S1 + a
    S2 = S2 + b



    return S1, S2

if __name__ == "__main__":


    S1 = cv2.imread("results/dpt-hybrid-midas-stack/depth_0.png", cv2.IMREAD_GRAYSCALE).astype(np.float64)
    S2 = cv2.imread("results/dpt-hybrid-midas-stack/depth_1.png", cv2.IMREAD_GRAYSCALE).astype(np.float64)


    S1, S2 = border_adjust_mean(S1, S2, False)
    fig, axs = plt.subplots(2)

    axs[0].imshow(1 - S1 , cmap='Grays')
    axs[1].imshow(1 - S2, cmap='Grays')


    plt.show()
