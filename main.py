
import cv2



file_name = "img.tiff"



img = cv2.imread(file_name, -1)


cv2.imshow('image', img)
cv2.waitKey(0)

