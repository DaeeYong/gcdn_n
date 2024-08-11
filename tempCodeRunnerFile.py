from dragon import dragonReadNet25 as d25
import cv2
import numpy as np
from collections import defaultdict
from dragon import dragonY
from dragon import dragonV
import cv2

img_path = '/Users/ivory/Documents/github/gcdn_n/media/y.jpg'
proto = '/Users/ivory/Documents/github/gcdn_n/body_25/pose_deploy.prototxt'
weights = '/Users/ivory/Documents/github/gcdn_n/body_25/pose_iter_584000.caffemodel'

net = cv2.dnn.readNetFromCaffe(proto, weights)

img = cv2.imread(img_path)
height, width, _ = img.shape
y_1 = round(height/3)
y_2 = round(height/3 * 2)
x_1 = round(width/3)
x_2 = round(width/3 * 2)


print(height, width)
inblob = d25.preprocess_image_blob(img)
net.setInput(inblob)
output = net.forward()
joint_pos_list, confidence_list, _ = d25.get_position_from_netoutput(output, img)
d25.mark_on_image(img, joint_pos_list)
cv2.imshow('img', img)
cv2.waitKey()

cv2.destroyAllWindows()
