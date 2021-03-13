import cv2
import numpy as np
from matplotlib import pyplot as plt

# img =cv2.imread ('C:/SAI/IIIT_2020/DataAnalytics/Final_Project/a.jpg',0)

# ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

# for i in range(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])

# plt.show()

img1 = cv2.imread('a.jpg', 1)
img2 = img1.copy()
img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

#--- Blur the gray scale image
img = cv2.GaussianBlur(img,(5, 5),0)

plt.imshow(img,cmap='gray')

#--- Perform Canny edge detection (in my case lower = 84 and upper = 255, because I resized the image, may vary in your case)
edges = cv2.Canny(img, 0, 180)

plt.imshow(edges,cmap='gray')
# cv2.imshow('Edges', edges )
#---Find and draw all existing contours
# _, contours = cv2.findContours(edges, cv2.RETR_TREE, 1)
# rep = cv2.drawContours(img1, contours, -1, (0,255,0), 3)
# cv2.imshow('Contours',rep)
# plt.imshow(rep,cmap='gray')

# #---Determine eccentricity
# cnt = contours
# for i in range(0, len(cnt)):
#     ellipse = cv2.fitEllipse(cnt[i])
#     (center,axes,orientation) =ellipse
#     majoraxis_length = max(axes)
#     minoraxis_length = min(axes)
#     eccentricity=(np.sqrt(1-(minoraxis_length/majoraxis_length)**2))
#     cv2.ellipse(img2,ellipse,(0,0,255),2)

# cv2.imshow('Detected ellipse', img2)
