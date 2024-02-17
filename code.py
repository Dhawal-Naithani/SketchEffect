#importing libraries
import cv2
import numpy as np


#function to obtian a blurred version of image 

def give_blurred(img_temp):

    filter_kernel = np.ones((3,3) , dtype=int)
    filter_kernel=filter_kernel/9
    img_temp=cv2.filter2D(img_temp, -1, filter_kernel)
    img_temp = cv2.bilateralFilter(img_temp,6,6,6)
    img_temp=cv2.medianBlur(img_temp,3)
    return img_temp


#function to obtain edges of the image

def give_edge(img_temp):

    c=7
    blk_size=23
    
    img_temp = cv2.adaptiveThreshold(img_temp,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blk_size, c)
    '''for i in range(5):
        img_temp=cv2.medianBlur(img_temp,3)'''
    img_temp = cv2.cvtColor(img_temp,cv2.COLOR_GRAY2BGR)

    return img_temp


#function to obtain a grayscale version of the image

def give_gray(img_temp):

    img_temp=cv2.cvtColor(img_temp,cv2.COLOR_BGR2GRAY)
    return img_temp


#function for applying kmeans clustering algorithm

def give_kmeans_clus(img_temp):

    img = img_temp.reshape(-1,3)
    img = np.float32(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 3, 0.9)
    k = 9
    _,label,center=cv2.kmeans(img, k, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    img_out = center[label.flatten()]
    img_out = img_out.reshape(img_temp.shape)
    return img_out


#reading the image
img_og = cv2.imread("messi.jpg")

#resizing image
x,y,z = np.shape(img_og)
a=1
img_og= cv2.resize(img_og,(y//a, x//a))


#performing operations on the image
img_gray = give_gray(img_og)
img_blur = give_blurred(img_gray)
img_edge = give_edge(img_blur)
img_kmc = give_kmeans_clus(img_og)

#combining the edges and kmc image to give final output
result= cv2.bitwise_and(img_edge,img_kmc)

#print(np.shape(img_blur))
#print(np.shape(img_edge))

cv2.imshow("original",img_og)
cv2.imshow("gray",img_gray)
cv2.imshow("edge",img_edge)
cv2.imshow("kmeans",img_kmc)
cv2.imshow("Result-CARTOON",result)

cv2.waitKey(0)
cv2.destroyAllWindows