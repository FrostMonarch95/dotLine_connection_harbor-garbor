import cv2
import numpy as np
import os
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum
target_vale=26
img = cv2.imread('ts.png')
cv2.waitKey(0)
cv2.destroyAllWindows()
img_color=img.copy()
img_color[img_color!=target_vale]=0
img=img[:,:,0].copy()
img[img!=target_vale]=0
img[img==target_vale]=30
gray=img.copy()
kernel=np.ones((5,5),dtype=np.uint8)
dil=cv2.dilate(gray,kernel,iterations=3)
edges = cv2.Canny(dil,50,150,apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/360,100)
all_theta=[]
for i in range(len(lines)):
    for rho,theta in lines[i]:
        print(theta*180/np.pi)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        cv2.line(img_color,(x1,y1),(x2,y2),(0,0,255),2)
        all_theta.append(theta)
cv2.imshow('img line ',img_color)
cv2.waitKey(0)
filter=[]
ksize=31
for the in all_theta:
    kern=cv2.getGaborKernel((ksize, ksize), 4.0, the, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    kern /= 1.5*kern.sum()
    filter.append(kern)
fimg=process(img,filter)
cv2.imshow("ori: ",img)
cv2.imshow("garbor:",fimg)
th3=np.zeros_like(fimg)
_,th3 = cv2.threshold(fimg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# kernel=np.ones((1,100),np.uint8)
# th3=cv2.erode(th3,kernel,iterations=1)
cv2.imshow('threshold: ',np.uint8(th3))
# ero=cv2.erode(th3,kernel,iterations=1)
# cv2.imshow("erode: ",ero)
cv2.waitKey(0)