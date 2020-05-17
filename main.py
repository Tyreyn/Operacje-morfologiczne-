import cv2
import numpy as np
from matplotlib import pyplot as plt

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
src = cv2.imread("mpo_zad0513.png")

# ***** TWORZENIE MASKI *****
bialy1 = np.array([120,120,120])
bialy2 = np.array([255,255,255])
maska = cv2.inRange(src, bialy1, bialy2)
gray = cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
# ***** OPERACJE MORFOLOGICZNE *****
dylatacja = cv2.dilate(maska,kernel, iterations=2)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

erozja = cv2.erode(dylatacja,kernel,iterations=7)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(6,6))

hitmiss = cv2.morphologyEx(erozja, cv2.MORPH_HITMISS, kernel)

hit = cv2.cvtColor(hitmiss,cv2.COLOR_GRAY2RGB)

hit[np.where((hit==[255,255,255]).all(axis=2))] = [0,128,0]


final = cv2.add(src , hit)
# ***** WYSWIETLANIE *****
plt.subplot(2,2,1),plt.imshow(maska,'gray')
plt.title('DYLATACJA'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,2),plt.imshow(erozja,'gray')
plt.title('EROZJA'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,3),plt.imshow(hit,'gray')
plt.title('HIT&MISS'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,4),plt.imshow(final,'gray')
plt.title('Wynikowe'), plt.xticks([]), plt.yticks([])
plt.show()
cv2.waitKey(0)