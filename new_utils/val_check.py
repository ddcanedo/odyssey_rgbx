import cv2
import os
import numpy as np

for image in os.listdir('../validation_results/'):
	lrm = cv2.imread('../datasets/hillforts/LRM/'+image)
	mask = cv2.imread('../datasets/hillforts/Masks/'+image,0)
	pred = cv2.imread('../validation_results/'+image,0)
	rgb = cv2.imread('../datasets/hillforts/RGB/'+image)


	lrm = cv2.resize(lrm, (500,500))
	rgb = cv2.resize(rgb, (500,500))
	mask = cv2.resize(mask, (500,500))
	pred = cv2.resize(pred, (500,500))

	ret, thresh = cv2.threshold(mask, 127, 255, 0)
	contours_gt, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	ret, thresh = cv2.threshold(pred, 127, 255, 0)
	contours_pred, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	cv2.drawContours(lrm, contours_gt, -1, (255,0,0), 1)
	cv2.drawContours(rgb, contours_gt, -1, (255,0,0), 1)
	cv2.drawContours(lrm, contours_pred, -1, (0,255,0), 1)
	cv2.drawContours(rgb, contours_pred, -1, (0,255,0), 1)
	print(len(contours_pred))

	
	print(image)
	cv2.imshow('lrm',lrm)
	cv2.imshow('rgb',rgb)
	cv2.waitKey()
