import cv2
import os
import numpy as np


def is_overlapping(box1, box2):
	"""
	Check if two bounding boxes overlap.
	Each box is defined as [xmin, xmax, ymin, ymax].
	"""
	# Check if one box is to the left of the other
	if box1[1] < box2[0] or box2[1] < box1[0]:
		return False
	# Check if one box is above the other
	if box1[3] < box2[2] or box2[3] < box1[2]:
		return False
	return True

def check_overlaps(boxes1,boxes2):
	"""
	Check if there are any overlapping boxes in a list of bounding boxes.
	"""
	d = {}
	for box1 in boxes1:
		for box2 in boxes2:
			if is_overlapping(box1, box2):
				print(box1)
				print(box2)
				print('------------')
				d[box1] = box2
	return d
	

count = 0
boxes1 = []
boxes2 = []
#path1 = '../new_data/Iberia/usableTPs/LRM/'
#path2 = 'TP/LRM/'
n_images = 0
for image in os.listdir('../datasets/hillforts/LRM/'):
	
	lrm = cv2.imread('../datasets/hillforts/LRM/'+image)
	mask = cv2.imread('../datasets/hillforts/Masks/'+image,0)
	rgb = cv2.imread('../datasets/hillforts/RGB/'+image)


	lrm = cv2.resize(lrm, (500,500))
	rgb = cv2.resize(rgb, (500,500))
	mask = cv2.resize(mask, (500,500))

	ret, thresh = cv2.threshold(mask, 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(lrm, contours, -1, (0,255,0), 1)
	cv2.drawContours(rgb, contours, -1, (0,255,0), 1)

	count += len(contours)

	print(image)
	print(len(contours))
	cv2.imshow('lrm',lrm)
	cv2.imshow('rgb',rgb)
	cv2.imshow('mask',mask)
	cv2.waitKey()
	n_images+=1

print(n_images)
'''	xmin, xmax, ymin, ymax = image.split('(')[1].split(')')[0].split('_')
	boxes1.append((float(xmin), float(xmax), float(ymin), float(ymax)))

for image in os.listdir(path2):
	xmin, xmax, ymin, ymax = image.split('(')[1].split(')')[0].split('_')
	boxes2.append((float(xmin), float(xmax), float(ymin), float(ymax)))

d = check_overlaps(boxes1,boxes2)

for k in d:
	xmin, xmax, ymin, ymax = k
	filename1 = path1 + 'Background(' + str(xmin) + '_' + str(xmax) + '_' + str(ymin) + '_' + str(ymax) + ').png'
	xmin, xmax, ymin, ymax = d[k]
	filename2 = path2 + 'Background(' + str(xmin) + '_' + str(xmax) + '_' + str(ymin) + '_' + str(ymax) + ').png'

	lrm1 = cv2.imread(filename1)
	lrm2 = cv2.imread(filename2)

	cv2.imshow(path1, lrm1)
	cv2.imshow(path2, lrm2)
	cv2.waitKey()'''

