from shapely.geometry import Polygon, box, Point, MultiPolygon
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import csv
import sys
import os
import cv2
from io import BytesIO
import matplotlib.pyplot as plt
import rasterio
from shapely.wkt import loads
from shapely.ops import unary_union
from torchvision import transforms
from utils.pyt_utils import parse_devices
from models.builder import EncoderDecoder as segmodel
from config import config
from utils.visualize import print_iou
from utils.metric import hist_info, compute_score
from owslib.wms import WebMapService
from timm.models.layers import to_2tuple
from utils.transforms import pad_image_to_shape, normalize
import shutil

csv.field_size_limit(sys.maxsize)
Image.MAX_IMAGE_PIXELS = None

def polys(row):
	geometry = loads(row)
	if isinstance(geometry, MultiPolygon):
		for polygon in geometry:
			x_coords, y_coords = polygon.exterior.xy
			xPoints = []
			yPoints = []
			polygonPoints = []
			# Iterate through the points
			for x, y in zip(x_coords, y_coords):
				xPoints.append(x)
				yPoints.append(y)
	return (xPoints, yPoints)
				


# Converts polygons to bounding boxes | Expected format is MULTIPOLYGON
def poly2bb(annotations, xMinImg, xMaxImg, yMaxImg, yMinImg, width, height):
	bbs = {}
	for key in annotations:
		geometry = loads(annotations[key])
		if isinstance(geometry, MultiPolygon):
			for polygon in geometry:

				x_coords, y_coords = polygon.exterior.xy
				xPoints = []
				yPoints = []
				polygonPoints = []
				# Iterate through the points
				for x, y in zip(x_coords, y_coords):
					xPoints.append(x)
					yPoints.append(y)
					polygonPoints.append([x, y])

				xMin = min(xPoints)
				xMax = max(xPoints)
				yMin = min(yPoints)
				yMax = max(yPoints)

				# The bounding box must be within the image limits
				if xMin >= xMinImg and xMax <= xMaxImg and yMin >= yMinImg and yMax <= yMaxImg:

					# Maps coordinates from GIS reference to image pixels
					xMinBb = round(mapping(xMin, xMinImg, xMaxImg, 0, width))
					xMaxBb = round(mapping(xMax, xMinImg, xMaxImg, 0, width))
					yMaxBb = round(mapping(yMin, yMinImg, yMaxImg, height, 0))
					yMinBb = round(mapping(yMax, yMinImg, yMaxImg, height, 0))

					polygon = []
					crsPolygon = []
					for p in polygonPoints:
						crsPolygon.append([p[0], p[1]])
						xPoly = mapping(p[0], xMinImg, xMaxImg, 0, width)
						yPoly = mapping(p[1], yMinImg, yMaxImg, height, 0)
						polygon.append([xPoly, yPoly])

					bbs[(xMinBb, xMaxBb, yMinBb, yMaxBb)] = [key, polygon, crsPolygon]

	return bbs

# Converts coordinates from GIS reference to image pixels
def mapping(value, min1, max1, min2, max2):
	return (((value - min1) * (max2 - min2)) / (max1 - min1)) + min2


def gen_masks(bbs, roi_bbs, cropped_img, crop, resolution):
	# Writes the mask
	mask = Image.new("L", cropped_img.size, 0)						
	for bb in roi_bbs:
		poly = bbs[bb][1]

		draw = ImageDraw.Draw(mask)

		cPoly = []
		for p in poly:
			cX = mapping(p[0], crop[0], crop[1], 0, resolution)
			cY = mapping(p[1], crop[2], crop[3], 0, resolution)
			cPoly.append((cX,cY))

		draw.polygon(cPoly, fill=255)

	return mask

# Check if a given polygon intersects with a cropped image.
def does_polygon_intersect_cropped_image(polygon, cropped_image_bounds):
	x_min, x_max, y_min, y_max = cropped_image_bounds

	p = Polygon(polygon)
	cropped_image_rect = box(x_min, y_min, x_max, y_max)

	# Check for intersection
	return p.intersects(cropped_image_rect)

# Get the extent resulting from the intersection between the user selected region and the loaded image
def getExtent(extent, coords, width, height):
	x1 = max(extent[0], coords[0])
	y1 = max(extent[1], coords[1])
	x2 = min(extent[2], coords[2])
	y2 = min(extent[3], coords[3])

	xMin = round(mapping(x1, extent[0], extent[2], 0, width))
	xMax = round(mapping(x2, extent[0], extent[2], 0, width))
	yMax = round(mapping(y1, extent[1], extent[3], height, 0))
	yMin = round(mapping(y2, extent[1], extent[3], height, 0))

	return (xMin, xMax, yMin,yMax)

# get bounding boxes from mask
def get_bounding_box(mask):
	y_indices, x_indices = np.where(np.array(mask) > 0)
	x_min, x_max = np.min(x_indices), np.max(x_indices)
	y_min, y_max = np.min(y_indices), np.max(y_indices)
	# add perturbation to bounding box coordinates
	H, W = np.array(mask).shape
	x_min = max(0, x_min - np.random.randint(0, 20))
	x_max = min(W, x_max + np.random.randint(0, 20))
	y_min = max(0, y_min - np.random.randint(0, 20))
	y_max = min(H, y_max + np.random.randint(0, 20))
	bbox = [x_min, y_min, x_max, y_max]
	return bbox

def clean_mask(mask, kernel_size=7):
	# Create a kernel for morphological operations
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	# Morphological closing to close small holes
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	# Morphological opening to remove small points
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

	return mask


def predict(img, modal_x, label, device, val_func):
	pred = sliding_eval_rgbX(img, modal_x, config.eval_crop_size, config.eval_stride_rate, val_func, device)
	hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
	results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

	result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
	result_img = np.array(result_img)
	result_img[result_img > 0] = 255

	return results_dict, result_img


def compute_metric(results):
	hist = np.zeros((config.num_classes, config.num_classes))
	correct = 0
	labeled = 0
	count = 0
	for d in results:
		hist += d['hist']
		correct += d['correct']
		labeled += d['labeled']
		count += 1

	iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
	result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
							config.class_names, show_no_back=False)
	return result_line, (iou, mean_IoU, freq_IoU, mean_pixel_acc, pixel_acc)


# add new funtion for rgb and modal X segmentation
def sliding_eval_rgbX(img, modal_x, crop_size, stride_rate, val_func, device=None):
	crop_size = to_2tuple(crop_size)
	ori_rows, ori_cols, _ = img.shape
	processed_pred = np.zeros((ori_rows, ori_cols, config.num_classes))

	for s in config.eval_scale_array:
		img_scale = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
		if len(modal_x.shape) == 2:
			modal_x_scale = cv2.resize(modal_x, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
		else:
			modal_x_scale = cv2.resize(modal_x, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

		new_rows, new_cols, _ = img_scale.shape
		processed_pred += scale_process_rgbX(img_scale, modal_x_scale, (ori_rows, ori_cols),
													crop_size, stride_rate, val_func, device)

	pred = processed_pred.argmax(2)

	return pred


def scale_process_rgbX(img, modal_x, ori_shape, crop_size, stride_rate, val_func, device=None):
	new_rows, new_cols, c = img.shape
	long_size = new_cols if new_cols > new_rows else new_rows

	if new_cols <= crop_size[1] or new_rows <= crop_size[0]:
		input_data, input_modal_x, margin = process_image_rgbX(img, modal_x, crop_size)
		score = val_func_process_rgbX(input_data, input_modal_x, val_func, device) 
		score = score[:, margin[0]:(score.shape[1] - margin[1]), margin[2]:(score.shape[2] - margin[3])]
	else:
		stride = (int(np.ceil(crop_size[0] * stride_rate)), int(np.ceil(crop_size[1] * stride_rate)))
		img_pad, margin = pad_image_to_shape(img, crop_size, cv2.BORDER_CONSTANT, value=0)
		modal_x_pad, margin = pad_image_to_shape(modal_x, crop_size, cv2.BORDER_CONSTANT, value=0)

		pad_rows = img_pad.shape[0]
		pad_cols = img_pad.shape[1]
		r_grid = int(np.ceil((pad_rows - crop_size[0]) / stride[0])) + 1
		c_grid = int(np.ceil((pad_cols - crop_size[1]) / stride[1])) + 1
		data_scale = torch.zeros(config.num_classes, pad_rows, pad_cols).cuda(device)

		for grid_yidx in range(r_grid):
			for grid_xidx in range(c_grid):
				s_x = grid_xidx * stride[0]
				s_y = grid_yidx * stride[1]
				e_x = min(s_x + crop_size[0], pad_cols)
				e_y = min(s_y + crop_size[1], pad_rows)
				s_x = e_x - crop_size[0]
				s_y = e_y - crop_size[1]
				img_sub = img_pad[s_y:e_y, s_x: e_x, :]
				if len(modal_x_pad.shape) == 2:
				    modal_x_sub = modal_x_pad[s_y:e_y, s_x: e_x]
				else:
				    modal_x_sub = modal_x_pad[s_y:e_y, s_x: e_x,:]

				input_data, input_modal_x, tmargin = process_image_rgbX(img_sub, modal_x_sub, crop_size)
				temp_score = val_func_process_rgbX(input_data, input_modal_x, val_func, device)

				temp_score = temp_score[:, tmargin[0]:(temp_score.shape[1] - tmargin[1]),
				                        tmargin[2]:(temp_score.shape[2] - tmargin[3])]
				data_scale[:, s_y: e_y, s_x: e_x] += temp_score
		score = data_scale
		score = score[:, margin[0]:(score.shape[1] - margin[1]),
				margin[2]:(score.shape[2] - margin[3])]

	score = score.permute(1, 2, 0)
	data_output = cv2.resize(score.cpu().numpy(), (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)

	return data_output

    

def val_func_process_rgbX(input_data, input_modal_x, val_func, device=None):
	input_data = np.ascontiguousarray(input_data[None, :, :, :], dtype=np.float32)
	input_data = torch.FloatTensor(input_data).cuda(device)

	input_modal_x = np.ascontiguousarray(input_modal_x[None, :, :, :], dtype=np.float32)
	input_modal_x = torch.FloatTensor(input_modal_x).cuda(device)

	
	with torch.cuda.device(input_data.get_device()):
		val_func.eval()
		val_func.to(input_data.get_device())
		with torch.no_grad():
			score = val_func(input_data, input_modal_x)
			score = score[0]
			if config.eval_flip:
				input_data = input_data.flip(-1)
				input_modal_x = input_modal_x.flip(-1)
				score_flip = val_func(input_data, input_modal_x)
				score_flip = score_flip[0]
				score += score_flip.flip(-1)
			score = torch.exp(score)

	return score

# for rgbd segmentation
def process_image_rgbX(img, modal_x, crop_size=None):
	p_img = img
	p_modal_x = modal_x

	if img.shape[2] < 3:
		im_b = p_img
		im_g = p_img
		im_r = p_img
		p_img = np.concatenate((im_b, im_g, im_r), amodal_xis=2)

	p_img = normalize(p_img, config.norm_mean, config.norm_std)
	if len(modal_x.shape) == 2:
		p_modal_x = normalize(p_modal_x, 0, 1)
	else:
		p_modal_x = normalize(p_modal_x, config.norm_mean, config.norm_std)

	if crop_size is not None:
		p_img, margin = pad_image_to_shape(p_img, crop_size, cv2.BORDER_CONSTANT, value=0)
		p_modal_x, _ = pad_image_to_shape(p_modal_x, crop_size, cv2.BORDER_CONSTANT, value=0)
		p_img = p_img.transpose(2, 0, 1)
		if len(modal_x.shape) == 2:
			p_modal_x = p_modal_x[np.newaxis, ...]
		else:
			p_modal_x = p_modal_x.transpose(2, 0, 1) # 3 H W

		return p_img, p_modal_x, margin

	p_img = p_img.transpose(2, 0, 1) # 3 H W

	if len(modal_x.shape) == 2:
		p_modal_x = p_modal_x[np.newaxis, ...]
	else:
		p_modal_x = p_modal_x.transpose(2, 0, 1)

	return p_img, p_modal_x



def load_model():
	model = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
	state_dict = torch.load(config.checkpoint_dir + '/finetuned_UK.pth')
	if 'model' in state_dict.keys():
		state_dict = state_dict['model']
	elif 'state_dict' in state_dict.keys():
		state_dict = state_dict['state_dict']
	elif 'module' in state_dict.keys():
		state_dict = state_dict['module']

	model.load_state_dict(state_dict, strict=True)

	ckpt_keys = set(state_dict.keys())
	own_keys = set(model.state_dict().keys())
	missing_keys = own_keys - ckpt_keys
	unexpected_keys = ckpt_keys - own_keys

	del state_dict

	return model


def valid_margin_contour(contour, contour_area, resolution, margin_rects, threshold=0.5):
	intersection_area = 0

	for rect in margin_rects:
		# Create a mask for the margin rect
		rect_mask = np.zeros((resolution, resolution), dtype=np.uint8)
		cv2.rectangle(rect_mask, (rect[0], rect[1]), (rect[2], rect[3]), 255, -1)

		# Create a mask for the contour
		contour_mask = np.zeros((resolution, resolution), dtype=np.uint8)
		cv2.drawContours(contour_mask, [contour], -1, 255, -1)

		# Calculate intersection
		intersection = cv2.bitwise_and(rect_mask, contour_mask)
		intersection_area += np.sum(intersection > 0)

	# Check if intersection is more than threshold
	if (intersection_area / contour_area) < threshold:
		return True
	return False




def main():
	# loads annotations and paths
	annotationsFolder = "../castros/annotations/" 
	imagesFolder = "../ODYSSEY/Images/LRM/"
	csvs = os.listdir(annotationsFolder)	

	resolution = 800

	# test with a selected region encompassing the whole LRMs
	coordinates = (-300000000,-300000000,300000000, 300000000)

	device = 0
	model = load_model()

	wms_url = 'https://ogc.apps.midgard.airbusds-cint.com/apgb/wms?guid=044130b5-d695-421d-bbc1-8ccac75c2db2'#'https://cartografia.dgterritorio.gov.pt/ortos2021/service?service=wmts&request=getcapabilities'
	layers  = ['AP-25CM-GB-LATEST']#['DGT_Ortos2021']
	srs = 'EPSG:27700'#'EPSG:3763'
	# Connect to the WMS service
	wms = WebMapService(wms_url)

	all_results = []
	all_detections = []
	annotationPolys = []

	area_threshold = 1480
	margin_threshold = 50 #pixels

	area_results = []
	margin_results = []
	above_min_area = []
	not_in_margins = []
	# Define the margin rectangles (top, bottom, left, right)
	margin_rects = [
		(0, 0, resolution, margin_threshold),  # Top margin
		(0, resolution - margin_threshold, resolution, resolution),  # Bottom margin
		(0, 0, margin_threshold, resolution),  # Left margin
		(resolution - margin_threshold, 0, resolution, resolution)  # Right margin
	]

	'''detection_count = 0
	image_count = 1
	for image in os.listdir('datasets/hillforts/LRM/'):
		print(image_count)
		print(image)

		modal_x = cv2.imread('datasets/hillforts/LRM/'+image, cv2.COLOR_BGR2RGB)
		gt = cv2.imread('datasets/hillforts/Masks/'+image,0)
		orthoimage = cv2.imread('datasets/hillforts/RGB/'+image, cv2.COLOR_BGR2RGB)
		gt[gt > 0] = 255
		ground_truth_mask = gt / 255
		

		results_dict, mask = predict(orthoimage, modal_x, ground_truth_mask, device, model)
		
		all_results.append(results_dict)


		ret, thresh = cv2.threshold(np.array(mask), 127, 255, 0)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cv2.imshow('LRM clean', modal_x)
		cv2.imshow('Ortoimage clean', orthoimage)


		margin_mask = np.zeros((resolution, resolution), dtype=np.uint8)
		for c in contours:
			# area filter
			area = cv2.contourArea(c)
			if area >= area_threshold:
				# margin filter
				margin_valid = valid_margin_contour(c, area, resolution, margin_rects, threshold=0.5)
				if margin_valid:
					cv2.fillPoly(margin_mask, pts=[c], color=255)


		ret, thresh = cv2.threshold(margin_mask, 127, 255, 0)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		if len(contours) > 0:
			detection_count += 1

		#largest_contours = max(contours, key=cv2.contourArea)



		cv2.drawContours(modal_x, contours, -1, (0,255,0), 2)
		cv2.drawContours(orthoimage, contours, -1, (0,255,0), 2)
		#cv2.drawContours(modal_x, [largest_contours], -1, (0,0,255), 2)
		#cv2.drawContours(orthoimage, [largest_contours], -1, (0,0,255), 2)
		
		#cv2.fillPoly(gt, pts=[largest_contours], color=255)
		#gt = clean_mask(gt)


		cv2.imshow('LRM', modal_x)
		cv2.imshow('Ortoimage', orthoimage)
		#cv2.imshow('gt',gt)

		while True:
			key = cv2.waitKey(0) & 0xFF  # Use bitwise AND to get the last 8 bits

			if key == ord('t') or key == ord('T'):
				#shutil.move('TP/LRM/'+image, 'usableTPs/LRM/'+image)
				#cv2.imwrite('usableTPs/Masks/'+image, gt)
				#shutil.move('TP/RGB/'+image, 'usableTPs/RGB/'+image) 
				#print("Usable")
				shutil.move('background/LRM/'+image, 'TP/LRM/'+image)
				shutil.move('background/Masks/'+image, 'TP/Masks/'+image)
				shutil.move('background/RGB/'+image, 'TP/RGB/'+image)
				print("True Positive")
				break

			elif key == ord('f') or key == ord('F'):
				shutil.move('background/LRM/'+image, 'FP/LRM/'+image)
				shutil.move('background/Masks/'+image, 'FP/Masks/'+image)
				shutil.move('background/RGB/'+image, 'FP/RGB/'+image)
				print("False Positive")
				break
			else:
				print("Inconclusive")
				break

		image_count+=1
		print('-------------------')

		if len(contours) == 0:
			shutil.move('dataset/LRM/'+image, 'background/LRM/'+image)
			shutil.move('dataset/Masks/'+image, 'background/Masks/'+image)
			shutil.move('dataset/RGB/'+image, 'background/RGB/'+image)
			true_bg+=1


	print(detection_count)'''

	# Iterates over the images
	for csvAnnotation in csvs:

		if csvAnnotation != 'inglaterra.csv':
			continue

		annotation = annotationsFolder + csvAnnotation

		for image in os.listdir(imagesFolder + csvAnnotation.split('.')[0]):
			if not image.startswith('Exeter'):
				continue

			print(imagesFolder+csvAnnotation.split('.')[0] + '/' + image)
			image = imagesFolder+csvAnnotation.split('.')[0] + '/' + image

			# Load image
			img = Image.open(image)

			if img.mode != 'RGB':
				img = img.convert('RGB')

			geoRef = rasterio.open(image)
			width, height = img.size

			# Parse corners of the image (GIS reference)
			xMinImg = geoRef.bounds[0]
			xMaxImg = geoRef.bounds[2]
			yMinImg = geoRef.bounds[1]
			yMaxImg = geoRef.bounds[3]

			# simulate the webservice format
			annotations = {}
			with open(annotation) as csvfile:
				reader = csv.DictReader(csvfile)
				# This considers that polygons are under the column name "WKT" and labels are under the column name "Id"
				polygons = "MULTIPOLYGON ((("
				count = 0
				for row in reader:
					xPoints, yPoints = polys(row['WKT'])

					if count != 0:
						polygons += ', (('

					for i in range(len(xPoints)):
						if i != len(xPoints)-1:
							polygons += str(xPoints[i]) + " " + str(yPoints[i]) + ','
						else:
							polygons += str(xPoints[i]) + " " + str(yPoints[i]) + '))'


					count += 1
				polygons += ')'

			annotations['castro'] = polygons

			# Get the extent resulting from the intersection between the user selected region and the loaded image
			extent = getExtent((xMinImg, yMinImg, xMaxImg, yMaxImg), coordinates, width, height)

			# bb : [annotation, poly]
			bbs = poly2bb(annotations, xMinImg, xMaxImg, yMaxImg, yMinImg, width, height)
			
			# Sliding window going through the extent of interest
			for i in range(extent[0], extent[1], resolution//2):
				for j in range(extent[2], extent[3], resolution//2):

					crop = (i, resolution+i, j, resolution+j)
					print(crop)
					cropped_img = img.crop((i, j, resolution+i, resolution+j))

					roi_bbs = []
					# Iterates over the bounding boxes
					for bb in bbs:
						if does_polygon_intersect_cropped_image(bbs[bb][1], crop):
							annotationPolys.append(Polygon(bbs[bb][2]))
							roi_bbs.append(bb)

					gt = gen_masks(bbs, roi_bbs, cropped_img, crop, resolution)
					gt = np.array(gt)
					gt[gt > 0] = 255
					ground_truth_mask = gt / 255
					if roi_bbs:

						# get sattelite image
						xMin = mapping(crop[0], 0, width, xMinImg, xMaxImg)
						xMax = mapping(crop[1], 0, width, xMinImg, xMaxImg)
						yMin = mapping(crop[3], height, 0, yMinImg, yMaxImg)
						yMax = mapping(crop[2], height, 0, yMinImg, yMaxImg)

						# Construct the GetMap request URL for the region
						orthoimage = wms.getmap(layers=layers,
												srs=srs,
												bbox=(xMin,yMin,xMax,yMax), 
												size=(resolution, resolution),
												format='image/png',
												timeout=300
												)

						# Read the image into a buffer
						buffer = BytesIO(orthoimage.read())

						# Convert the buffer to a numpy array
						image_array = np.asarray(bytearray(buffer.read()), dtype=np.uint8)

						# Decode the numpy array into an image
						orthoimage = cv2.imdecode(image_array, cv2.COLOR_BGR2RGB)

						
						modal_x = np.array(cropped_img)
						

						results_dict, mask = predict(orthoimage, modal_x, ground_truth_mask, device, model)
					
						all_results.append(results_dict)


						ret, thresh = cv2.threshold(gt, 127, 255, 0)
						gt_contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
						cv2.drawContours(modal_x, gt_contours, -1, (0,255,255), 2)

						ret, thresh = cv2.threshold(np.array(mask), 127, 255, 0)
						contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
						cv2.drawContours(modal_x, contours, -1, (255,0,0), 2)

						area_mask = np.zeros((resolution, resolution), dtype=np.uint8)
						margin_mask = np.zeros((resolution, resolution), dtype=np.uint8)
						for c in contours:
							polygon = []
							for p in c:
								p = p.squeeze()
								# cropped image to image extent
								x = mapping(p[0], 0, resolution, crop[0], crop[1])
								y = mapping(p[1], 0, resolution, crop[2], crop[3])

								# image extent to GIS
								x = mapping(x, 0, width, xMinImg, xMaxImg)
								y = mapping(y, height, 0, yMinImg, yMaxImg)

								polygon.append((x,y))
							all_detections.append(polygon)

							# area filter
							area = cv2.contourArea(c)
							if area >= area_threshold:
								cv2.fillPoly(area_mask, pts=[c], color=1)
								above_min_area.append(polygon)	

								# margin filter
								margin_valid = valid_margin_contour(c, area, resolution, margin_rects, threshold=0.5)
								if margin_valid:
									cv2.fillPoly(margin_mask, pts=[c], color=1)
									not_in_margins.append(polygon)



						# build new gt masks for the filters
						area_gt = np.zeros((resolution, resolution), dtype=np.uint8)
						margin_gt = np.zeros((resolution, resolution), dtype=np.uint8)
						for c in gt_contours:
							area = cv2.contourArea(c)
							if area >= area_threshold:
								cv2.fillPoly(area_gt, pts=[c], color=1)
								margin_valid = valid_margin_contour(c, area, resolution, margin_rects, threshold=0.5)
								if margin_valid:
									cv2.fillPoly(margin_gt, pts=[c], color=1)

						# area
						hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, area_mask, area_gt)
						results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

						area_results.append(results_dict)


						area_mask[area_mask > 0] = 255
						ret, thresh = cv2.threshold(area_mask, 127, 255, 0)
						contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
						cv2.drawContours(modal_x, contours, -1, (0,255,0), 2)


						# margin
						hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, margin_mask, margin_gt)
						results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

						margin_results.append(results_dict)


						margin_mask[margin_mask > 0] = 255
						ret, thresh = cv2.threshold(margin_mask, 127, 255, 0)
						contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
						cv2.drawContours(modal_x, contours, -1, (0,0,255), 2)


						fig, axes = plt.subplots(1, 2, figsize=(15, 5))
						axes[0].imshow(modal_x)
						axes[0].set_title("LRM")
						axes[1].imshow(orthoimage)
						axes[1].set_title("Orthoimage")


						#plt.show()
						plt.close(fig)


						

							

						




	result_line, (iou, mean_IoU, freq_IoU, mean_pixel_acc, pixel_acc) = compute_metric(all_results)
	with open('test_results/all_results.txt', 'w') as file:  
		file.write(result_line) 


	result_line, (iou, mean_IoU, freq_IoU, mean_pixel_acc, pixel_acc) = compute_metric(area_results)
	with open('test_results/area_results.txt', 'w') as file:  
		file.write(result_line) 

	result_line, (iou, mean_IoU, freq_IoU, mean_pixel_acc, pixel_acc) = compute_metric(margin_results)
	with open('test_results/margin_results.txt', 'w') as file:  
		file.write(result_line) 


	filtered_list = [item for item in all_detections if len(item) >= 3]

	polygons = [Polygon(coords) for coords in filtered_list]

	# Perform the union of all polygons
	all_unioned = unary_union(polygons)

	f = open('test_results/all_union.csv', 'w')
	writer = csv.writer(f)
	writer.writerow(['WKT', 'Id'])
	writer.writerow([str(all_unioned), "Castro"])
	f.close()




	filtered_list = [item for item in above_min_area if len(item) >= 3]

	polygons = [Polygon(coords) for coords in filtered_list]

	# Perform the union of all polygons
	area_unioned = unary_union(polygons)

	f = open('test_results/area_union.csv', 'w')
	writer = csv.writer(f)
	writer.writerow(['WKT', 'Id'])
	writer.writerow([str(area_unioned), "Castro"])
	f.close()



	filtered_list = [item for item in not_in_margins if len(item) >= 3]

	polygons = [Polygon(coords) for coords in filtered_list]

	# Perform the union of all polygons
	margin_unioned = unary_union(polygons)

	f = open('test_results/margin_union.csv', 'w')
	writer = csv.writer(f)
	writer.writerow(['WKT', 'Id'])
	writer.writerow([str(margin_unioned), "Castro"])
	f.close()



	print(len(all_unioned))
	print(len(area_unioned))
	print(len(margin_unioned))

	with open('test_results/poly_count.txt', 'w') as file:  
		file.write('All: ' + str(len(all_unioned)) +'\n')
		file.write('After area filter: ' + str(len(area_unioned)) +'\n')
		file.write('After area and margin filters: ' + str(len(margin_unioned)))




if __name__ == "__main__":
	main()
