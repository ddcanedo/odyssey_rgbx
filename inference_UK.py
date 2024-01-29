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
from tqdm import tqdm
from utils.pyt_utils import ensure_dir

csv.field_size_limit(sys.maxsize)
Image.MAX_IMAGE_PIXELS = None


# Converts coordinates from GIS reference to image pixels
def mapping(value, min1, max1, min2, max2):
	return (((value - min1) * (max2 - min2)) / (max1 - min1)) + min2


def predict(img, modal_x, device, val_func, confidence_threshold):
	preds = sliding_eval_rgbX(img, modal_x, config.eval_crop_size, config.eval_stride_rate, val_func, confidence_threshold, device)

	result_imgs = []
	for pred in preds:
		result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
		result_img = np.array(result_img)
		result_img[result_img > 0] = 255
		result_imgs.append(result_img)

	return result_imgs


# add new funtion for rgb and modal X segmentation
def sliding_eval_rgbX(img, modal_x, crop_size, stride_rate, val_func, confidence_threshold, device=None):
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

	processed_pred_tensor = torch.from_numpy(processed_pred)
	# Apply softmax across the channel dimension (last dimension in this case)
	probabilities = nn.functional.softmax(processed_pred_tensor, dim=2)
	probabilities = probabilities.numpy()

	mask = probabilities[:, :, 1:] <= confidence_threshold

	# Iterate over each class (except background) and apply the mask
	for i in range(1, probabilities.shape[2]):  # Start from 1 to exclude background class
		probabilities[:, :, i][mask[:, :, i - 1]] = 0

	preds = []
	preds.append(processed_pred.argmax(2)) # append prediction with higher probability
	preds.append(probabilities.argmax(2)) # append prediction where classes except background need to be above a certain threshold probability

	return preds


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
	# change this path
	imagesFolder = "../ODYSSEY/Images/LRM/inglaterra/"

	resolution = 800

	device = 0
	model = load_model()

	wms_url = 'https://ogc.apps.midgard.airbusds-cint.com/apgb/wms?guid=044130b5-d695-421d-bbc1-8ccac75c2db2'#'https://cartografia.dgterritorio.gov.pt/ortos2021/service?service=wmts&request=getcapabilities'
	layers  = ['AP-25CM-GB-LATEST']#['DGT_Ortos2021']
	srs = 'EPSG:27700'#'EPSG:3763'
	# Connect to the WMS service
	wms = WebMapService(wms_url)


	area_threshold = 1480
	margin_threshold = 50 # pixels
	confidence_threshold = 0.95

	# Define the margin rectangles (top, bottom, left, right)
	margin_rects = [
		(0, 0, resolution, margin_threshold),  # Top margin
		(0, resolution - margin_threshold, resolution, resolution),  # Bottom margin
		(0, 0, margin_threshold, resolution),  # Left margin
		(resolution - margin_threshold, 0, resolution, resolution)  # Right margin
	]


	# Iterates over the images
	for lrm in os.listdir(imagesFolder):
		if lrm.split('.')[-1] != 'tif':
			continue
		
		predictions1 = []
		predictions2 = []

		print(imagesFolder + lrm)
		image = imagesFolder + lrm

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

		
		# Sliding window going through the extent of interest
		for i in tqdm(range(0, width, resolution//2)):
			for j in tqdm(range(0, height, resolution//2)):

				crop = (i, resolution+i, j, resolution+j)

				cropped_img = img.crop((i, j, resolution+i, resolution+j))

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

				masks = predict(orthoimage, modal_x, device, model, confidence_threshold)

				for z in range(len(masks)):
					ret, thresh = cv2.threshold(np.array(masks[z]), 127, 255, 0)
					contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

						# area filter
						area = cv2.contourArea(c)
						if area >= area_threshold:
							# margin filter
							margin_valid = valid_margin_contour(c, area, resolution, margin_rects, threshold=0.5)
							if margin_valid:
								if z == 0:
									predictions1.append(polygon)
								else:
									predictions2.append(polygon)


		filtered_list = [item for item in predictions1 if len(item) >= 3]

		polygons = [Polygon(coords) for coords in filtered_list]

		# Perform the union of all polygons
		margin_unioned = unary_union(polygons)

		saving_dir = 'test_results/uk_results_50/'
		ensure_dir(saving_dir)

		f = open(saving_dir + lrm.split('.')[0] + '.csv', 'w')
		writer = csv.writer(f)
		writer.writerow(['WKT', 'Id'])
		writer.writerow([str(margin_unioned), "Castro"])
		f.close()



		filtered_list = [item for item in predictions2 if len(item) >= 3]

		polygons = [Polygon(coords) for coords in filtered_list]

		# Perform the union of all polygons
		margin_unioned = unary_union(polygons)

		saving_dir = 'test_results/uk_results_' + str(int(confidence_threshold*100)) + '/'
		ensure_dir(saving_dir)

		f = open(saving_dir + lrm.split('.')[0] + '.csv', 'w')
		writer = csv.writer(f)
		writer.writerow(['WKT', 'Id'])
		writer.writerow([str(margin_unioned), "Castro"])
		f.close()


if __name__ == "__main__":
	main()