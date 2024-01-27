import os
import sys
import csv
import numpy as np 
from PIL import Image, ImageDraw
import rasterio
import csv
from shapely.wkt import loads
from shapely.geometry import Point, Polygon, MultiPolygon
import random
from owslib.wms import WebMapService
import cv2

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
					for p in polygonPoints:
						xPoly = mapping(p[0], xMinImg, xMaxImg, 0, width)
						yPoly = mapping(p[1], yMinImg, yMaxImg, height, 0)
						polygon.append([xPoly, yPoly])

					bbs[(xMinBb, xMaxBb, yMinBb, yMaxBb)] = [key, polygon]

	return bbs


# Converts coordinates from GIS reference to image pixels
def mapping(value, min1, max1, min2, max2):
	return (((value - min1) * (max2 - min2)) / (max1 - min1)) + min2


# Returns a list with 100% visible objects and their respective labels
def checkVisibility(image, crop, processedObjects, bbs):
	visibleObjects = []

	for bb in bbs:
		# The object is 100% inside the cropped image
		if bb[0] >= crop[0] and bb[1] <= crop[1] and bb[2] >= crop[2] and bb[3] <= crop[3]:
			# The object was already processed
			if bb in processedObjects:
				return []
			else:
				visibleObjects.append(bb)
		# The object is 100% outside the cropped image
		elif bb[1] < crop[0] or bb[0] > crop[1] or bb[3] < crop[2] or bb[2] > crop[3]:
			continue
		# The object is partially visible
		else:
			return []

	# Update list of processed objects
	processedObjects.extend(visibleObjects)
	return visibleObjects



# Creates a dataset folder in YOLO format
def createDatasetDir(datasetPath, LRMPath, RGBPath, masksPath):
	if not os.path.exists(datasetPath):
		os.makedirs(datasetPath)
		os.makedirs(LRMPath)
		os.makedirs(RGBPath)
		os.makedirs(masksPath)
	else:
		if not os.path.exists(LRMPath):
			os.makedirs(LRMPath)
		if not os.path.exists(RGBPath):
			os.makedirs(RGBPath)	
		if not os.path.exists(masksPath):
			os.makedirs(masksPath)


def main():
	random.seed(4)
	# loads annotations and paths
	annotationsFolder = "annotations/" 
	imagesFolder = "../ODYSSEY/Images/LRM/"
	csvs = os.listdir(annotationsFolder)


	# Creates dataset
	datasetPath = "dataset/"
	LRMPath = datasetPath + "LRM/"
	RGBPath = datasetPath + "RGB/"
	masksPath = datasetPath + "Masks/"

	createDatasetDir(datasetPath, LRMPath, RGBPath, masksPath)


	# Standard resolution
	resolution = 800 




	# Iterates over the images
	for csvAnnotation in csvs:

		annotation = annotationsFolder + csvAnnotation

		if csvAnnotation == 'inglaterra.csv':
			wms_url = 'https://ogc.apps.midgard.airbusds-cint.com/apgb/wms?guid=044130b5-d695-421d-bbc1-8ccac75c2db2'
			layers  = ['AP-25CM-GB-LATEST']
			srs = 'EPSG:27700'
		else:
			continue
		'''elif csvAnnotation == 'galiza.csv':
			# Define the WMS service URL
			wms_url = 'https://www.ign.es/wms-inspire/pnoa-ma?request=GetCapabilities&service=WMS'
			layers  = ['OI.OrthoimageCoverage']
			srs = 'EPSG:25829
		else:
			wms_url = 'https://cartografia.dgterritorio.gov.pt/ortos2021/service?service=wmts&request=getcapabilities'
			layers  = ['DGT_Ortos2021']
			srs = 'EPSG:3763'''
			

		# Connect to the WMS service
		wms = WebMapService(wms_url)


		for image in os.listdir(imagesFolder + csvAnnotation.split('.')[0]):
			print(imagesFolder+csvAnnotation.split('.')[0] + '/' + image)
			if(image.startswith('Exeter')):
				continue
			image = imagesFolder+csvAnnotation.split('.')[0] + '/' + image
						

			# List to save all the objects that are processed to keep uniqueness
			processedObjects = []

			# Load image
			img = Image.open(image)

			geoRef = rasterio.open(image)
			width, height = img.size

			print(img.size)

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

			#print(annotations)

			# Transforms the polygons into bounding boxes
			bbs = poly2bb(annotations, xMinImg, xMaxImg, yMaxImg, yMinImg, width, height)

			print(len(bbs))
			#print(bbs)
			# Iterates over the bounding boxes
			for bb in bbs:
				# Check if bounding box is unique
				if bb not in processedObjects:
					# Randomizes a list of unique points covering a range around the object
					x = list(range(bb[1]-resolution//2, bb[0]+resolution//2))
					y = list(range(bb[3]-resolution//2, bb[2]+resolution//2))
					random.shuffle(x)
					random.shuffle(y)

					# List to save 100% visible objects
					visibleObjects = []
					crop = []

					# Iterates over the list of random points
					while len(x) > 0 and len(y) > 0:
						i = random.choice(x)
						j = random.choice(y)
						x.remove(i)
						y.remove(j)
						# Gets a region of interest expanding the random point into a region of interest with a certain resolution
						crop = (i-resolution//2, i+resolution//2, j-resolution//2, j+resolution//2)
						# Checks if that region of interest only covers 100% visible objects
						visibleObjects = checkVisibility(image, crop, processedObjects, bbs)
						if visibleObjects:
							break

					# If we obtain a list of visible objects within a region of interest, we save it
					if visibleObjects:
						


						# image extent to GIS
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

						# Uses the coordinates as the name of the image and text files
						coords = '('+ str(xMin) + '_' + str(xMax) + '_' + str(yMin) + '_' + str(yMax) + ')'

						imgName = RGBPath + image.split("/")[-1].split(".")[0] + coords + ".png"
						out = open(imgName, 'wb')
						out.write(orthoimage.read())
						out.close()


						# Writes the image
						croppedImg = img.crop((crop[0], crop[2], crop[1], crop[3]))
						imgName = LRMPath + image.split("/")[-1].split(".")[0] + coords + ".png"
						croppedImg.save(imgName)

						# Writes the mask
						mask = Image.new("L", croppedImg.size, 0)
						for obj in visibleObjects:						
							
							poly = bbs[obj][1]

							draw = ImageDraw.Draw(mask)

							cPoly = []
							for p in poly:
								cX = mapping(p[0], crop[0], crop[1], 0, resolution)
								cY = mapping(p[1], crop[2], crop[3], 0, resolution)
								cPoly.append((cX,cY))

							draw.polygon(cPoly, fill=255)
						
						maskName = masksPath + image.split("/")[-1].split(".")[0] + coords + ".png"
						mask.save(maskName)

						

			img.close()


if __name__ == "__main__":
	main()