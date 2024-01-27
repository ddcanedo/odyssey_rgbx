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


def count_polys(annotations):
	count = 0
	for key in annotations:
		geometry = loads(annotations[key])
		if isinstance(geometry, MultiPolygon):
			for polygon in geometry:
				count+=1
				
	return count



def main():

	og_path = '../test_results/og_results/og_union.csv'
	#refined_path = '../PNPG_hillforts_inference.csv'

	annotations = {}
	with open(og_path) as csvfile:

		reader = csv.DictReader(csvfile)  # Using DictReader to access columns by name
		for row in reader:
			annotations['castro'] = row['WKT']
	


	count = count_polys(annotations)

	print(count)
		


if __name__ == "__main__":
	main()