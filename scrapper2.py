import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

import random
import string

filename_length = 12 


folderPath = r"data_object_detection\test\labelTxt"
ImagePath = r"data_object_detection\test\images"
outputPath = r"output_images"

files = os.listdir(folderPath)[:200]
keywords = ['bed', 'chair', 'couch', 'table', 'refrigerator', 'tv']

for fileName in files:
    filePath = os.path.join(folderPath, fileName)
    fileRootName, fileExtension = os.path.splitext(fileName)
    result = []

    with open(filePath, 'r') as file:
        for line in file:
            text = line.split(" ")
            for word in reversed(text):
                if word in keywords:
                    result.append([text[0:len(text) - 1], fileRootName])
                    break

    if result:
        for i in range(len(result)):
            coordinates = result[i][0]
            xCoordinate = []
            yCoordinate = []

            for line in range(0, len(coordinates) - 1, 2):
                xCoordinate.append(float(coordinates[line]))
                yCoordinate.append(float(coordinates[line + 1]))

            if not fileRootName.endswith(".jpg"):
                fileRootName += ".jpg"

            ImageFile = os.path.join(ImagePath, fileRootName)
            coord = []
            newFilename = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(filename_length)) + '.jpg'

            for i in range(len(xCoordinate)):
                coord.append([xCoordinate[i], yCoordinate[i]])

            srcPoints = np.array(coord, dtype=np.float32)
            image = cv2.imread(ImageFile)
            mask = np.zeros_like(image)

            cv2.fillPoly(mask, [srcPoints.astype(int)], (255, 255, 255))
            extracted_region = cv2.bitwise_and(image, mask)

            height, width, _ = extracted_region.shape

            black_image = np.zeros_like(image)

            x_position = (black_image.shape[1] - width) // 2
            y_position = (black_image.shape[0] - height) // 2

            black_image[y_position:y_position + height, x_position:x_position + width] = extracted_region

            cv2.imwrite(f"data/object/{newFilename}", black_image)

            mask = np.zeros_like(image)
            cv2.fillPoly(mask, [srcPoints.astype(int)], (255, 255, 255))
            inverted_mask = cv2.bitwise_not(mask)

            masked_image = cv2.bitwise_and(image, inverted_mask)

            cv2.imwrite(f"data/mask/{newFilename}", masked_image)
            cv2.imwrite(f"data/original/{newFilename}", image)
