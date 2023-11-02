import os
import numpy as np
import cv2

folderPath = r"data_object_detection\test\labelTxt"
ImagePath = r"data_object_detection\test\images"
outputPath = r"output_images"

files = os.listdir(folderPath)[:5]
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
            coordinates = [float(val) for val in result[i][0]
            if not fileRootName.endswith(".jpg"):
                fileRootName += ".jpg"
            ImageFile = os.path.join(ImagePath, fileRootName)
            
            coord = []
            for i in range(len(coordinates)):
                coord.append([xCoordinate[i], yCoordinate[i])

            srcPoints = np.array(coord, dtype=np.float32)
            image = cv2.imread(ImageFile)
            mask = np.zeros_like(image, dtype=np.uint8)

            cv2.fillPoly(mask, [srcPoints.astype(int)], (255, 255, 255, 0))  # Transparent mask

            # Apply the transparent mask to the original image
            masked_image = cv2.bitwise_and(image, mask)

            # Save the masked image with transparency
            output_image_path = os.path.join(outputPath, f'masked_{fileRootName}')
            cv2.imwrite(output_image_path, masked_image)
