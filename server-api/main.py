from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Image crack detection library
import uuid
import cv2
import math
import numpy as np
import scipy.ndimage

#CORS WHITELIST URLs
origins = [
    "http://localhost:3000",
]

app = FastAPI()
# CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def process_uploaded_image(imageDir):

    # return imageDir

    def orientated_non_max_suppression(mag, ang):
      ang_quant = np.round(ang / (np.pi/4)) % 4
      winE = np.array([[0, 0, 0],[1, 1, 1], [0, 0, 0]])
      winSE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
      winS = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
      winSW = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

      magE = non_max_suppression(mag, winE)
      magSE = non_max_suppression(mag, winSE)
      magS = non_max_suppression(mag, winS)
      magSW = non_max_suppression(mag, winSW)

      mag[ang_quant == 0] = magE[ang_quant == 0]
      mag[ang_quant == 1] = magSE[ang_quant == 1]
      mag[ang_quant == 2] = magS[ang_quant == 2]
      mag[ang_quant == 3] = magSW[ang_quant == 3]
      return mag

    def non_max_suppression(data, win):
        data_max = scipy.ndimage.filters.maximum_filter(data, footprint=win, mode='constant')
        data_max[data != data_max] = 0
        return data_max

    # start calulcation
    gray_image = cv2.imread(r'storage/crack1.jpeg', 0) ### CRACK IMAGE TO BE PROCESSED (VALUE WILL BE IMAGE PATH)

    with_nmsup = True #apply non-maximal suppression
    fudgefactor = 1.3 #with this threshold you can play a little bit
    sigma = 21 #for Gaussian Kernel
    kernel = 2*math.ceil(2*sigma)+1 #Kernel size

    gray_image = gray_image/255.0
    blur = cv2.GaussianBlur(gray_image, (kernel, kernel), sigma)
    gray_image = cv2.subtract(gray_image, blur)

    # compute sobel response
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(sobelx, sobely)
    ang = np.arctan2(sobely, sobelx)

    # threshold
    threshold = 4 * fudgefactor * np.mean(mag)
    mag[mag < threshold] = 0

    temp_filename_uuid = str(uuid.uuid4().hex)
    temp_filename = f"{temp_filename_uuid}-wallcrack.jpg"
    temp_file_directory = './storage/'

    #either get edges directly
    if with_nmsup is False:
        mag = cv2.normalize(mag, 0, 255, cv2.NORM_MINMAX)
        kernel = np.ones((5,5),np.uint8)
        result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(f"{temp_file_directory}/{temp_filename}", result)
        # cv2.imshow('im', result)
        # cv2.waitKey()

        return f"{temp_file_directory}/{temp_filename}"

    #or apply a non-maximal suppression
    else:

        # non-maximal suppression
        mag = orientated_non_max_suppression(mag, ang)
        # create mask
        mag[mag > 0] = 255
        mag = mag.astype(np.uint8)

        kernel = np.ones((5,5),np.uint8)
        result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)

        
        cv2.imwrite(f"{temp_file_directory}/{temp_filename}", result)

        # cv2.imshow('im', result)
        # cv2.waitKey()

        return f"{temp_file_directory}/{temp_filename}"

# ---------------------------------------------------------------------------------------------------------- #

class ImageModel(BaseModel):
  filetype: str

@app.post('/api/v1/image/process')
async def process_image(file: UploadFile):
  try:
    temp_uploaded_file_location = f"storage/{file.filename}"
    with open(temp_uploaded_file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    return FileResponse(process_uploaded_image(temp_uploaded_file_location))

    # return {
    #   "message": f"file '{file.filename}' saved at '{temp_uploaded_file_location}'"
    # }
  except Exception as error:
    return {
      "error": repr(error)
    }
