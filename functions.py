import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pytesseract 
from PIL import Image, ImageOps
import re
from commonregex import CommonRegex
import pgeocode
import cv2





def extract_text(url):
    import pytesseract 
    from PIL import Image
    
    """extracts test out of image
    --inputs= image_url """
    #opening the image
    

    frame=cv2.imread(url)
    frame= im.convert(mode="L")
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
   
    #imgThreshold = cv2.adaptiveThreshold(frame,255,cv2.THRESH_OTSU)
    
    config = ('-l eng --oem 1 --psm 3')
    text = pytesseract.image_to_string(frame , config=config)
    
    return text



def extract_data(text):

        def extract_prices(text):
            prices = CommonRegex(text).prices
            if len(prices) > 1:
                return prices
            else:
                return np.nan


            return CommonRegex(text).prices

        
        def extract_geo(text):
            #zip_code 
            us_zip = r'(\d{5}\-?\d{0,4})'
            zip_code = re.search(us_zip, text)
            try:
                zip_code = zip_code[0]
                nomi = pgeocode.Nominatim('us')
                geodata = nomi.query_postal_code(zip_code)
                longitude = geodata['longitude']
                latitude = geodata['latitude']
            except:
                zip_code = np.nan
                longitude = np.nan
                latitude = np.nan


            address = CommonRegex(text).street_addresses
            if len(address) == 1:
                address =  address[0]
            else:
                address = np.nan


            return zip_code, longitude, latitude, address
        geos=[]
        geos.append(extract_geo(text))

        zip_code, longitude, latitude, address= extract_geo(text)
        df = pd.DataFrame(data = geos, columns = ['Zipcode','Latitude', 'Longitude', 'Address'])

        df['prices'] = [extract_prices(text)]
        

        
        def extract_time(text):
            times = CommonRegex(text).times
            if len(times) == 1:
                return times[0]
            else:
                return np.nan
            
        df['time']= extract_time(text)
        
        return df
   
def clean_image(pathImage):
    import cv2
    import numpy as np
    import utlis

    #cap.set(10,160)
    heightImg = 1920
    widthImg  = 1080

    
    #pathImage = r"images/large-receipt-image-dataset-SRD/1013-receipt.jpg"

    img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF  REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
    #thres=utlis.valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.adaptiveThreshold(imgBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2) # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

        ## FIND ALL COUNTOURS
    imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS


        # FIND THE BIGGEST COUNTOUR
    biggest, maxArea = utlis.biggestContour(contours) # FIND THE BIGGEST CONTOUR
    if biggest.size != 0:
            biggest=utlis.reorder(biggest)
            cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
            imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)
            pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            #REMOVE 20 PIXELS FORM EACH SIDE
            imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
            imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))
           
    count=1
    cv2.imwrite("Scanned/myImage"+str(count)+".jpg",imgWarpColored)
    item_name=("Scanned/myImage"+str(count)+".jpg")
    return item_name


