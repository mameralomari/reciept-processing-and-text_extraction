import pandas as pd
import numpy as np

import pytesseract 
from PIL import Image, ImageOps
import re
from commonregex import CommonRegex
import pgeocode
import cv2

# function that gets thresh-hold 

def get_threshhold(image):   
    # Set total number of bins in the histogram
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Get normalized histogram if it is required
    if is_normalized:
        hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    print("Otsu's algorithm implementation thresholding result: ", threshold)
    return threshold





def extract_text(url):
    import pytesseract 
    from PIL import Image
    
    """extracts test out of image
    --inputs= image_url """
    #opening the image
    

    frame=cv2.imread(url)
    #grey= im.convert(mode="L")
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    #thresh=get_threshhold(r"images/large-receipt-image-dataset-SRD/1084-receipt.jpg")
    #kernel = np.ones((1, 1), np.uint8)
    #image = cv2.dilate(grey, kernel, iterations=1)
    #kernel = np.ones((1, 1), np.uint8)
    #image = cv2.erode(image, kernel, iterations=1)
    #image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    #image = cv2.medianBlur(image, 3)
    #imgThreshold = cv2.adaptiveThreshold(frame,255,cv2.THRESH_OTSU)
    #ret3,th3 = cv2.threshold(grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    config = ('-l eng --oem 3 --psm 6')
    text = pytesseract.image_to_string(grey, config=config)
    
    return text


def extract_data(text):
    
    """ this function extracts data from the string using common regex, the output is a pandas dataframe of address,zipcode,prices, and latitude and longitude
    input= text produced by extraction functions""" 

   
    def extract_prices(text):
        if len(text) == 0:
            return np.nan
        
        prices = CommonRegex(text).prices
        if len(prices) > 0:
            return max(prices)
        else:
            return np.nan

        
    def extract_geo(text):
        if len(text) == 0:
            return np.nan, np.nan, np.nan, np.nan
        
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
        
    def extract_time(text):
        if len(text) == 0:
            return np.nan
        times = CommonRegex(text).times
        if len(times) == 1:
            return times[0]
        else:
            return np.nan
        
    def extract_store_names(text):
        string = text.strip()
        if len(string) >0:
            return re.search('[^(\n)]+', string, flags=0)[0]
        else:
            return np.nan
    
    geos=[]
    
    geos.append(extract_geo(text))

    zip_code, longitude, latitude, address= extract_geo(text)
    
    df = pd.DataFrame(data = geos, columns = ['Zipcode','Latitude','Longitude','Address'])
    
    df['prices'] = [extract_prices(text)]
         
    df['time']= extract_time(text)
    
    df['Store Names'] = extract_store_names(text)
        
    return df


def clean_image(pathImage):
    
    """this function tries to cut the reciept out, but if the reciept was touching the edges of the image, the crooping wouldn't be possible, instead the function witll produce a clean cut larger reciept image
    Input = Image Path"""
    import cv2
    import numpy as np
    import utlis

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


def run_camera():
    """ this function runs the camera and snaps a picture of the item when the letter (S) is pressed, 
    what it tries to do is find the largest contour that is shaped like a square and then once it is identified, you can snap a shot with it"""
    
    import cv2
    import numpy as np
    import utlis


    ########################################################################
    webCamFeed = True
    pathImage = "1.jpg"
    cap = cv2.VideoCapture(0)
    cap.set(10,160)
    heightImg = 1920
    widthImg  = 1080
    ########################################################################

    utlis.initializeTrackbars()
    count=0

    while True:

        if webCamFeed:success, img = cap.read()
        else:img = cv2.imread(pathImage)
        img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
        imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
        thres=utlis.valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS
        imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1]) # APPLY CANNY BLUR
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

            # APPLY ADAPTIVE THRESHOLD
            imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
            imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
            imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
            imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)

            # Image Array for Display
            imageArray = ([imgContours,imgWarpColored])

        else:
            imageArray = ([imgContours],
                          [imgBlank])

        # LABELS FOR DISPLAY
        lables = [["Original","Gray","Threshold","Contours"],
                  ["Biggest Contour","Warp Prespective","Warp Gray","Adaptive Threshold"]]

        stackedImage = utlis.stackImages(imageArray,0.75,lables)
        cv2.imshow("Result",stackedImage)

        # SAVE IMAGE WHEN 's' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("Scanned/myImage"+str(count)+".jpg",imgWarpColored)
            item_name=("Scanned/myImage"+str(count)+".jpg")
            cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                          (1100, 350), (0, 255, 0), cv2.FILLED)
            cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                        cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.imshow('Result', stackedImage)
            count += 1
            cv2.destroyAllWindows()
    
            return item_name 





def extract_using_camera():
    """this function extracts images out of Camera_Feed and run it through py tesseract then extract text out of the image""" 
    item=run_camera()
    text=extract_text(item)
    df=extract_data(text)
    
    return df , text



def extract_from_image(url):
    """ Extracts text from images
    inputs= image_url
    """
    
    image=clean_image(url)
    text=extract_text(image)

    df=extract_data(text)
    
    return df ,text
    
    

print('hello')