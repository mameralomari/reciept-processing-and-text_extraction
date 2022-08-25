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

def contour_to_rect(contour):
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    # top-left point has the smallest sum
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # compute the difference between the points:
    # the top-right will have the minumum difference 
    # the bottom-left will have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect / resize_ratio

def wrap_perspective(img, rect):
    # unpack rectangle points: top left, top right, bottom right, bottom left
    (tl, tr, br, bl) = rect
    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    # destination points which will be used to map the screen to a "scanned" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    # warp the perspective to grab the screen
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

def extract_text(url):
    import pytesseract 
    from PIL import Image
    
    """extracts test out of image
    --inputs= image_url """
    #opening the image
    

    frame=cv2.imread(url)
    #grey= im.convert(mode="L")
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    d = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])  
        
    #boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(blurred, rectKernel)
    edged = cv2.Canny(dilated, 100, 200, apertureSize=3)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    
    image_with_largest_contours = cv2.drawContours(edged.copy(), largest_contours, -1, (0,255,0), 3)

    largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    image_with_largest_contours = cv2.drawContours(frame.copy(), largest_contours, -1, (0,255,0), 3)
    def approximate_contour(contour):
        peri = cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, 0.032 * peri, True)
    
    def get_receipt_contour(contours):    
    # loop over the contours
        for c in contours:
            approx = approximate_contour(c)
            # if our approximated contour has four points, we can assume it is receipt's rectangle
            if len(approx) == 4:
                return approx
        
    receipt_contour = get_receipt_contour(largest_contours)
    
    #image_with_receipt_contour = cv2.drawContours(frame.copy(), [receipt_contour], -1, (0, 255, 0), 2)
    scanned = wrap_perspective(frame.copy(), contour_to_rect(receipt_contour))
    
    #threshold= 160
    #image=grey.point(lambda x: 255 if x > threshold else 0)
    
    #configuration for pytesseract to operate with text extraction
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    
    
    config = ('-l eng --oem 1 --psm 3')
    text = pytesseract.image_to_string(image_with_largest_contours , config=config)
    
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
   