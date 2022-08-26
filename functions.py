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
    #grey= im.convert(mode="L")
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
   
    
    
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
   