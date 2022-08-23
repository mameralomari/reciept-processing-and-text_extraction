
def extract_text(url):
    import pytesseract 
    from PIL import Image
    
    """extracts test out of image
    --inputs= image_url """
    #opening the image
    im=Image.open(url)
    grey= im.convert(mode="L")
    #threshold= 160
    #image=grey.point(lambda x: 255 if x > threshold else 0)
    
    #configuration for pytesseract to operate with text extraction
    config = ('-l eng --oem 1 --psm 3')
    text = pytesseract.image_to_string(grey, config=config)
    
    return text

