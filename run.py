import functions 

def run():

    camera= str(input("would you like to use your camera? y/n "))
    if camera == "y":
        df=functions.extract_using_camera()
        print (df)
    else:
        image= input("please input image path or url ")
        df=functions.extract_from_image(image)
        print(df)
    