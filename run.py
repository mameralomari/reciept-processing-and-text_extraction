import pipeline 

camera= str(input("would you like to use your camera? y/n "))
if camera == "y":
    df=pipeline.extract_using_camera()
    print (df)
else:
    image= input("please input image path or url ")
    df=pipeline.extract_from_image(image)
    print(df)
    