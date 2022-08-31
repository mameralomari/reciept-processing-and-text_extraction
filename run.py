import functions 

print (" to run the program please run the command (run.run()) /n example df , text= run.run()")
def run():
    """ this function runs either the camera or read an image from a url
    outputs 
    text = string
    df= pd.dataframe""" 
    
    camera= str(input("would you like to use your camera? y/n "))
    if camera == "y":
        df,text=functions.extract_using_camera()
        print (df,text)
        return df ,text
    else:
        image= input("please input image path or url ")
        df,text=functions.extract_from_image(image)
        print(df,text)
        return df, text
    