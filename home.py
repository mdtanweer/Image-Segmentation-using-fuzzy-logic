from tkinter import *
import tkinter.font as font
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageTk

top=Tk()
def greyscale():
    image = cv2.imread(r"panda.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.subplot(121),plt.imshow(image,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(gray,cmap = 'gray')
    plt.title('Grayscale Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def edge():
    image = cv2.imread(r"panda.jpg")

    edges = cv2.Canny(image,100,200)

    plt.subplot(121),plt.imshow(image,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def face():
    def nothing(x):
        pass

    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    cv2.namedWindow("Frame")
    cv2.createTrackbar("Neighbours", "Frame", 5, 20, nothing)

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        neighbours = cv2.getTrackbarPos("Neighbours", "Frame")

        faces = face_cascade.detectMultiScale(gray, 1.3, neighbours)
        for rect in faces:
            (x, y, w, h) = rect
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        cv2.imshow("Frame", frame)

        if cv2.waitKey(3)==ord('q'):
            break


    # Destroys all of the HighGUI windows. 
    cv2.destroyAllWindows() 
      
    # release the captured frame 
    cap.release() 

def color():
    nemo = cv2.imread(r'fish.PNG')
    plt.imshow(nemo)
    plt.show()


    nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
    plt.imshow(nemo)
    plt.show()


    hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)

    light_orange = (10,190 ,200)
    dark_orange = (255, 255, 255)



    mask = cv2.inRange(hsv_nemo, light_orange, dark_orange)
    result = cv2.bitwise_and(nemo, nemo, mask=mask)

    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.show()


    light_white = (0, 0, 200)
    dark_white = (145, 60, 255)

    mask_white = cv2.inRange(hsv_nemo, light_white, dark_white)
    result_white = cv2.bitwise_and(nemo, nemo, mask=mask_white)

    plt.subplot(1, 2, 1)
    plt.imshow(mask_white, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(result_white)
    plt.show()


    final_mask = mask + mask_white

    final_result = cv2.bitwise_and(nemo, nemo, mask=final_mask)
    plt.subplot(1, 2, 1)
    plt.imshow(final_mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(final_result)
    plt.show()

def foreground():
    img = cv2.imread(r'panda.jpg')
    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (50,50,450,290)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    plt.imshow(img),plt.colorbar(),plt.show()

def imagearray():
    img = cv2.imread("simpsons.jpg")
    averaging = cv2.blur(img, (21, 21))
    gaussian = cv2.GaussianBlur(img, (21, 21), 0)
    median = cv2.medianBlur(img, 5)
    bilateral = cv2.bilateralFilter(img, 9, 350, 350)

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(img)
    axarr[0,1].imshow(averaging)
    axarr[1,0].imshow(gaussian)
    axarr[1,1].imshow(median)

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def corner():
    cap = cv2.VideoCapture(0)
    def nothing(x):
        pass
    cv2.namedWindow("Frame")
    cv2.createTrackbar("quality", "Frame", 1, 100, nothing)
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quality = cv2.getTrackbarPos("quality", "Frame")
        quality = quality / 100 if quality > 0 else 0.01
        corners = cv2.goodFeaturesToTrack(gray, 100, quality, 20)
        if corners is not None:
            corners = np.int0(corners)
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(3)&0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
def template():
    img = cv2.imread("simpsons.jpg")
    #cv2.imshow("img", img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread("barts_face.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("img", template)
    w, h = template.shape[::-1]
    result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= 0.4)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def home():
    top.iconbitmap("gui.ico")
    top.title("Home Page")
    top.geometry("1120x750")
    top.configure(bg='sky blue')
    top.resizable(0,0)
    photos = PhotoImage(file="images.png")
    label = Label(top, image=photos)
    label.place(x=0, y=0)
    
    label1=Label(top,text="IMAGE SEGMENTATION",bg='white',font=('bold',60))
    label1.place(x=100,y=10)
    
    photo=PhotoImage(file="image.png")
    
    label = Label(top,image=photo,height=30,width=30)
    label.place(x=100,y=100)

    label = Label(top,image=photo,height=30,width=30)
    label.place(x=140,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=180,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=220,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=260,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=300,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=340,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=380,y=100)
    
    label=Label(top,image=photo,height=30,width=30)
    label.place(x=420,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=460,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=500,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=540,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=580,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=620,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=660,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=700,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=740,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=780,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=820,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=860,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=900,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=940,y=100)

    label=Label(top,image=photo,height=30,width=30)
    label.place(x=980,y=100)

    label=Label(top,text="Face detection from webcam",font=('bold',20))
    label.place(x=120,y=180)
    button3=Button(top,text="      Face        ",bg='yellow',fg='blue',font=('bold',10),command=face)
    button3.place(x=250,y=230)

    label=Label(top,text="Edge Detection of an image",font=('bold',20))
    label.place(x=660,y=180)
    button2=Button(top,text="Edge detection",bg='yellow',fg='blue',font=('bold',10),command=edge)
    button2.place(x=810,y=230)

    label=Label(top,text="Check the gray scale of an image",font=('bold',20))
    label.place(x=100,y=280)
    button1=Button(top,text="Grey Scale    ",bg='yellow',fg='blue',font=('bold',10),command=greyscale)
    button1.place(x=250,y=330)

    label=Label(top,text="Different format of an image",font=('bold',20))
    label.place(x=660,y=280)
    button6=Button(top,text=" Imaginearray ",bg='yellow',fg='blue',font=('bold',10),command=imagearray)
    button6.place(x=810,y=330) 

    label=Label(top,text="Elemenating background of image",font=('bold',20))
    label.place(x=100,y=380)
    button5=Button(top,text=" Foreground ",bg='yellow',fg='blue',font=('bold',10),command=foreground)
    button5.place(x=250,y=430)

    label=Label(top,text="Template matching of an image",font=('bold',20))
    label.place(x=640,y=380)
    button8=Button(top,text="Template Match",bg='yellow',fg='blue',font=('bold',10),command=template)
    button8.place(x=810,y=430)

    label=Label(top,text="Detect corner of image by webcam",font=('bold',20))
    label.place(x=100,y=480)
    button7=Button(top,text="     Corner     ",bg='yellow',fg='blue',font=('bold',10),command=corner)
    button7.place(x=250,y=530)

    label=Label(top,text="Check the gray scale of an image",font=('bold',20))
    label.place(x=640,y=480)
    button4=Button(top,text=" Multiple color ",bg='yellow',fg='blue',font=('bold',10),command=color)
    button4.place(x=810,y=530)
    

    button9=Button(top,text="back",bg='white',fg='blue',font=('bold',10),command=front)
    button9.place(x=10,y=600)
    top.mainloop()
def front():
    top.iconbitmap("gui.ico")
    top.title("Front Page")
    top.geometry("1120x750")
    top.configure(bg='white')
    top.resizable(0,0)
    photos = PhotoImage(file="images2.png")
    label = Label(top, image=photos)
    label.place(x=0, y=0)
    
    label1=Label(top,text="SOFT COMPUTING PROJECT",bg='white',font=('bold',40))
    label1.place(x=150,y=10)

    image = Image.open("ai.png")
    photo1 = ImageTk.PhotoImage(image)
    label=Label(top,image=photo1)#,height=250,width=416)
    label.place(x=45,y=80)
   
    button1=Button(top,text="click to continue",command=home,bg='white',fg='blue',font=('bold',10))
    button1.place(x=935,y=85)
   
    label1=Label(top,text="Name ",fg='red',bg='white',font=('bold',20))
    label1.place(x=730,y=550)
    label1=Label(top,text="Reg No",fg='red',bg='white',font=('bold',20))
    label1.place(x=920,y=550)
    
    label1=Label(top,text="Md Tanweer",bg='white',font=('bold',15))
    label1.place(x=730,y=590)
    label1=Label(top,text="11705659",bg='white',font=('bold',15))
    label1.place(x=920,y=590)
    label1=Label(top,text="Nitesh Kumar",bg='white',font=('bold',15))
    label1.place(x=730,y=620)
    label1=Label(top,text="11709293",bg='white',font=('bold',15))
    label1.place(x=920,y=620)
    label1=Label(top,text="Prashat Sirohi",bg='white',font=('bold',15))
    label1.place(x=730,y=650)
    label1=Label(top,text="11709395",bg='white',font=('bold',15))
    label1.place(x=920,y=650)
    label1=Label(top,text="Shivmangal singh",bg='white',font=('bold',15))
    label1.place(x=730,y=680)
    label1=Label(top,text="11709300",bg='white',font=('bold',15))
    label1.place(x=920,y=680)

    label1=Label(top,text="Submitted to",fg='red',bg='white',font=('bold',20))
    label1.place(x=60,y=650)
    label1=Label(top,text="Dr Ashmita Pandey",bg='white',font=('bold',15))
    label1.place(x=60,y=690)

    top.mainloop()
front()

