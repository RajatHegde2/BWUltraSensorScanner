from tkinter import *
from tkinter import ttk
import time
import tkinter as tk
import cv2
import os
import sys
import numpy as np
import tkinter.font as tkFont
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average
from tkinter import filedialog
from threading import Thread
root = Tk()
root.state("zoomed")  #to make it full screen

root.title("Alarm Detecting System")
root.configure(bg="grey80")

topFrame = Frame(root, width=1350, height=50,background="green")  # Added "container" Frame.
topFrame.pack(side=TOP, fill=X, expand=1, anchor=N)
customFont = tkFont.Font(family="Helvetica", size=18)

titleLabel = Label(topFrame, font=('arial', 12, 'bold'),
                   text="Alarm Detecting System",
                   bd=5, anchor=W)
titleLabel.pack(side=LEFT)

clockFrame = Frame(topFrame, width=100, height=50, bd=4, relief="ridge",background="green")
clockFrame.pack(side=RIGHT)
clockLabel = Label(clockFrame, font=('arial', 12, 'bold'), bd=5, anchor=E)
clockLabel.pack()

Bottom = Frame(root, width=1350, height=50, bd=4, relief="ridge")
Bottom.pack(side=BOTTOM, fill=X, expand=1, anchor=S)

def tick(curtime=''):  #acts as a clock, changing the label when the time goes up
    newtime = time.strftime('%H:%M:%S')
    if newtime != curtime:
        curtime = newtime
        clockLabel.config(text=curtime)
    clockLabel.after(200, tick, curtime)

tick()  #start clock
leftFrame = Frame(root,width=400,height=200,background="grey")  # Added "container" Frame.
leftFrame.pack(side=LEFT, fill=X, expand=1, anchor=N)
rightFrame = Frame(root,width=400,height=800,background="grey")  # Added "container" Frame.
rightFrame.pack(side=RIGHT, fill=X, expand=1, anchor=N)

text1 = tk.Text(leftFrame, height=2, width=70)
T = Text(rightFrame, height=800, width=100,font=customFont)
T.pack()
    
# This drives the program into an infinite loop.
def start_submit_thread():
    
    submit_thread = Thread(target=run)

    submit_thread.start()

def UploadFile(event=None):
    filename = filedialog.askopenfilename()
    text1.config(state="normal")
    text1.delete(1.0, "end-1c")
    text1.insert(tk.INSERT, str(filename))


def run():
    file = text1.get(1.0, "end-1c")
    
    #BEFORE RUN
    filename = file[file.rfind("/") : file.rfind(".") ]
    #filename = "Hi Alarm-4"
    path = file

    cap = cv2.VideoCapture(path)
    count = 1;
    # This drives the program into an infinite loop. 
    _, frame = cap.read()

    if not _:
        print("Cannot open the file")
        exit

    framesPath1 = "./"
    maskDir = "Mask\\"
    resDir = "Res\\"

    if not os.path.isdir(framesPath1+filename):
        os.mkdir(framesPath1+filename)

    if not os.path.isdir(framesPath1+filename+"\\"+maskDir):
        os.mkdir(framesPath1+filename+"\\"+maskDir)

    if not os.path.isdir(framesPath1+filename+"\\"+resDir):
        os.mkdir(framesPath1+filename+"\\"+resDir)


    while(_):
        
         
            # Converts images from BGR to HSV 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
        lower_red = np.array([150,50,50]) 
        upper_red = np.array([255,180,180]) 

    # Here we are defining range of given color in HSV 
    # This creates a mask of coloured(given) 
    # objects found in the frame. 
        mask = cv2.inRange(hsv, lower_red, upper_red) 
    # The bitwise and of the frame and mask is done so 
    # that only the  coloured objects are highlighted 
    # and stored in res 
        res = cv2.bitwise_and(frame,frame, mask= mask) 
        #cv2.imwrite("D:/Hkton/Python/SPS/outvid.mp4",res)
        cv2.imshow('frame',frame) 
        cv2.imshow('mask',mask) 
        cv2.imwrite(framesPath1+ filename+"\\"+maskDir+"frame%d.jpg" % count, mask)
        cv2.imwrite(framesPath1+filename+"\\"+resDir+"frame%d.jpg" % count, res)
        count = count +1
        cv2.imshow('res',res) 
    # This displays the frame, mask 
    # and res which we created in 3 separate windows. 
        #if(frame.empty()):
        #    break
        _, frame = cap.read()
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') : 
            break

    # Destroys all of the HighGUI windows. 
    cv2.destroyAllWindows() 

    # release the captured frame 
    cap.release() 
    p=time.time()
    Similarity(framesPath1+filename+"\\"+resDir,count-1)
    q=time.time()
    gily="Time taken :"+str(int(q-p))+" seconds\n"
    T.insert(END,gily)
    
    #AFTER RUN
   
def Similarity(path, noOfFrames):
    i=1;
    file2="Reference.jpg"
    f=open("Similarity.txt",'w')
    per = 0
    while(i <= noOfFrames):
        file1=path+"frame"+str(i)+".jpg"
        #file1="F:/main/wifi/myframe"+str(i)+".jpg"
        main(file1,file2,f)
        per = float(i/noOfFrames)
        i+=1
        per = int(per*100)
        T.delete('1.0',END)
        pika="Completed : "
        pika+=str(per)+"%\n"
        T.insert(END,pika)
    print("end Similarity")
    f.close()
    getFrequency(noOfFrames)
    #END SIM
def main(file1,file2,f):
    # read images as 2D arrays (convert to grayscale for simplicity)
    img = to_grayscale(imread(file1).astype(float))
    ref = to_grayscale(imread(file2).astype(float))
    # compare
    n_m, n_0 = compare_images(img, ref)
   # print ( n_m/img1.size)
    f.write(str(n_0*1.0/img.size))
    f.write("\n")
    
    
    
def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  
    m_norm = sum(abs(diff))  
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    return (m_norm, z_norm)

def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr
def normalize(arr):
    rng = arr.max()-arr.min()
    if rng==0:
        rng=1
    amin = arr.min()
    return (arr-amin)*255/rng

'''                                     Frequency                                           '''

def getFrequency(frameCount):
    path = "./Similarity.txt"
    f=open(path,"r")
    divisions(f, frameCount)
    f.close()
    #END GETFREQ

def divisions(f, frameCount):
    count = 0
    lines = 0
    while True:
        num1 = f.readline()
        if( lines >= frameCount):
            break
        n1 = float(num1)
        num2 = f.readline()
        lines = lines + 2
        if(lines > frameCount):
            break
        n2 = float(num2)
        if n1>n2 and n2 != 0:
            val = n1/n2
        elif n1!=0:
            val = n2/n1
        
        if(val > 5):
            count =count + 1
            print(val)
            print("\n")
    
    bika="Count :"+str(count)+ "\nFrameCount: "+str(frameCount)+"\n" 
    freq = float(count/frameCount)
    bika+="frequency :"+str(freq)+"\n\n"
    if freq > 0.058800:
        bika+="High Alarm!!\n"
    elif freq<0.0005:
        bika+="Reverse Intelliflash...\n"
    else :
        bika+="Low Alarm..\n"
    T.insert(END, bika)
    #END DIVISION

button1 = tk.Button(leftFrame,text='Browse File', command=UploadFile,font=customFont)
button1.pack()
text1.config(state="disabled")
text1.pack(padx=100)
button = tk.Button(leftFrame, height=1,width=5,text='Run', command=start_submit_thread,font=customFont)
button.pack()
root.mainloop()
