from imageai.Detection import ObjectDetection
import os
import cv2

execution_path = "D:\\HackUTD\\couldWork" #wherever your "yolov.pt" file is


    

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not found or could not be opened.")
else:
    ret,frame=cap.read()


#user chooses file to test here
fileCarousel=["R2.jpg","R1.jpg","C2.jpg","J1.jpg","J3.jpg","P2.jpg","R1.jpg"]


"""
folder_path='C:\\path\\to\\picture\\folder'

files=os.listdir(folder_path)
files=[os.path.join(folder_path,file) for file in files if os.path.isfile(os.path.join(folder_path,file))]

if files:
    latest_file= max(files, key= os.path.getmtime)
"""
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolov3.pt"))
detector.loadModel()




detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "R2.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

items = ["plant","clock","vase","chair","table","dining table","person","people"]
count=0
for eachObject in detections:
    if eachObject["name"] in items:
        count+=1 + items.index(eachObject["name"])
    #print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

if count != 0:
    print("your buisness has...",count,"...value!")
