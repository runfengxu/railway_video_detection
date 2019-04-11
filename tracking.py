from imageai.Detection import VideoObjectDetection
import os
import time
start = time.time()



execution_path = os.getcwd()

detector = VideoObjectDetection()

detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(os.path.join(execution_path,"yolo-tiny.h5"))
detector.loadModel()


video_path = detector.detectCustomObjectsFromVideo(input_file_path=os.path.join(execution_path,"video2.mp4"),output_file_path=os.path.join(execution_path,"detectedvideo2"),frames_per_second=30,log_progress=True)
print(video_path)

end = time.time()

print("\ntime:",end-start)

execution_path = os.getcwd()

detector2 = ObjectDetection()

detector2.setModelTypeAsRetinaNet()

detector2.setModelPath(os.path.join(execution_path,"resnet50_coco_best_v2.0.1.h5"))
detector2.loadModel()

detections = detector2.detectObjectsFromImage(input_image=os.path.join(execution_path,"unnamed.png"),output_image_path=os.path.join(execution_path,"image3new2.jpg"))


end = time.time()

position = []
item=[]

for eachObject in detections:
    position.append(eachObject["box_points"])
    item.append(eachObject['name'])
print("------------------")

item_list=pd.Series(position)
item_list.index=item