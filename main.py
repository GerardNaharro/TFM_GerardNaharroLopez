# TFM: FOOTBALL TRACKING AND ANALYTICS

import os
import cv2
from roboflow import Roboflow
from ultralytics import YOLO

if __name__ == '__main__':
    HOME = os.getcwd()
    #print(HOME)
    clips = "D:\clips_futbol"


    # Load a model
    #model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

    #rf = Roboflow(api_key="6rs9HD0Q8xmCqLykRvot")

    #project = rf.workspace("universitat-de-les-illes-balears").project("tfm_football_tracking")
    #dataset = project.version(1).download("yolov8")

    #project = rf.workspace("nikhil-chapre-xgndf").project("detect-players-dgxz0")
    #dataset = project.version(6).download("yolov8")

    dataset_location = HOME + "\Detect-Players--6\data.yaml"

    '''
    yolo task=detect \
    mode=train \
    model=yolov8s.pt \
    data={dataset.location}/data.yaml \
    epochs=100 \
    imgsz=640 # Must be a 32 multiple
    '''

    #training the model
    #results = model.train(data = dataset_location, imgsz = 1216, epochs = 465, patience = 0, batch = -1, name = "Segunda_prueba_larga_nuevoEntreno_S_autobatch")



    own_trained_location = HOME + "/runs\detect\Segunda_prueba_larga_nuevoEntreno_S_autobatch\weights/best.pt"
    # Load the own trained model
    model = YOLO(own_trained_location)


    # Define path to video file
    source = clips + '\clip8.mp4'


    results = model.track(source= source, show=True, tracker="bytetrack.yaml", save=False, line_width = 2, iou = 0.7, conf = 0.25, persist=False)

    """cap = cv2.VideoCapture(source)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            #results = model(frame)

            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, show=False, save=True, tracker="botsort.yaml")


            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()"""
