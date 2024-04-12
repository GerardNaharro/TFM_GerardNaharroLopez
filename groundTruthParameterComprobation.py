import os
import json
import cv2
from ultralytics import YOLO
from kalmanfilter import KalmanFilter
import scipy.stats
from decimal import Decimal
import pandas as pd
import numpy as np
from itertools import product


metrics_threshold = 15


def update_XYpos(xPos, yPos, xelem, yelem):
    xPos.append(xelem)
    yPos.append(yelem)

    if len(xPos) > 5:
        xPos.pop(0)
        yPos.pop(0)


if __name__ == '__main__':
    HOME = os.getcwd()
    clips = "D:\clips_futbol\listos"

    own_trained_location = HOME + "/runs\detect\Segunda_prueba_larga_nuevoEntreno_S_autobatch\weights/best.pt"
    # Load the own trained model
    model = YOLO(own_trained_location)

    clip_name = '/RMAvsSEV.mp4'

    # Define path to video file
    video_path = clips + clip_name

    with open(HOME + "/newMetrics/" + clip_name[1:-4] + ".json", 'r') as f:
        groundTruth = json.load(f)



    #ballPos = ()
    '''VideoWidth = cap.get(3)  # float `width`
    VideoHeight = cap.get(4)  # float `height`'''


    m = np.arange(22, 35, 0.1)
    n = np.arange(0.45, 0.55, 0.025)
    num_iter = 0

    data = [['CLIP NAME', clip_name]]

    for param1, param2 in product(m, n):
        print(num_iter)
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        frameMetrics = []
        frameNumber = 0
        sd = 5
        xPos = []
        yPos = []
        yoloConfidence = 0
        previousBallPos = None
        ballPos = None
        yoloBallPos = None
        print(param1, param2)

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                #print(int(frameNumber))
                # Run YOLOv8 tracking on the frame, NOT persisting tracks between frames
                results = model(frame)

                # Extract bounding boxes, classes, names, and confidences
                boxes = results[0].boxes.xyxy.tolist()
                classes = results[0].boxes.cls.tolist()
                names = results[0].names
                confidences = results[0].boxes.conf.tolist()
                # print(names)

                # remove worst ball detections if more than one ball detection
                if classes.count(0.0) > 1:
                    # print("MAS DE UNA")
                    vals = [(n, x) for n, (i, x) in enumerate(zip(classes, confidences)) if i == 0.0]
                    del vals[vals.index(max(vals, key=lambda item: item[1]))]

                    # Reverse the list so that we ensure that indices are always accesible
                    vals.sort(reverse=True)
                    for i in range(len(vals)):
                        del classes[vals[i][0]]
                        del confidences[vals[i][0]]
                        del boxes[vals[i][0]]

                ball = False

                # Iterate through the results
                for box, cls, conf in zip(boxes, classes, confidences):
                    x1, y1, x2, y2 = box
                    confidence = round(Decimal(conf), 3)
                    detected_class = cls
                    name = names[int(cls)]

                    if name == 'Ball':
                        # YOLO is quite sure that the detected ball is, in fact, a ball
                        yoloConfidence = confidence
                        yoloBallPos = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        yoloBox = box
                        #color = (255, 255, 255)
                        nam = "Pelota"

                if previousBallPos is not None:

                    if len(xPos) == 5:
                        sd = (np.std(xPos) + np.std(yPos)) / 2
                        if sd == 0:
                            sd = 1

                    p = scipy.stats.norm((ballPos[0], ballPos[1]), sd).pdf((yoloBallPos[0], yoloBallPos[1]))
                    pt = p[0] + p[1]
                    #pt = round(Decimal(pt), 3)

                    velocity = (ballPos[0] - previousBallPos[0], ballPos[1] - previousBallPos[1])
                    gaussPred = (ballPos[0] + velocity[0], ballPos[1] + velocity[1])
                    previousBallPos = ballPos
                    pt = float(yoloConfidence) * pt * param1
                    probs = [yoloConfidence, pt, param2]
                    #print(probs)
                    idx = probs.index(max(probs))
                    if idx == 2:
                        #print("velocity prediction")
                        ballPos = (gaussPred[0], gaussPred[1])

                        if groundTruth[frameNumber] is not None:
                            if (np.abs(ballPos[0] - groundTruth[frameNumber][0]) <= metrics_threshold) and (
                                    np.abs(ballPos[1] - groundTruth[frameNumber][1]) <= metrics_threshold):
                                #print("GOOD DETECTION")
                                frameMetrics.append("g")
                                color = (0, 255, 0)
                            else:
                                #print("BAD DETECTION")
                                frameMetrics.append("b")
                                color = (0, 0, 255)
                        else:
                            #print("BAD DETECTION")
                            frameMetrics.append("b")
                            color = (0, 0, 255)

                        #out = cv2.circle(frame, (gaussPred[0], gaussPred[1]), 10, color, 4)
                    else:
                        #print("yolo or yolo gaussian")
                        ballPos = yoloBallPos
                        if groundTruth[frameNumber] is not None:
                            if (np.abs(ballPos[0] - groundTruth[frameNumber][0]) <= metrics_threshold) and (
                                    np.abs(ballPos[1] - groundTruth[frameNumber][1]) <= metrics_threshold):
                                #print("GOOD DETECTION")
                                frameMetrics.append("g")
                                color = (0, 255, 0)
                            else:
                                #print("BAD DETECTION")
                                frameMetrics.append("b")
                                color = (0, 0, 255)
                        else:
                            #print("BAD DETECTION")
                            frameMetrics.append("b")
                            color = (0, 0, 255)

                        #out = cv2.circle(frame, (ballPos[0], ballPos[1]), 10, color, 4)

                    update_XYpos(xPos, yPos, ballPos[0], ballPos[1])


                    #print("probabilidad yolo = " + str(yoloConfidence))
                    #print("probabilidad gaussiana = " + str(pt))

                    yoloConfidence = 0

                elif yoloBallPos is not None:  #esto antes era un else, comprobar funcionamiento (cambio de else a elif por PSGvsBVB, en el primer frame no hay pelota)
                    previousBallPos = ballPos

                    if len(xPos) == 5:
                        sd = (np.std(xPos) + np.std(yPos)) / 2
                        if sd == 0:
                            sd = 1

                    #print("yolo or yolo gaussian")
                    ballPos = yoloBallPos

                    if groundTruth[frameNumber] is not None:
                        if (np.abs(ballPos[0] - groundTruth[frameNumber][0]) <= metrics_threshold) and (
                                np.abs(ballPos[1] - groundTruth[frameNumber][1]) <= metrics_threshold):
                            #print("GOOD DETECTION")
                            frameMetrics.append("g")
                            color = (0, 255, 0)
                        else:
                            #print("BAD DETECTION")
                            frameMetrics.append("b")
                            color = (0, 0, 255)
                    else:
                        #print("BAD DETECTION")
                        frameMetrics.append("b")
                        color = (0, 0, 255)

                    #out = cv2.circle(frame, (ballPos[0], ballPos[1]), 10, color, 4)


                    update_XYpos(xPos, yPos, ballPos[0], ballPos[1])
                    #print("probabilidad yolo = " + str(yoloConfidence))
                    yoloConfidence = 0


                # Display the annotated frame
                #cv2.imshow("YOLOv8 Tracking", out)
                #cv2.waitKey(0)
                '''# waiting using waitKey method
                if 220 <= frameNumber <= 232:
                    cv2.waitKey(0)'''

                frameNumber += 1
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        # outVideo.release()
        cv2.destroyAllWindows()



        data.append([param1,param2, frameMetrics.count("g"), frameMetrics.count("b")])
        num_iter += 1
    # Creates DataFrame.
    df = pd.DataFrame(data)
    # saving the dataframe
    name = clip_name[1:-4] + '_newGaussianParametersACOTADOS' + '.csv'
    df.to_csv(name)