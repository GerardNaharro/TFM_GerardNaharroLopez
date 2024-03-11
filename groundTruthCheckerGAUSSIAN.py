import os
import json
import cv2
from ultralytics import YOLO
from kalmanfilter import KalmanFilter
import scipy.stats
from decimal import Decimal
import pandas as pd
import numpy as np


metrics_threshold = 10

if __name__ == '__main__':
    HOME = os.getcwd()
    clips = "D:\clips_futbol\listos"

    own_trained_location = HOME + "/runs\detect\Segunda_prueba_larga_nuevoEntreno_S_autobatch\weights/best.pt"
    # Load the own trained model
    model = YOLO(own_trained_location)

    clip_name = '/BVBvsPSG.mp4'

    # Define path to video file
    video_path = clips + clip_name

    with open(HOME + "/newMetrics/" + clip_name[1:-4] + ".json", 'r') as f:
        groundTruth = json.load(f)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    previousBallPos = None
    ballPos = ()
    VideoWidth = cap.get(3)  # float `width`
    VideoHeight = cap.get(4)  # float `height`
    frameMetrics = []
    frameNumber = 0

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            print(int(frameNumber))
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
                    if confidence >= 0.52 or previousBallPos is None:
                        ball = True
                        if len(ballPos) != 0:
                            previousBallPos = ballPos
                        else:
                            previousBallPos = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                        ballPos = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        if groundTruth[frameNumber] is not None:
                            if (np.abs(ballPos[0] - groundTruth[frameNumber][0]) <= metrics_threshold) and (np.abs(ballPos[1] - groundTruth[frameNumber][1]) <= metrics_threshold):
                                print("YOLO GOOD DETECTION")
                                frameMetrics.append("z")
                                color = (0, 255, 0)
                            else:
                                print("YOLO BAD DETECTION")
                                frameMetrics.append("x")
                                color = (0, 0, 255)
                        else:
                            print("YOLO BAD DETECTION")
                            frameMetrics.append("x")
                            color = (0, 0, 255)

                        nam = "Pelota"
                    # The detected ball could not really be a ball, so we apply a normal distribution
                    else:
                        # print(ballPos)
                        p = scipy.stats.norm((ballPos[0], ballPos[1]), 20).pdf(((x1 + x2) / 2, (y1 + y2) / 2))
                        pt = p[0] + p[1]
                        print(confidence)
                        print(pt)
                        if pt >= 0.032:
                            ball = True
                            previousBallPos = ballPos
                            ballPos = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            if groundTruth[frameNumber] is not None:
                                if (np.abs(ballPos[0] - groundTruth[frameNumber][0]) <= metrics_threshold) and (
                                        np.abs(ballPos[1] - groundTruth[frameNumber][1]) <= metrics_threshold):
                                    print("YOLO GOOD DETECTION")
                                    frameMetrics.append("z")
                                    color = (0, 255, 0)
                                else:
                                    print("YOLO BAD DETECTION")
                                    frameMetrics.append("x")
                                    color = (0, 0, 255)
                            else:
                                print("YOLO BAD DETECTION")
                                frameMetrics.append("x")
                                color = (0, 0, 255)
                            nam = "Pelota"
                        else:
                            continue

                if ball and name == "Ball":
                    # Visualize the results on the frame
                    # annotated_frame = results[0].plot()
                    out = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                    t_size = cv2.getTextSize(str(confidence), 0, fontScale=1 / 2, thickness=1)[0]
                    out = cv2.rectangle(frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3),
                                        color, -1)
                    out = cv2.putText(frame, nam + ": " + str(confidence), (int(x1), int(y1) - 4), 0, 1 / 2, [0, 0, 0],
                                      thickness=1,
                                      lineType=cv2.LINE_AA)


            if not ball and previousBallPos is not None:
                velocity = (ballPos[0] - previousBallPos[0], ballPos[1] - previousBallPos[1])
                gaussPred = (ballPos[0] + velocity[0], ballPos[1] + velocity[1])

                p = scipy.stats.norm((ballPos[0], ballPos[1]), 20).pdf((gaussPred[0], gaussPred[1]))
                pt = p[0] + p[1]
                print(pt)
                if pt >= 0.032:
                    previousBallPos = ballPos
                    ballPos = (gaussPred[0], gaussPred[1])

                    if groundTruth[frameNumber] is not None:
                        if (np.abs(ballPos[0] - groundTruth[frameNumber][0]) <= metrics_threshold) and (
                                np.abs(ballPos[1] - groundTruth[frameNumber][1]) <= metrics_threshold):
                            print("GAUSSIAN GOOD DETECTION")
                            frameMetrics.append("c")
                            color = (0, 255, 0)
                        else:
                            print("GAUSSIAN BAD DETECTION")
                            frameMetrics.append("v")
                            color = (0, 0, 255)
                    else:
                        print("GAUSSIAN BAD DETECTION")
                        frameMetrics.append("v")
                        color = (0, 0, 255)

                    out = cv2.circle(frame, (gaussPred[0], gaussPred[1]), 10, color, 4)


                else:
                    if groundTruth[frameNumber] is not None:
                        print("BAD NO DETECTION")
                        frameMetrics.append("b")
                        out = frame
                    else:
                        print("GOOD NO DETECTION")
                        frameMetrics.append("n")
                        out = frame

            if not ball and previousBallPos is None:
                if groundTruth[frameNumber] is not None:
                    print("BAD NO DETECTION")
                    frameMetrics.append("b")
                    out = frame
                else:
                    print("GOOD NO DETECTION")
                    frameMetrics.append("n")
                    out = frame

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", out)

            # waiting using waitKey method
            if 220 <= frameNumber <= 232:
                cv2.waitKey(0)
            frameNumber += 1
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    # outVideo.release()
    cv2.destroyAllWindows()



    data = [['CLIP NAME', clip_name], ['YOLO GOOD DETECTION', frameMetrics.count("z")],
                ['YOLO BAD DETECTION', frameMetrics.count("x")], ['GAUSSIAN GOOD DETECTION', frameMetrics.count("c")],
                ['GAUSSIAN BAD DETECTION', frameMetrics.count("v")], ['BAD NO DETECTION', frameMetrics.count("b")], ['GOOD NO DETECTION', frameMetrics.count("n")]]
    # Creates DataFrame.
    df = pd.DataFrame(data)
    # saving the dataframe
    name = clip_name[1:-4] + '_newGAUSSIAN' + '.csv'
    df.to_csv(name)