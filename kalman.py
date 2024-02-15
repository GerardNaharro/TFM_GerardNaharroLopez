# TFM: FOOTBALL TRACKING AND ANALYTICS

import os
import cv2
from decimal import Decimal
from kalmanfilter import KalmanFilter
from roboflow import Roboflow
import numpy as np
from ultralytics import YOLO



'''# Class KalmanFilter for the easy usage of the kalman filter implemented in OpenCv
class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)  # 4 state variables and 2 measurements variables
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)  # only positions are directly observable
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]], np.float32) * 0.02
    # The transition matrix defines how our state vectors evolve from time step t to t+1
    # based on a simple linear motion model where objects move linearly at constant velocity.
    # The first two rows map position estimates onto future position estimates
    # (position must advance by current speed values), while last two rows mantain unchanged predictions
    # about velocities.

    # Predict function to make the prediction of the next position of the object
    def predict(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        predicted = self.kf.predict()  # Computes a predicted state.
        self.kf.correct(measured)  # Updates the predicted state from the measurement.
        x, y = int(predicted[0]), int(predicted[1])
        return x, y'''

# !!!    MEJORAR FUNCION     !!!

def get_team(img, masks):

    lower_team1, upper_team1, lower_team2, upper_team2 = masks

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    team1_mask = cv2.inRange(hsv, lower_team1, upper_team1)
    team2_mask = cv2.inRange(hsv, lower_team2, upper_team2)

    out_team1 = cv2.bitwise_and(img, img, mask=team1_mask)
    out_team2 = cv2.bitwise_and(img, img, mask=team2_mask)

    count1 = np.sum(out_team1 != 0)
    count2 = np.sum(out_team2 != 0)

    if count1 < count2:
        color = (0,255,0)
        team = "Verdes"
    else:
        color = (0,0,255)
        team = "Rojos"

    return team, color

if __name__ == '__main__':
    HOME = os.getcwd()
    #print(HOME)
    clips = "D:\clips_futbol"


    own_trained_location = HOME + "/runs\detect\Segunda_prueba_larga_nuevoEntreno_S_autobatch\weights/best.pt"
    # Load the own trained model
    model = YOLO(own_trained_location)


    # Define path to video file
    video_path = clips + '\clip4.mp4'

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    kf = KalmanFilter()
    pred = None

    #   TO IMPROVE
    red_mask_lower = (0, 77, 75)
    red_mask_upper = (22, 255, 255)

    green_mask_lower = (38, 56, 155)
    green_mask_upper = (91, 204, 255)
    #   TO IMPROVE


    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, NOT persisting tracks between frames
            results = model(frame)



            # Extract bounding boxes, classes, names, and confidences
            boxes = results[0].boxes.xyxy.tolist()
            classes = results[0].boxes.cls.tolist()
            names = results[0].names
            confidences = results[0].boxes.conf.tolist()
            print(names)

            ball = False
            # Iterate through the results
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box
                confidence = round(Decimal(conf),3)
                detected_class = cls
                name = names[int(cls)]

                if name == 'Ball':
                    print(x1 , " " , y1)
                    print(x2 , " " , y2)
                    print(confidence)
                    ball = True
                    # Predict the position of the ball with kalman filter
                    pred = kf.predict((x1+x2)/2, (y1+y2)/2)
                    color = (255, 255, 255)
                    nam = "Pelota"
                elif name == 'Referee':
                    color = (0, 0, 0)
                    nam = "Arbitro"
                else:
                    # !!!   FILTRADO DE COLOR   !!!
                    # crop image to get only the player
                    cropped_player = frame[int(y1): int(y2), int(x1): int(x2)]
                    cv2.imshow("Cropped", cropped_player)
                    nam, color = get_team(cropped_player, (red_mask_lower, red_mask_upper, green_mask_lower, green_mask_upper))
                    #cv2.waitKey(0)

                # Visualize the results on the frame
                #annotated_frame = results[0].plot()
                out = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                t_size = cv2.getTextSize(str(confidence), 0, fontScale=1 / 2, thickness=1)[0]
                out = cv2.rectangle(frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3),
                              color, -1)
                out = cv2.putText(frame, nam + ": " + str(confidence), (int(x1), int(y1) - 4), 0, 1 / 2, [0, 0, 0], thickness=1,
                            lineType=cv2.LINE_AA)



            if not ball:
                if pred is not None:
                    # Visualize the results on the frame
                    out = cv2.circle(frame, (pred[0],pred[1]), 10, (255,255,255), 4)

                    pred = kf.predict(pred[0], pred[1])


            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", out)


            # waiting using waitKey method
            #cv2.waitKey(0)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
