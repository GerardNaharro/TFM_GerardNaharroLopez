# TFM: FOOTBALL TRACKING AND ANALYTICS
import colorsys
import os
import cv2
import pandas as pd
from decimal import Decimal
from kalmanfilter import KalmanFilter
from roboflow import Roboflow
import numpy as np
from ultralytics import YOLO


possession_threshold = 30
possessions = {}
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


def hsv2rgb(h,s,v):
    rgb = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))
    return tuple([rgb[2], rgb[1], rgb[0]])

def load_masks(df, team1, team2):

    team1_home_hsv = df['Home'][df['Team']==team1].item()
    team1_away_hsv = df['Away'][df['Team']==team1].item()

    team2_home_hsv = df['Home'][df['Team'] == team2].item()
    team2_away_hsv = df['Away'][df['Team'] == team2].item()

    team1_gk_home_hsv = df['GK_Home'][df['Team']==team1].item()
    team1_gk_away_hsv = df['GK_Away'][df['Team']==team1].item()

    team2_gk_home_hsv = df['GK_Home'][df['Team'] == team2].item()
    team2_gk_away_hsv = df['GK_Away'][df['Team'] == team2].item()

    # eval converts from string to tuple
    return eval(team1_home_hsv), eval(team1_away_hsv), eval(team2_home_hsv), eval(team2_away_hsv), eval(team1_gk_home_hsv), eval(team1_gk_away_hsv), eval(team2_gk_home_hsv), eval(team2_gk_away_hsv)

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

def get_team_improved(img, team1_home_hsv,team1_away_hsv, team2_home_hsv, team2_away_hsv, team1_gk_home_hsv, team1_gk_away_hsv, team2_gk_home_hsv, team2_gk_away_hsv):


    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    team1_mask_home = cv2.inRange(hsv, team1_home_hsv[0], team1_home_hsv[1])
    team1_mask_away = cv2.inRange(hsv, team1_away_hsv[0], team1_away_hsv[1])

    team1_gk_mask_home = cv2.inRange(hsv, team1_gk_home_hsv[0], team1_gk_home_hsv[1])
    team1_gk_mask_away = cv2.inRange(hsv, team1_gk_away_hsv[0], team1_gk_away_hsv[1])

    team2_mask_home = cv2.inRange(hsv, team2_home_hsv[0], team2_home_hsv[1])
    team2_mask_away = cv2.inRange(hsv, team2_away_hsv[0], team2_away_hsv[1])

    team2_gk_mask_home = cv2.inRange(hsv, team2_gk_home_hsv[0], team2_gk_home_hsv[1])
    team2_gk_mask_away = cv2.inRange(hsv, team2_gk_away_hsv[0], team2_gk_away_hsv[1])

    out_team1_home = cv2.bitwise_and(img, img, mask=team1_mask_home)
    out_team1_away = cv2.bitwise_and(img, img, mask=team1_mask_away)

    out_team1_gk_home = cv2.bitwise_and(img, img, mask=team1_gk_mask_home)
    out_team1_gk_away = cv2.bitwise_and(img, img, mask=team1_gk_mask_away)

    out_team2_home = cv2.bitwise_and(img, img, mask=team2_mask_home)
    out_team2_away = cv2.bitwise_and(img, img, mask=team2_mask_away)

    out_team2_gk_home = cv2.bitwise_and(img, img, mask=team2_gk_mask_home)
    out_team2_gk_away = cv2.bitwise_and(img, img, mask=team2_gk_mask_away)

    results = []
    results.append(np.sum(out_team1_home != 0))
    results.append(np.sum(out_team1_away != 0))
    results.append(np.sum(out_team1_gk_home != 0))
    results.append(np.sum(out_team1_gk_away != 0))

    results.append(np.sum(out_team2_home != 0))
    results.append(np.sum(out_team2_away != 0))
    results.append(np.sum(out_team2_gk_home != 0))
    results.append(np.sum(out_team2_gk_away != 0))

    max_value = max(results)
    #print(results)

    referee_threshold = 150

    if max_value > referee_threshold:
        if results.index(max_value) <= 3:
            team = "Equipo Rojo"
            if results.index(max_value) % 2 == 0:
                #colorhsv = [round((x + y)/2) for x, y in zip(team1_home_hsv[0], team1_home_hsv[1])]
                #color = colorsys.hsv_to_rgb(team1_home_hsv[0][0]/255,team1_home_hsv[0][1]/255,team1_home_hsv[0][2]/255)
                color = hsv2rgb(((team1_home_hsv[1][0] + team1_home_hsv[0][0]) / 2) / 180, team1_home_hsv[1][1] / 255,
                                            team1_home_hsv[1][2] / 255)
            else:
                #colorhsv = [round((x + y) / 2) for x, y in zip(team1_away_hsv[0], team1_away_hsv[1])]
                color = hsv2rgb(( (team1_away_hsv[1][0] + team1_away_hsv[0][0]) / 2) /180,team1_away_hsv[1][1]/255,team1_away_hsv[1][2]/255)

        else:
            team = "Equipo Verde"
            if results.index(max_value) % 2 == 0:
                #colorhsv = [round((x + y)/2) for x, y in zip(team2_home_hsv[0], team2_home_hsv[1])]
                color = hsv2rgb(((team2_home_hsv[1][0] + team2_home_hsv[0][0])/2)/180,team2_home_hsv[1][1]/255,team2_home_hsv[1][2]/255)
            else:
                #colorhsv = [round((x + y) / 2) for x, y in zip(team2_away_hsv[0], team2_away_hsv[1])]
                color = hsv2rgb(((team2_away_hsv[1][0] + team2_away_hsv[0][0]) /2)/180,team2_away_hsv[1][1]/255,team2_away_hsv[1][2]/255)
    else:
        team = "Arbitro"
        color = (0,0,0)

    return team, color


def getPossessionTeam(players, ball):
    minim = None
    for i in players:
        left_foot_distance = np.linalg.norm(tuple(x - y for x, y in zip(ball, i[0])))
        right_foot_distance = np.linalg.norm(tuple(x - y for x, y in zip(ball, i[1])))
        if minim is None:
            minim = min(left_foot_distance, right_foot_distance)
            z = i[2]
        elif min(left_foot_distance, right_foot_distance) < minim:
            minim = min(left_foot_distance, right_foot_distance)
            z = i[2]


    if minim <= possession_threshold:
        return z
    else:
        return None


if __name__ == '__main__':
    df = pd.read_csv("hsv_teams.csv", header=0, sep = ';')
    #print(list(df.columns))
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
    used = False

    VideoWidth = cap.get(3)  # float `width`
    VideoHeight = cap.get(4)  # float `height`

    #   TO IMPROVE
    red_mask_lower = (0, 77, 75)
    red_mask_upper = (22, 255, 255)

    green_mask_lower = (38, 56, 155)
    green_mask_upper = (91, 204, 255)
    #   TO IMPROVE

    team1_home_hsv, team1_away_hsv, team2_home_hsv, team2_away_hsv, team1_gk_home_hsv, team1_gk_away_hsv, team2_gk_home_hsv, team2_gk_away_hsv = load_masks(df, "Equipo rojo", "Equipo verde")
    possessions["Equipo Rojo"] = 0
    possessions["Equipo Verde"] = 0

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
            #print(names)

            #remove worst ball detections if more than one ball detection
            if classes.count(0.0) > 1:
                print("MAS DE UNA")
                vals = [(n, x) for n, (i, x) in enumerate(zip(classes, confidences)) if i == 0.0]
                del vals[vals.index(max(vals, key=lambda item: item[1]))]
                for i in range(len(vals)):
                    del classes[vals[i][0]]
                    del confidences[vals[i][0]]
                    del boxes[vals[i][0]]




            ball = False
            players = []
            ballPos = ()
            # Iterate through the results
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box
                confidence = round(Decimal(conf),3)
                detected_class = cls
                name = names[int(cls)]

                if name == 'Ball':
                    #print(x1 , " " , y1)
                    #print(x2 , " " , y2)
                    #print(confidence)
                    ball = True
                    # Predict the position of the ball with kalman filter
                    if used:
                        kf = KalmanFilter()
                        print("reset")
                        used = False

                    pred = kf.predict((x1+x2)/2, (y1+y2)/2)
                    ballPos = ((x1 + x2)/2, (y1 + y2)/2)
                    color = (255, 255, 255)
                    nam = "Pelota"
                elif name == 'Referee':
                    color = (0, 0, 0)
                    nam = "Arbitro"
                else:
                    # !!!   FILTRADO DE COLOR   !!!
                    # crop image to get only the player
                    cropped_player = frame[int(y1): int(y2), int(x1): int(x2)]
                    #cv2.imshow("Cropped", cropped_player)
                    #nam, color = get_team(cropped_player, (red_mask_lower, red_mask_upper, green_mask_lower, green_mask_upper))
                    nam, color = get_team_improved(cropped_player,
                                          team1_home_hsv, team1_away_hsv, team2_home_hsv, team2_away_hsv, team1_gk_home_hsv, team1_gk_away_hsv, team2_gk_home_hsv, team2_gk_away_hsv)

                    if nam != "Arbitro":
                        players.append(((x1,y2), (x2, y2), nam))

                    #cv2.waitKey(0)

                # Visualize the results on the frame
                #annotated_frame = results[0].plot()
                out = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                t_size = cv2.getTextSize(str(confidence), 0, fontScale=1 / 2, thickness=1)[0]
                out = cv2.rectangle(frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3),
                              color, -1)
                out = cv2.putText(frame, nam + ": " + str(confidence), (int(x1), int(y1) - 4), 0, 1 / 2, [0, 0, 0], thickness=1,
                            lineType=cv2.LINE_AA)
                #out = cv2.circle(frame, (int(x1), int(y2)), 5, (0, 0, 255), -1)
                #out = cv2.circle(frame, (int(x2), int(y2)), 5, (255, 255, 0), -1)




            if not ball:
                if pred is not None:
                    ballPos = (pred[0],pred[1])
                    out = cv2.circle(frame, (pred[0],pred[1]), 10, (255,255,255), 4)

                    pred = kf.predict(pred[0], pred[1])
                    used = True


            # Possession calculations
            teamInPossession = getPossessionTeam(players,ballPos)
            if teamInPossession is not None:
                possessions[teamInPossession] += 1


            #out = cv2.putText(frame, "Equipo rojo" + ": " + str(100 * (possessions["Equipo rojo"] / (possessions["Equipo rojo"] + possessions["Equipo verde"]))) + "%", )
            print("Pusesió")
            if possessions["Equipo Rojo"] != 0 or possessions["Equipo Verde"] != 0:
                print("Equipo Rojo = " + str(100 * (possessions["Equipo Rojo"] / (possessions["Equipo Rojo"] + possessions["Equipo Verde"]))))
                print("Equipo Verde = " + str(100 * (possessions["Equipo Verde"] / (possessions["Equipo Rojo"] + possessions["Equipo Verde"]))))
            print("------------------------------------------------------")
            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", out)


            # waiting using waitKey method
            cv2.waitKey(0)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
