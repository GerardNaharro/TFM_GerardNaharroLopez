# TFM: FOOTBALL TRACKING AND ANALYTICS
import colorsys
import os
import cv2
import pandas as pd
from decimal import Decimal
from roboflow import Roboflow
import numpy as np
from ultralytics import YOLO
import scipy.stats

metrics = False
possession_threshold = 40
possessions = {}
passes = {}
missed_passes = {}
field_color1 = (34,12,30)
field_color2 = (53,218,217)


def hsv2rgb(h,s,v):
    rgb = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))
    return tuple([rgb[2], rgb[1], rgb[0]])

def get_abbr(clipname):
    subs = 'vs'
    ind = clipname.index(subs)
    return clipname[1:ind],clipname[ind + 2:-4]


def get_names(df, abbr1,abbr2):
    name1 = df['Team'][df['Abbr'] == abbr1].item()
    name2 = df['Team'][df['Abbr'] == abbr2].item()
    return name1, name2


def load_masks(df, team1, team2):

    team1_home_hsv = df['Home'][df['Team'] == team1].item()
    team1_away_hsv = df['Away'][df['Team'] == team1].item()

    team2_home_hsv = df['Home'][df['Team'] == team2].item()
    team2_away_hsv = df['Away'][df['Team'] == team2].item()

    team1_gk_home_hsv = df['GK_Home'][df['Team'] == team1].item()
    team1_gk_away_hsv = df['GK_Away'][df['Team'] == team1].item()

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
            team = team1
            if results.index(max_value) % 2 == 0:
                #colorhsv = [round((x + y)/2) for x, y in zip(team1_home_hsv[0], team1_home_hsv[1])]
                #color = colorsys.hsv_to_rgb(team1_home_hsv[0][0]/255,team1_home_hsv[0][1]/255,team1_home_hsv[0][2]/255)
                color = hsv2rgb(((team1_home_hsv[1][0] + team1_home_hsv[0][0]) / 2) / 180, team1_home_hsv[1][1] / 255,
                                            team1_home_hsv[1][2] / 255)
            else:
                #colorhsv = [round((x + y) / 2) for x, y in zip(team1_away_hsv[0], team1_away_hsv[1])]
                color = hsv2rgb(( (team1_away_hsv[1][0] + team1_away_hsv[0][0]) / 2) /180,team1_away_hsv[1][1]/255,team1_away_hsv[1][2]/255)

        else:
            team = team2
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
    ind = None
    for i in players:
        left_foot_distance = np.linalg.norm(tuple(x - y for x, y in zip(ball, i[0])))
        right_foot_distance = np.linalg.norm(tuple(x - y for x, y in zip(ball, i[1])))
        if minim is None:
            minim = min(left_foot_distance, right_foot_distance)
            z = i[2]
            ind = i
        elif min(left_foot_distance, right_foot_distance) < minim:
            minim = min(left_foot_distance, right_foot_distance)
            z = i[2]
            ind = i

    #print(str(minim))
    if minim is not None and ((ind[0][0] < ball[0] < ind[1][0] and (ind[0][1] - 50) < ball[1] < (ind[0][1] + 50)) or (minim <= possession_threshold)):
        return z, minim
    else:
        return None, None


def update_XYpos(xPos, yPos, xelem, yelem):
    xPos.append(xelem)
    yPos.append(yelem)

    if len(xPos) > 5:
        xPos.pop(0)
        yPos.pop(0)

def line_filtering(frame):
    filtered = cv2.bitwise_and(frame, frame, mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), field_color1, field_color2))
    #cv2.imshow("line filtering", filtered)
    edges = cv2.Canny(filtered, 25, 150)
    #cv2.imshow("yo soy cany cany cany cany", edges)
    kernel = np.ones((3, 3), np.uint8)
    kernel1 = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(edges, kernel, iterations=1)
    #v2.imshow('dilatasao', img_dilation)
    #img_dilation = cv2.erode(img_dilation, kernel1, iterations=1)
    #cv2.imshow('erosao', img_dilation)
    minLineLength = 250
    maxLineGap = 50
    frame2 = frame.copy()
    lines = cv2.HoughLinesP(img_dilation, rho= (2* np.pi) / 3,theta= (2* np.pi) / 3, threshold=10, minLineLength=minLineLength, maxLineGap=maxLineGap)
    supp_line = [0,0,0,0]
    if lines is not None:
        for line in lines:
            #color = list(np.random.choice(range(256), size=3))
            #color = (int(color[0]), int(color[1]), int(color[2]))
            x1, y1, x2, y2 = line[0]
            dist = np.abs(y1 - y2)
            #cv2.line(frame2, (x1, y1), (x2, y2), color, 2)
            if dist > np.abs(supp_line[1] - supp_line[3]):
                supp_line = line[0]
        print(supp_line)
        x1, y1, x2, y2 = supp_line
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 4)
        cv2.line(frame, (x2, y2), (x2, 0), (255, 0, 255), 4)
        cv2.line(frame, (x1, y1), (x1, 1080), (255, 0, 255), 4)
        #cv2.imshow("soy un palo.", frame2)

if __name__ == '__main__':
    df = pd.read_csv("hsv_teams.csv", header=0, sep = ';')
    #print(list(df.columns))
    HOME = os.getcwd()
    #print(HOME)
    clips = "D:\clips_futbol\listos"


    own_trained_location = HOME + "/runs\detect\Segunda_prueba_larga_nuevoEntreno_S_autobatch\weights/best.pt"
    # Load the own trained model
    model = YOLO(own_trained_location)

    clip_name = '/ASMvsMCY.mp4'


    # Define path to video file
    video_path = clips + clip_name

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    previousBallPos = None
    ballPos = None
    teamInPossession = None
    lastTeamInPossession = None
    VideoWidth = cap.get(3)  # float `width`
    VideoHeight = cap.get(4)  # float `height`


    #outVideo = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

    abbr1, abbr2 = get_abbr(clip_name)
    team1, team2 = get_names(df, abbr1, abbr2)

    team1_home_hsv, team1_away_hsv, team2_home_hsv, team2_away_hsv, team1_gk_home_hsv, team1_gk_away_hsv, team2_gk_home_hsv, team2_gk_away_hsv = load_masks(df, team1, team2)
    possessions[team1] = 0
    possessions[team2] = 0
    passes[team1] = 0
    passes[team2] = 0
    missed_passes[team1] = 0
    missed_passes[team2] = 0

    sd = 5
    xPos = []
    yPos = []

    yoloConfidence = 0
    yoloBox = None

    if metrics:
        frameMetrics = []

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame, NOT persisting tracks between frames
            line_filtering(frame)
            results = model(frame)



            # Extract bounding boxes, classes, names, and confidences
            boxes = results[0].boxes.xyxy.tolist()
            classes = results[0].boxes.cls.tolist()
            names = results[0].names
            confidences = results[0].boxes.conf.tolist()
            #print(names)

            #remove worst ball detections if more than one ball detection
            if classes.count(0.0) > 1:
                #print("MAS DE UNA")
                vals = [(n, x) for n, (i, x) in enumerate(zip(classes, confidences)) if i == 0.0]
                del vals[vals.index(max(vals, key=lambda item: item[1]))]

                # Reverse the list so that we ensure that indices are always accesible
                vals.sort(reverse = True)
                for i in range(len(vals)):
                    del classes[vals[i][0]]
                    del confidences[vals[i][0]]
                    del boxes[vals[i][0]]




            ball = False
            players = []

            # Iterate through the results
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box
                confidence = round(Decimal(conf),3)
                detected_class = cls
                name = names[int(cls)]

                if name == 'Ball':
                    yoloConfidence = confidence
                    yoloBallPos = (int((x1 + x2)/2), int((y1 + y2)/2))
                    yoloBox = box
                    color = (255, 255, 255)
                    nam = "Pelota"

                elif name == 'Referee':
                    color = (0, 0, 0)
                    nam = "Arbitro"
                else:
                    # crop image to get only the player
                    crop = frame[int(y1): int(y2), int(x1): int(x2)]
                    h, w, _ = crop.shape
                    cropped_player = crop[int(0.2*h): int(0.7*h), int(0.3*w): int(0.7*w)]

                    #cropped_player = frame[int(y1): int(y2), int(x1): int(x2)]
                    #cv2.imshow("Cropped", cropped_player)
                    #nam, color = get_team(cropped_player, (red_mask_lower, red_mask_upper, green_mask_lower, green_mask_upper))
                    nam, color = get_team_improved(cropped_player,
                                          team1_home_hsv, team1_away_hsv, team2_home_hsv, team2_away_hsv, team1_gk_home_hsv, team1_gk_away_hsv, team2_gk_home_hsv, team2_gk_away_hsv)

                    # Yolo, sometimes, makes wrong classifications, saying that a referee is a player
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
                pt = float(yoloConfidence) * pt * 22
                probs = [yoloConfidence, pt, 0.52]
                print(probs)
                idx = probs.index(max(probs))
                if idx == 2:
                    print("velocity prediction")
                    ballPos = (gaussPred[0], gaussPred[1])
                    out = cv2.circle(frame, (gaussPred[0], gaussPred[1]), 10, (255, 255, 255), 4)
                else:
                    print("yolo or yolo gaussian")
                    ballPos = yoloBallPos


                update_XYpos(xPos, yPos, ballPos[0], ballPos[1])


                #print("probabilidad yolo = " + str(yoloConfidence))
                #print("probabilidad gaussiana = " + str(pt))

                yoloConfidence = 0
            else:
                previousBallPos = ballPos

                if len(xPos) == 5:
                    sd = (np.std(xPos) + np.std(yPos)) / 2
                    if sd == 0:
                        sd = 1

                print("yolo or yolo gaussian")
                ballPos = yoloBallPos
                update_XYpos(xPos, yPos, ballPos[0], ballPos[1])
                print("probabilidad yolo = " + str(yoloConfidence))
                yoloConfidence = 0

            # Possession and passes calculations
            temp, dist = getPossessionTeam(players,ballPos)
            print("pusasio actual: " + str(temp))
            print("distansia actual: " + str(dist))

            if temp is None and teamInPossession is not None:
                lastTeamInPossession = teamInPossession
                print("ara mateix no hi ha pusasió")

            teamInPossession = temp
            if teamInPossession is not None:
                possessions[teamInPossession] += 1
                if lastTeamInPossession is not None and lastTeamInPossession == teamInPossession:
                    passes[teamInPossession] += 1
                    lastTeamInPossession = None
                    print("això ha estat un bon pase noi")
                elif lastTeamInPossession is not None and lastTeamInPossession != teamInPossession:
                    missed_passes[lastTeamInPossession] += 1
                    lastTeamInPossession = None
                    print("aquest no es va criar a la masia, mal pase noi")


            #out = cv2.putText(frame, "Equipo rojo" + ": " + str(100 * (possessions["Equipo rojo"] / (possessions["Equipo rojo"] + possessions["Equipo verde"]))) + "%", )
            if possessions[team1] != 0 or possessions[team2] != 0:
                out = cv2.putText(frame, team1 + " = " + str(100 * (possessions[team1] / (possessions[team1] + possessions[team2]))) + "%", (60, 100), 0, 1 / 2, [0, 0, 0],
                                  thickness=2,
                                  lineType=cv2.LINE_AA)
                out = cv2.putText(frame, team2 + " = " + str(100 * (possessions[team2] / (possessions[team1] + possessions[team2]))) + "%", (60, 115), 0, 1 / 2, [0, 0, 0],
                                  thickness=2,
                                  lineType=cv2.LINE_AA)
                out = cv2.putText(frame,"Actual = " + str(teamInPossession), (60, 130), 0, 1 / 2,
                                  [0, 0, 0],
                                  thickness=2,
                                  lineType=cv2.LINE_AA)
            #print("Pusesió")
            #if possessions[team1] != 0 or possessions[team2] != 0:
                #print(team1 + " = " + str(100 * (possessions[team1] / (possessions[team1] + possessions[team2]))))
                #print(team2 + " = " + str(100 * (possessions[team2] / (possessions[team1] + possessions[team2]))))
            print("------------------------------------------------------")
            print("ADN barça:")
            print(team1 + " pases = " + str(passes[team1]))
            print(team1 + " pases fallados = " + str(missed_passes[team1]))
            print(team2 + " pases = " + str(passes[team2]))
            print(team2 + " pases fallados = " + str(missed_passes[team2]))
            # Display the annotated frame
            cv2.imshow("YOLOv8 detection", out)


            #outVideo.write(out)

            if not metrics:
                # waiting using waitKey method
                cv2.waitKey(0)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("z"):
                    print("YOLO GOOD DETECTION")
                    frameMetrics.append("z")
                elif key == ord("x"):
                    print("YOLO BAD DETECTION")
                    frameMetrics.append("x")
                elif key == ord("c"):
                    print("KALMAN GOOD DETECTION")
                    frameMetrics.append("c")
                elif key == ord("v"):
                    print("KALMAN BAD DETECTION")
                    frameMetrics.append("v")
                elif key == ord("b"):
                    print("NO DETECTION")
                    frameMetrics.append("b")

            #print("PREVIOUS: " + str(previousBallPos))
            #print("ACTUAL: " + str(ballPos))


        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    #outVideo.release()
    cv2.destroyAllWindows()

    if metrics:
        data = [['CLIP NAME', clip_name], ['YOLO GOOD DETECTION', frameMetrics.count("z")], ['YOLO BAD DETECTION', frameMetrics.count("x")], ['KALMAN GOOD DETECTION', frameMetrics.count("c")],
                ['KALMAN BAD DETECTION', frameMetrics.count("v")], ['NO DETECTION', frameMetrics.count("b")]]
        # Creates DataFrame.
        df = pd.DataFrame(data)
        # saving the dataframe
        name = clip_name[1:-4] + '.csv'
        df.to_csv(name)