import cv2
import os
import json

# function to display the coordinates
# of the points clicked on the image
def click_event(event, x, y,flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        groundTruth.append((x,y))
        params[0] = True

        # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print("NONE")
        groundTruth.append(None)
        params[0] = True



if __name__ == "__main__":
    groundTruth = []
    HOME = os.getcwd()
    clips = "D:\clips_futbol\listos"
    clip_name = '/BMUvsBLV.mp4'
    # Define path to video file
    video_path = clips + clip_name

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    clicked = [False]

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            cv2.imshow("FRAME", frame)
            # setting mouse handler for the image
            # and calling the click_event() function
            cv2.setMouseCallback('FRAME', click_event, clicked)
            while not clicked[0]:
                cv2.waitKey(1)
            clicked = [False]
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    # outVideo.release()
    cv2.destroyAllWindows()

    with open(clip_name[1:-4]+".json", 'w') as f:
        json.dump(groundTruth, f, indent=2)
