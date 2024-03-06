# Program To Read video
# and Extract Frames

import cv2
import os

# Function to extract frames
def FrameCapture(path):

	# Path to video file
	vidObj = cv2.VideoCapture(path)

	# Used as counter variable
	count = 0

	# checks whether frames were extracted
	success = 1

	while success:

    	# vidObj object calls read
		# function extract frames
		success, image = vidObj.read()

		# Saves the frames with frame-count
		cv2.imwrite("frames/"+clip_name[1:-4]+"/frame%d.jpg" % count, image)

		count += 1


# Driver Code
if __name__ == '__main__':

    HOME = os.getcwd()
    # print(HOME)
    clips = "D:\clips_futbol\listos"
    clip_name = '/PSGvsRNS.mp4'

    # Define path to video file
    video_path = clips + clip_name

    # Calling the function
    FrameCapture(video_path)
