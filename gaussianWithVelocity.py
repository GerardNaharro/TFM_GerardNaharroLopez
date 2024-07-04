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
#from elements.perspective_transform import Perspective_Transform
#from elements.assets import transform_matrix
from pathlib import Path
import scipy.io as sio
from PIL import Image
from scipy.ndimage import label
from skimage.measure import regionprops
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pyflann
from Perspective_Transformation.python_codes.util.synthetic_util import SyntheticUtil
from Perspective_Transformation.python_codes.util.iou_util import IouUtil
from Perspective_Transformation.python_codes.util.projective_camera import ProjectiveCamera
from Perspective_Transformation.python_codes.deep.camera_dataset import CameraDataset
from Perspective_Transformation.python_codes.options.test_options import TestOptions
from Perspective_Transformation.python_codes.models.models import create_model
from Perspective_Transformation.python_codes.deep.siamese import BranchNetwork, SiameseNetwork
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import threading


metrics = False
save = True
possession_threshold = 40
possessions = {}
passes = {}
missed_passes = {}
side_time = {}
field_color1 = (34,12,30)
field_color2 = (90,255,255)
middle_line = None
area_line = None
refined_h = None
retrieved_image = None
seg_map = None
edge_map = None
side = None
side_aux = None
zone = None
actual_homography_matrix = None
new_homography_matrix = None
discarded = 0


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

# Function to draw the full line going through two points
def draw_line_between_points(height, width, pt1, pt2, frame, color):

    # Calculate the slope and angle of the line between the two points.
    delta_y = pt2[1] - pt1[1]
    delta_x = pt2[0] - pt1[0]

    if delta_x == 0:
        # Avoid division by zero, handle case of a vertical line
        slope = np.inf
    else:
        slope = delta_y / delta_x

    angle = np.arctan(slope)

    # to calculate the coordinates of the points at the edges of the image
    # to extend the line
    x1 = int(pt1[0] - (pt1[1] - 0) / slope) if slope != 0 else pt1[0]
    y1 = 0
    x2 = int(pt2[0] + (height - pt2[1]) / slope) if slope != 0 else pt2[0]
    y2 = height - 1

    # draw
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness=4)

def line_filtering(frame, temp_frame):
    global middle_line
    found = False
    kernel = np.ones((3, 3), np.uint8)
    filtered = cv2.bitwise_and(temp_frame, temp_frame,
                               mask=cv2.inRange(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2HSV), field_color1,field_color2))
    edges = cv2.Canny(filtered, 25, 150)
    img_dilation = cv2.dilate(edges, kernel, iterations=1)
    minLineLength = 240
    maxLineGap = 20
    lines = cv2.HoughLinesP(img_dilation, rho=1, theta=np.pi / 180, threshold=10, minLineLength=minLineLength,
                            maxLineGap=maxLineGap)
    supp_line = [0, 0, 0, 0]
    if lines is not None:
        for line in lines:

            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            angle = np.math.atan(slope) * 180. / np.pi
            size = np.abs(y1 - y2)
            if 85 <= angle <= 95 or -85 >= angle >= -95:
                found = True
                if size > np.abs(supp_line[1] - supp_line[3]):
                    supp_line = line[0]
        if found:
            x1, y1, x2, y2 = supp_line
            middle_line = (x1, x2)
            #print(middle_line)

            #draw_line_between_points(1080, 1920, (x1, y1), (x2, y2), frame, (0, 255, 0))
            #cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)
        else:
            middle_line = None
    else:
        middle_line = None



def line_filtering_2(frame, temp_frame):
    global area_line
    found = False
    supp_angle = None
    kernel = np.ones((3, 3), np.uint8)
    filtered = cv2.bitwise_and(temp_frame, temp_frame, mask = cv2.inRange(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2HSV), field_color1, field_color2))
    edges = cv2.Canny(filtered, 25, 150)
    img_dilation = cv2.dilate(edges, kernel, iterations=1)
    minLineLength = 175
    maxLineGap = 30
    lines = cv2.HoughLinesP(img_dilation, rho= 1, theta= np.pi / 180, threshold=10, minLineLength=minLineLength, maxLineGap=maxLineGap)
    supp_line = [0,0,0,0]
    if lines is not None:
        for line in lines:
            '''color = list(np.random.choice(range(256), size=3))
            color = (int(color[0]), int(color[1]), int(color[2]))'''
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            angle = np.math.atan(slope) * 180. / np.pi
            size = np.abs(y1 - y2)
            if 20 <= angle <= 50 or -130 >= angle >= -160:
                found = True
                supp_angle = angle
                'cv2.line(temp_frame,(x1,y1),(x2,y2),color)'
                if size > np.abs(supp_line[1] - supp_line[3]):
                    supp_line = line[0]
                    side = 1
            elif 130 <= angle <= 160 or -20 >= angle >= -50:
                found = True
                supp_angle = angle
                'cv2.line(temp_frame, (x1, y1), (x2, y2), color)'
                if size > np.abs(supp_line[1] - supp_line[3]):
                    supp_line = line[0]
                    side = 0
        'cv2.imshow("palo areas", temp_frame)'
        if found:
            x1, y1, x2, y2 = supp_line
            print(str(supp_angle))
            area_line = ((x1, x2), side)
            draw_line_between_points(1080, 1920, (x1, y1), (x2, y2), frame, (255, 0, 255))
        else:
            area_line = None
    else:
        area_line = None

def get_pitch_side(ballPos):
    # If the middle line of the pitch is detected
    if middle_line is not None:
        if ballPos[0] < middle_line[0]:
            #print("ball is on left side of the pitch MIDDLE")
            return 0
        elif ballPos[0] > middle_line[1]:
            #print("ball is on right side of the pitch MIDDLE")
            return 1
        else:
            return None
    # We detect one of the areas
    elif area_line is not None:
        if area_line[1] == 0:
            print("ball is on left side of the pitch AREA")
        else:
            print("ball is on left side of the pitch AREA")
        return area_line[1]
    else:
        return None


def initialize_deep_feature(deep_model_directory):
    cuda_id = 0  # use -1 for CPU and 0 for GPU
    # 2: load network
    branch = BranchNetwork()
    net = SiameseNetwork(branch)

    if os.path.isfile(deep_model_directory):
        checkpoint = torch.load(deep_model_directory, map_location=lambda storage, loc: storage)
        net.load_state_dict(checkpoint['state_dict'])
        print('load model file from {}.'.format(deep_model_directory))
    else:
        print('Error: file not found at {}'.format(deep_model_directory))

        # 3: setup computation device
    device = 'cuda'
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(cuda_id))
        net = net.to(device)
        cudnn.benchmark = True
    print('computation device: {}'.format(device))

    normalize = transforms.Normalize(mean=[0.0188],
                                     std=[0.128])

    data_transform = transforms.Compose(
        [transforms.ToTensor(),
         normalize,
         ]
    )

    return net, data_transform, device

def initialize_two_GAN(directory):
    opt = TestOptions().parse(directory)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.continue_train = False

    model = create_model(opt)
    return model


def testing_two_GAN(image, model):
    # test

    if __name__ == '__main__':
        image = Image.fromarray(image)
        osize = [512, 256]

        cropsize = osize
        image = transforms.Compose(
            [transforms.Resize(osize, Image.BICUBIC), transforms.RandomCrop(cropsize), transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(image)
        image = image.unsqueeze(0)

        model.set_input(image)
        model.test()

        visuals = model.get_current_visuals()
        edge_map = visuals['fake_D']
        seg_map = visuals['fake_C']
        edge_map = cv2.resize(edge_map, (1280, 720), interpolation=5)
        seg_map = cv2.resize(seg_map, (1280, 720))

    return edge_map, seg_map


def generate_deep_feature(edge_map, net, data_transform, device):
    """
    Extract feature from a siamese network
    input: network and edge images
    output: feature and camera
    """
    # parameters
    batch_size = 1

    # resize image
    pivot_image = edge_map

    pivot_image = cv2.resize(pivot_image, (320, 180))

    pivot_image = cv2.cvtColor(pivot_image, cv2.COLOR_RGB2GRAY)

    pivot_images = np.reshape(pivot_image, (1, pivot_image.shape[0], pivot_image.shape[1]))

    print('Note: assume input image resolution is 180 x 320 (h x w)')

    data_loader = CameraDataset(pivot_images,
                                pivot_images,
                                batch_size,
                                -1,
                                data_transform,
                                is_train=False)

    features = []

    with torch.no_grad():
        for i in range(len(data_loader)):
            x, _ = data_loader[i]
            x = x.to(device)
            feat = net.feature_numpy(x)  # N x C
            features.append(feat)
            # append to the feature list

    features = np.vstack((features))
    return features, pivot_image

def generate_transform_matrix(frame2, twoGanModel, data_transform, device, current_directory, template_h, template_w, database_features):
    global  refined_h, retrieved_image, seg_map, edge_map, new_homography_matrix
    edge_map, seg_map = testing_two_GAN(frame2, twoGanModel)
    edge_map = eliminar_regiones_pequenas(edge_map, 7500)
    #edge_map = cv2.erode(edge_map, np.ones((3, 3), np.uint8), iterations=1)
    test_features, reduced_edge_map = generate_deep_feature(edge_map, net, data_transform, device)


    # World Cup soccer template
    data = sio.loadmat(current_directory + "/data_2/worldcup2014.mat")
    model_points = data['points']
    model_line_index = data['line_segment_index']

    # Retrieve a camera using deep features
    flann = pyflann.FLANN()
    result, _ = flann.nn(database_features, test_features[query_index], 1, algorithm="kdtree", trees=16,
                         checks=128)
    retrieved_index = result[0]

    # Retrieval camera: get the nearest-neighbor camera from database
    retrieved_camera_data = database_cameras[retrieved_index]

    u, v, fl = retrieved_camera_data[0:3]
    rod_rot = retrieved_camera_data[3:6]
    cc = retrieved_camera_data[6:9]

    retrieved_camera = ProjectiveCamera(fl, u, v, cc, rod_rot)

    retrieved_h = IouUtil.template_to_image_homography_uot(retrieved_camera, template_h, template_w)

    retrieved_image = SyntheticUtil.camera_to_edge_image(retrieved_camera_data, model_points, model_line_index,
                                                         im_h=720, im_w=1280, line_width=2)

    # Refine camera: refine camera pose using Lucas-Kanade algorithm
    dist_threshold = 50
    query_dist = SyntheticUtil.distance_transform(edge_map)
    retrieved_dist = SyntheticUtil.distance_transform(retrieved_image)

    query_dist[query_dist > dist_threshold] = dist_threshold
    retrieved_dist[retrieved_dist > dist_threshold] = dist_threshold

    h_retrieved_to_query = SyntheticUtil.find_transform(retrieved_dist, query_dist)

    refined_h = h_retrieved_to_query @ retrieved_h

    new_homography_matrix = np.linalg.inv(refined_h)




def transform_matrix(matrix, p, vid_shape, gt_shape):
    p = (p[0]*1280/vid_shape[1], p[1]*720/vid_shape[0])
    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))

    p_after = (int(px*gt_shape[1]/115) , int(py*gt_shape[0]/74))

    return p_after

def update_possession_threshold(height, ballYTransformed):
    global possession_threshold, zone

    if ballYTransformed <= (height // 3) - 7:
        possession_threshold = 15
        print("primer tercio")
        zone = "primer tercio"
    elif (height // 3) - 7 < ballYTransformed <= ((height // 3) * 2) + 7:
        possession_threshold = 30
        print("segundo tercio")
        zone = "segundo tercio"
    else:
        possession_threshold = 40
        print("tercer tercio")
        zone = "tercer tercio"

def pitch_side_aux(width, ballXtransformed):
    global side_aux

    if ballXtransformed < (width // 2):
        side_aux = 0
    else:
        side_aux = 1


def eliminar_regiones_pequenas(imagen_binaria, umbral_area = 6000):
    # Etiquetar las regiones conectadas en la imagen
    etiquetas, num_etiquetas = label(imagen_binaria)

    # Obtener las propiedades de las regiones etiquetadas
    propiedades = regionprops(etiquetas)

    # Crear una máscara para almacenar las regiones que deseamos mantener
    mascara = np.zeros_like(imagen_binaria)

    # Filtrar las regiones cuyo tamaño es mayor o igual que el umbral especificado
    for region in propiedades:
        if region.area >= umbral_area:
            mascara[etiquetas == region.label] = 1

    # Aplicar la máscara a la imagen binaria original para eliminar las regiones pequeñas
    imagen_filtrada = imagen_binaria * mascara

    return imagen_filtrada

def homographyMatrixUpdate(frameNum):
    global new_homography_matrix,actual_homography_matrix, discarded
    if frameNum == 0 or discarded == 10:
        actual_homography_matrix = new_homography_matrix
        discarded = 0
        #print("primer frame")
    else:
        # Definir los vectores de base ei (pueden ser cualquier vector)
        ei = np.eye(3)

        # Calcular las matrices de proyección pi_a y pi_b
        pi_a = np.dot(actual_homography_matrix, ei)
        pi_b = np.dot(new_homography_matrix, ei)

        # Calcular la diferencia entre las matrices de proyección
        dif = pi_a - pi_b
        dif = np.linalg.norm(dif)
        print("dif = " + str(dif))
        if dif <= 0.2:
            actual_homography_matrix = new_homography_matrix
            discarded = 0
            #print("la cambiamos")
        else:
            discarded += 1
            #print("se queda la anterior")




if __name__ == '__main__':
    df = pd.read_csv("hsv_teams.csv", header=0, sep = ';')
    #print(list(df.columns))
    HOME = os.getcwd()
    #print(HOME)
    clips = "D:\clips_futbol\listos"


    own_trained_location = HOME + "/runs\detect\Segunda_prueba_larga_nuevoEntreno_S_autobatch\weights/best.pt"
    # Load the own trained model
    model = YOLO(own_trained_location)
    #persepctive_transform = Perspective_Transform()

    clip_name = '/RMAvsSEV.mp4'


    # Define path to video file
    video_path = clips + clip_name

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    previousBallPos = None
    ballPos = None
    yoloBallPos = None
    teamInPossession = None
    lastTeamInPossession = None
    VideoWidth = cap.get(3)  # float `width`
    VideoHeight = cap.get(4)  # float `height`




    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # Black Image (Soccer Field)
    bg_ratio = int(np.ceil(w / (3 * 115)))
    gt_img = cv2.imread('inference/green.jpg')
    gt_img = cv2.resize(gt_img, (115 * bg_ratio, 74 * bg_ratio))
    gt_h, gt_w, _ = gt_img.shape
    frame_num = 0

    query_index = 0
    current_directory = str(Path(__file__).resolve().parent) + "/Perspective_Transformation/python_codes"
    # Load data
    # database
    deep_database_directory = current_directory + "/data_2/features/feature_camera_91k.mat"
    data = sio.loadmat(deep_database_directory)
    database_features = data['features']
    database_cameras = data['cameras']
    deep_model_directory = current_directory + "/deep/deep_network.pth"
    net, data_transform, device = initialize_deep_feature(deep_model_directory)
    template_h = 74  # yard, soccer template
    template_w = 115

    # testing edge image from two-GAN
    twoGanModel = initialize_two_GAN(current_directory)


    #warped_out = cv2.VideoWriter(current_directory + r"/warped_output.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 1,
                                #(460, 296))
    #retrieved_out = cv2.VideoWriter(current_directory + r"/retrieved_output.avi",
                                   #cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 1, (1280, 720))

    if save:
        outVideo = cv2.VideoWriter(current_directory + clip_name[1:-4] + r"_yoloedTFM.avi",cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (1920,1080))
        outBird = cv2.VideoWriter(current_directory + clip_name[1:-4] + r"_birdviewTFM.avi",cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (gt_w, gt_h))

    abbr1, abbr2 = get_abbr(clip_name)
    team1, team2 = get_names(df, abbr1, abbr2)

    team1_home_hsv, team1_away_hsv, team2_home_hsv, team2_away_hsv, team1_gk_home_hsv, team1_gk_away_hsv, team2_gk_home_hsv, team2_gk_away_hsv = load_masks(df, team1, team2)
    possessions[team1] = 0
    possessions[team2] = 0
    possessions["primer tercio"] = 0
    possessions["segundo tercio"] = 0
    possessions["tercer tercio"] = 0
    '''passes[team1] = 0
    passes[team2] = 0
    missed_passes[team1] = 0
    missed_passes[team2] = 0'''
    side_time["left"] = 0
    side_time["right"] = 0

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
        bg_img = gt_img.copy()



        if success:
            # Run YOLOv8 inference on the frame, NOT persisting tracks between frames
            # First we make a clean copy of the frame, so we don't disturb any lines or objects detection
            # also, we create a smaller copy of the frame for the perspective transformation
            if cap.get(3) != 1280.0 or cap.get(4) != 720.0:
                frame2 = cv2.resize(frame, (1280, 720))  # ===> for videos which resolutions are greater


            if frame_num % 5 == 0:
                #thread_perspective_transform = threading.Thread(target= generate_transform_matrix, args=(frame2, twoGanModel, data_transform, device, current_directory, template_h, template_w, database_features))
                #refined_h = generate_transform_matrix(frame2, twoGanModel, data_transform, device, current_directory, template_h, template_w, database_features)
                #thread_perspective_transform.start()
                generate_transform_matrix(frame2, twoGanModel, data_transform, device, current_directory, template_h, template_w, database_features)
                homographyMatrixUpdate(frame_num)


            copy_frame = frame.copy()
            results = model(frame)

            copy_frame = copy_frame[10:-10 , 10:-10]

            # Extract bounding boxes, classes, names, and confidences
            boxes = results[0].boxes.xyxy.tolist()
            classes = results[0].boxes.cls.tolist()
            names = results[0].names
            confidences = results[0].boxes.conf.tolist()
            #track_ids = results[0].boxes.id.int().tolist()
            '''print(classes)
            print(names)'''

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

                x_center = (x1 + x2) / 2
                y_center = y2
                #print(str(x_center) + ", " + str(y_center))

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

                    #nam, color = get_team(cropped_player, (red_mask_lower, red_mask_upper, green_mask_lower, green_mask_upper))
                    nam, color = get_team_improved(cropped_player,
                                          team1_home_hsv, team1_away_hsv, team2_home_hsv, team2_away_hsv, team1_gk_home_hsv, team1_gk_away_hsv, team2_gk_home_hsv, team2_gk_away_hsv)


                    # Yolo, sometimes, makes wrong classifications, saying that a referee is a player
                    if nam != "Arbitro":
                        players.append(((x1,y2), (x2, y2), nam))

                    #cv2.waitKey(0)

                # Visualize the results on the frame
                if name != "Ball":
                    #annotated_frame = results[0].plot()
                    out = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                    t_size = cv2.getTextSize(str(confidence), 0, fontScale=1 / 2, thickness=1)[0]
                    out = cv2.rectangle(frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3),
                                  color, -1)
                    out = cv2.putText(frame, nam + ": " + str(confidence), (int(x1), int(y1) - 4), 0, 1 / 2, [0, 0, 0], thickness=1,
                                lineType=cv2.LINE_AA)
                    #out = cv2.circle(frame, (int(x1), int(y2)), 5, (0, 0, 255), -1)
                    #out = cv2.circle(frame, (int(x2), int(y2)), 5, (255, 255, 0), -1)


                    #thread_perspective_transform.join()
                    #homographyMatrixUpdate(frame_num)

                    # prueba bird eye
                    coord = transform_matrix(actual_homography_matrix, (x_center * (1280/1920), y_center * (720/1080)), (720, 1280), (gt_h, gt_w))
                    #print(coord)
                    cv2.circle(bg_img, coord, 3, color, -1)
                    # ----


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
                probs = [yoloConfidence, pt, 0.5]
                print(probs)
                idx = probs.index(max(probs))
                if idx == 2:
                    #print("velocity prediction")
                    ballPos = (gaussPred[0], gaussPred[1])
                    #out = cv2.circle(frame, (gaussPred[0], gaussPred[1]), 10, (255, 255, 255), 4)

                    coord = transform_matrix(actual_homography_matrix,
                                             (gaussPred[0] * (1280 / 1920), gaussPred[1] * (720 / 1080)), (720, 1280),
                                             (gt_h, gt_w))
                    # print(coord)

                    out = cv2.rectangle(frame, (int(ballPos[0] - 10), int(ballPos[1] - 10)),
                                        (int(ballPos[0] + 10), int(ballPos[1] + 10)), (255, 255, 255), 1)
                    t_size = cv2.getTextSize(str(confidence), 0, fontScale=1 / 2, thickness=1)[0]
                    out = cv2.rectangle(frame, (int(ballPos[0] - 10), int(ballPos[1] - 10) - t_size[1] - 3), (int(ballPos[0] - 10 + t_size[0]), int(ballPos[1] - 10) + 3) , (255, 255, 255), -1)
                    out = cv2.putText(frame,"pelota: prediction",
                                      (int(ballPos[0] - 10), int(ballPos[1] - 10) - 4), 0, 1 / 2, [0, 0, 0],
                                      thickness=1,
                                      lineType=cv2.LINE_AA)
                    cv2.circle(bg_img, coord, 3, (255, 255, 255), -1)

                else:
                    #print("yolo or yolo gaussian")
                    ballPos = yoloBallPos
                    coord = transform_matrix(actual_homography_matrix,
                                             (ballPos[0] * (1280 / 1920), ballPos[1] * (720 / 1080)), (720, 1280),
                                             (gt_h, gt_w))

                    out = cv2.rectangle(frame, (int(ballPos[0] - 10), int(ballPos[1] - 10)), (int(ballPos[0] + 10), int(ballPos[1] + 10)), (255, 255, 255), 1)
                    t_size = cv2.getTextSize(str(confidence), 0, fontScale=1 / 2, thickness=1)[0]
                    out = cv2.rectangle(frame, (int(ballPos[0] - 10), int(ballPos[1] - 10) - t_size[1] - 3), (int(ballPos[0] - 10 + t_size[0]), int(ballPos[1] - 10) + 3) , (255, 255, 255), -1)
                    out = cv2.putText(frame, "pelota: " + str(yoloConfidence), (int(ballPos[0] - 10), int(ballPos[1] - 10) - 4), 0, 1 / 2, [0, 0, 0],
                                      thickness=1,
                                      lineType=cv2.LINE_AA)
                    cv2.circle(bg_img, coord, 3, (255,255,255), -1)

                update_possession_threshold(gt_h, coord[1])
                pitch_side_aux(gt_w, coord[0])
                update_XYpos(xPos, yPos, ballPos[0], ballPos[1])




                yoloConfidence = 0
            elif yoloBallPos is not None:
                previousBallPos = ballPos

                if len(xPos) == 5:
                    sd = (np.std(xPos) + np.std(yPos)) / 2
                    if sd == 0:
                        sd = 1

                #print("yolo or yolo gaussian")
                ballPos = yoloBallPos
                coord = transform_matrix(actual_homography_matrix,
                                         (ballPos[0] * (1280 / 1920), ballPos[1] * (720 / 1080)), (720, 1280),
                                         (gt_h, gt_w))

                out = cv2.rectangle(frame, (int(ballPos[0] - 15), int(ballPos[1] - 15)),
                                    (int(ballPos[0] + 15), int(ballPos[1] + 15)), (255, 255, 255), 1)
                t_size = cv2.getTextSize(str(confidence), 0, fontScale=1 / 2, thickness=1)[0]
                out = cv2.rectangle(frame, (int(ballPos[0] - 15), int(ballPos[1] - 10) - t_size[1] - 3), (int(ballPos[0] - 10 + t_size[0]), int(ballPos[1] - 10) + 3) , (255, 255, 255), -1)
                out = cv2.putText(frame, "pelota: " + str(yoloConfidence), (int(ballPos[0] - 10), int(ballPos[1] - 10) - 4),
                                  0, 1 / 2, [0, 0, 0],
                                  thickness=1,
                                  lineType=cv2.LINE_AA)
                cv2.circle(bg_img, coord, 3, (255,255,255), -1)
                update_possession_threshold(gt_h, ballPos[1])
                pitch_side_aux(gt_w, coord[0])
                update_XYpos(xPos, yPos, ballPos[0], ballPos[1])
                yoloConfidence = 0

            if ballPos is not None:

                line_filtering(frame, copy_frame)

                # Side of the pitch calculation
                if middle_line is not None:
                    side = get_pitch_side(ballPos)
                else:
                    side = side_aux

                if side is not None:
                    if side == 0:
                        side_time["left"] += 1
                        '''out = cv2.putText(frame, "Actual = LEFT", (900, 500), 0, 1 / 2,
                                          [255, 0, 255],
                                          thickness=2,
                                          lineType=cv2.LINE_AA)'''
                    else:
                        side_time["right"] += 1
                        '''out = cv2.putText(frame, "Actual = RIGHT", (900, 500), 0, 1 / 2,
                                          [255, 0, 255],
                                          thickness=2,
                                          lineType=cv2.LINE_AA)'''


                # Possession calculations
                possessions[zone] += 1
                temp, dist = getPossessionTeam(players,ballPos)

                teamInPossession = temp
                if teamInPossession is not None:
                    possessions[teamInPossession] += 1

                '''if temp is None and teamInPossession is not None:
                    lastTeamInPossession = teamInPossession

                teamInPossession = temp
                if teamInPossession is not None:
                    possessions[teamInPossession] += 1
                    if lastTeamInPossession is not None and lastTeamInPossession == teamInPossession:
                        passes[teamInPossession] += 1
                        lastTeamInPossession = None
                        #print("això ha estat un bon pase noi")
                    elif lastTeamInPossession is not None and lastTeamInPossession != teamInPossession:
                        missed_passes[lastTeamInPossession] += 1
                        lastTeamInPossession = None
                        #print("aquest no es va criar a la masia, mal pase noi")'''


            '''if possessions[team1] != 0 or possessions[team2] != 0:
                out = cv2.putText(frame, team1 + " = " + str(round(100 * (possessions[team1] / (possessions[team1] + possessions[team2])))) + "%", (60, 100+90), 0, 1 / 2, [0, 0, 0],
                                  thickness=2,
                                  lineType=cv2.LINE_AA)
                out = cv2.putText(frame, team2 + " = " + str(round(100 * (possessions[team2] / (possessions[team1] + possessions[team2])))) + "%", (60, 115+90), 0, 1 / 2, [0, 0, 0],
                                  thickness=2,
                                  lineType=cv2.LINE_AA)
                out = cv2.putText(frame,"Actual = " + str(teamInPossession), (60, 130+90), 0, 1 / 2,
                                  [0, 0, 0],
                                  thickness=2,
                                  lineType=cv2.LINE_AA)'''

            '''if side_time["left"] != 0 or side_time["right"] != 0:
                out = cv2.putText(frame, "left = " + str(
                    round(100 * (side_time["left"] / (side_time["left"] + side_time["right"])))) + "%", (60, 145+90), 0, 1 / 2,
                                  [0, 0, 0],
                                  thickness=2,
                                  lineType=cv2.LINE_AA)
                out = cv2.putText(frame, "right = " + str(
                    round(100 * (side_time["right"] / (side_time["left"] + side_time["right"])))) + "%", (60, 160+90), 0, 1 / 2,
                                  [0, 0, 0],
                                  thickness=2,
                                  lineType=cv2.LINE_AA)'''

            if possession_threshold == 15:
                actual_third = "primer tercio"
            elif possession_threshold == 30:
                actual_third = "segundo_tercio"
            else:
                actual_third = "tercer tercio"

            '''out = cv2.putText(frame, actual_third, (60, 235 + 90), 0,
                              1 / 2,
                              [255, 0, 255],
                              thickness=1,
                              lineType=cv2.LINE_AA)

            if possessions["primer tercio"] != 0 or possessions["segundo tercio"] != 0 or possessions["tercer tercio"] != 0:
                out = cv2.putText(frame, "primer tercio = " + str(round(100 * (possessions["primer tercio"] / (possessions["primer tercio"] + possessions["segundo tercio"] + possessions["tercer tercio"])))) + "%", (60, 250+90), 0, 1 / 2, [0, 0, 0],
                                  thickness=2,
                                  lineType=cv2.LINE_AA)
                out = cv2.putText(frame, "segundo tercio = " + str(round(100 * (possessions["segundo tercio"] / (possessions["primer tercio"] + possessions["segundo tercio"] + possessions["tercer tercio"])))) + "%", (60, 265+90), 0, 1 / 2, [0, 0, 0],
                                  thickness=2,
                                  lineType=cv2.LINE_AA)
                out = cv2.putText(frame,"tercer tercio = " + str(round(100 * (possessions["tercer tercio"] / (possessions["primer tercio"] + possessions["segundo tercio"] + possessions["tercer tercio"])))) + "%", (60, 280+90), 0, 1 / 2,
                                  [0, 0, 0],
                                  thickness=2,
                                  lineType=cv2.LINE_AA)'''


            #print("Pusesió")
            #if possessions[team1] != 0 or possessions[team2] != 0:
                #print(team1 + " = " + str(100 * (possessions[team1] / (possessions[team1] + possessions[team2]))))
                #print(team2 + " = " + str(100 * (possessions[team2] / (possessions[team1] + possessions[team2]))))
            '''print("------------------------------------------------------")
            print("ADN barça:")
            print(team1 + " pases = " + str(passes[team1]))
            print(team1 + " pases fallados = " + str(missed_passes[team1]))
            print(team2 + " pases = " + str(passes[team2]))
            print(team2 + " pases fallados = " + str(missed_passes[team2]))'''
            # Display the annotated frame

            bg_img = cv2.line(bg_img, (0, (gt_h //3) - 7), (gt_w, (gt_h //3) - 7), (0,0,255), 1)
            bg_img = cv2.line(bg_img, (0, ((gt_h // 3) * 2) + 7), (gt_w, ((gt_h // 3) * 2) + 7), (0, 0, 255), 1)
            if not save:
                cv2.imshow('Bird eye', bg_img)
                cv2.imshow("YOLOv8 detection", out)

                # prints debug -------------------------------

                ## Warp source image to destination based on homography
                #im_out = cv2.warpPerspective(seg_map, np.linalg.inv(refined_h), (115, 74), borderMode=cv2.BORDER_CONSTANT)

                # frame2 = cv2.resize(frame2, (1280, 720), interpolation=cv2.INTER_CUBIC)

                #model_address = current_directory + "/model.jpg"
                #model_image = cv2.imread(model_address)
                #model_image = cv2.resize(model_image, (115, 74))

                #new_image = cv2.addWeighted(model_image, 1, im_out, 1, 0)

                #new_image = cv2.resize(new_image, (460, 296), interpolation=1)

                # Display images
                # cv2.imshow('frame', frame)
                # cv2.waitKey()
                # cv.imshow('overlayed image', im_out_2)
                # cv.waitKey()
                #cv2.imshow('Edge image of retrieved camera', retrieved_image)
                #cv2.imshow("NN",edge_map)
                #cv2.imshow("Warped Source Image", new_image)
                cv2.waitKey(0)

                # ---------------------------------
            else:
                outVideo.write(out)
                outBird.write(bg_img)
            print(frame_num)

            if not metrics:
                # waiting using waitKey method
                #cv2.waitKey(0)

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

            frame_num += 1
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    if save:
        outVideo.release()
        outBird.release()
    else:
        cv2.destroyAllWindows()

    if metrics:
        data = [['CLIP NAME', clip_name], ['YOLO GOOD DETECTION', frameMetrics.count("z")], ['YOLO BAD DETECTION', frameMetrics.count("x")], ['KALMAN GOOD DETECTION', frameMetrics.count("c")],
                ['KALMAN BAD DETECTION', frameMetrics.count("v")], ['NO DETECTION', frameMetrics.count("b")]]
        # Creates DataFrame.
        df = pd.DataFrame(data)
        # saving the dataframe
        name = clip_name[1:-4] + '.csv'
        df.to_csv(name)

    # Rutas de las imágenes de los escudos
    ruta_escudo_team1 = "imagenes/escudos/" + team1 + ".png"
    ruta_escudo_team2 = "imagenes/escudos/" + team2 + ".png"

    # Cargar las imágenes de los escudos
    img_escudo_team1 = Image.open(ruta_escudo_team1)
    img_escudo_team2 = Image.open(ruta_escudo_team2)
    gt_img = cv2.imread('inference/green.jpg')
    gt_h, gt_w, _ = gt_img.shape
    bg_img = gt_img.copy()
    bg_img = cv2.line(bg_img, (0, (gt_h // 3) - 7), (gt_w, (gt_h // 3) - 7), (0, 0, 0), 12)
    bg_img = cv2.line(bg_img, (0, ((gt_h // 3) * 2) + 7), (gt_w, ((gt_h // 3) * 2) + 7), (0, 0, 0), 12)

    # Crear los subplots
    fig, axs = plt.subplots(4, 1, figsize=(15, 5))

    # Subplot 1: Escudos de los equipos con porcentajes de posesión
    axs[0].imshow(img_escudo_team1)
    axs[0].text(1.1, 0.5, str(round(100 * (possessions[team1] / (possessions[team1] + possessions[team2])))) + "%",
                color='black', fontsize=24, ha='left', va='center',
                transform=axs[0].transAxes)
    axs[0].axis('off')

    axs[1].imshow(img_escudo_team2)
    axs[1].text(1.1, 0.5, str(round(100 * (possessions[team2] / (possessions[team1] + possessions[team2])))) + "%",
                color='black', fontsize=24, ha='left', va='center',
                transform=axs[1].transAxes)
    axs[1].axis('off')

    # Subplot 2: Porcentaje de posesión en diferentes zonas del campo
    axs[2].imshow(gt_img)

    # Porcentaje de posesión en el lado izquierdo
    left_percentage = str(round(100 * (side_time["left"] / (side_time["left"] + side_time["right"]))))
    axs[2].text(0.28, 0.5, str(left_percentage) + "%", color='white', fontsize=16, ha='center', va='center',
                transform=axs[2].transAxes)

    # Porcentaje de posesión en el lado derecho
    right_percentage = str(round(100 * (side_time["right"] / (side_time["left"] + side_time["right"]))))
    axs[2].text(0.72, 0.5, str(right_percentage) + "%", color='white', fontsize=16, ha='center', va='center',
                transform=axs[2].transAxes)

    axs[2].axis('off')

    # Subplot 3: Porcentaje de posesión en diferentes tercios del campo
    axs[3].imshow(bg_img)

    # Porcentaje de posesión en el lado izquierdo
    upper_percentage = str(round(100 * (possessions["primer tercio"] / (
                possessions["primer tercio"] + possessions["segundo tercio"] + possessions["tercer tercio"]))))
    axs[3].text(0.5, 0.85, str(upper_percentage) + "%", color='white', fontsize=12, ha='center', va='center',
                transform=axs[3].transAxes)

    # Porcentaje de posesión en el lado derecho
    middle_percentage = str(round(100 * (possessions["segundo tercio"] / (
                possessions["primer tercio"] + possessions["segundo tercio"] + possessions["tercer tercio"]))))
    axs[3].text(0.5, 0.5, str(middle_percentage) + "%", color='white', fontsize=12, ha='center', va='center',
                transform=axs[3].transAxes)

    lower_percentage = str(round(100 * (possessions["tercer tercio"] / (
                possessions["primer tercio"] + possessions["segundo tercio"] + possessions["tercer tercio"]))))
    axs[3].text(0.5, 0.15, str(lower_percentage) + "%", color='white', fontsize=12, ha='center', va='center',
                transform=axs[3].transAxes)

    axs[3].axis('off')

    # Mostrar los subplots
    #plt.show()
    plt.savefig(clip_name[1:-4] + '_possessions.png', bbox_inches='tight')