from pickletools import uint8
import cv2
import numpy as np
import threading
import socket
import time
import os
import transforms3d
import open3d as o3d
from mediapipe.python.solutions import drawing_styles, drawing_utils, holistic

# from calculate.PoseShapeCalculate import PoseshapeCalculator
# from calculate.LeftHandCalculate import LeftHandCalculate
from calculate import *
from livelink.pylivelink import PyLiveLink, BlendShape
from utils.drawing import Drawing
from calculate.blendshape_calculator import BlendshapeCalculator

# taken from: https://github.com/Rassibassi/mediapipeDemos
from customs.face_geometry import (  # isort:skip
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)

# points of the face model that will be used for SolvePnP later
points_idx = [33, 263, 61, 291, 199]
points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()

# Calculates the 3d rotation and 3d landmarks from the 2d landmarks
def calculate_rotation(face_landmarks, pcf: PCF, image_shape):
    frame_width, frame_height, channels = image_shape
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )

    dist_coeff = np.zeros((4, 1))

    landmarks = np.array(
        [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark[:468]]
    )

    landmarks = landmarks.T
    # print(landmarks.shape)

    metric_landmarks, pose_transform_mat = get_metric_landmarks(
        landmarks.copy(), pcf
    )

    model_points = metric_landmarks[0:3, points_idx].T
    image_points = (
        landmarks[0:2, points_idx].T
        * np.array([frame_width, frame_height])[None, :]
    )

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeff,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    return pose_transform_mat, metric_landmarks, rotation_vector, translation_vector


class MediapipeHolistic():

    def __init__(self, input = 0, ip='127.0.0.1', port = 11111, show_3d = False, hide_image = False, show_debug = False) -> None:

        self.input = input
        self.show_image = not hide_image
        self.show_debug = show_debug

        self.holistic = holistic.Holistic(
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.live_link = PyLiveLink(fps = 60, filter_size = 4)
        self.pose_calulator = PoseshapeCalculator()
        self.lhand_calculate = LeftHandCalculate()
        self.rhand_calculate = RightHandCalculate()
        self.blendshape_calulator = BlendshapeCalculator()

        self.ip = ip
        self.upd_port = port
        self.show_3d = show_3d
        
        self.image_height, self.image_width, channels = (480, 640, 3)
        # self.image_height, self.image_width, channels = (720, 1280, 3)

        #pseudo camera internals
        focal_length = self.image_width
        center = (self.image_width / 2, self.image_height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

        self.pcf = PCF(
            near=1,
            far=10000,
            frame_height=self.image_height,
            frame_width=self.image_width,
            fy=camera_matrix[1, 1],
        )

        # self.drawing_spec = drawing_utils.DrawingSpec(thickness=1, circle_radius=1)  # I comment it 09/12/2022    
        self.lock = threading.Lock()
        self.got_new_data = False
        self.network_data = b''
        self.network_thread = threading.Thread(target=self._network_loop, daemon=True)
        self.image = None
        self.image_blackground = np.zeros((480, 640, 3), dtype="uint8")


    def start(self):
        cap = None
        image = None

        # check if input is an image        
        if isinstance(self.input, str) and (self.input.lower().endswith(".jpg") or self.input.lower().endswith(".png")):
            image = cv2.imread(self.input)
            self.file = True   
        else:   
            input = self.input  
            try:
                input = int(self.input)
            except ValueError:
                input = self.input 

        # os.name == 'nt' nt = windows
        # os.name == 'possix' possix = Mac

        if os.name == 'nt':
            # will improve webcam input startup on windows 
            cap = cv2.VideoCapture(input, cv2.CAP_DSHOW) 
        else:
            cap = cv2.VideoCapture(input)                

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        
        # run the network loop in a separate thread
        self.network_thread.start()

        if cap is not None:
            # for camera and videos
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                if not self._process_image(image):
                    break  
            print("Video capture received no more frames.")        
            cap.release()
        
        else:
            # for input images
            while image is not None:
                if not self._process_image(image):
                    break

    
    def _network_loop(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:            
            s.connect((self.ip, self.upd_port))
            while True: 
                with self.lock:
                    if self.got_new_data:                               
                        s.sendall(self.network_data)
                        self.got_new_data = False
                time.sleep(0.01) # default 0.01
            


    def _process_image(self, image):
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Recolor image
        results = self.holistic.process(image) # Make Detection

        # Draw the holistic annotation on the image. # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_black = cv2.rectangle(self.image_blackground, (0,0), (640,480), (0,0,0), -10)

        # print(image.shape) # probably witdth*height of camera or something

        if results.pose_landmarks:
            
            pose_world_landmarks = results.pose_world_landmarks.landmark  # x y z 3D world
            pose_landmarks = results.pose_landmarks.landmark
            # left_hand_landmarks = results.left_hand_landmarks.landmark
            # right_hand_landmarks = results.right_hand_landmarks
            self.pose_calulator._value_store(
                self.live_link, pose_world_landmarks, pose_landmarks
            )

            drawing_utils.draw_landmarks(
                image=image, # image_black
                landmark_list=results.pose_landmarks,
                connections=holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
                )
        
        if results.face_landmarks:

            # Iris in Holistic still release
            pose_transform_mat, metric_landmarks, rotation_vector, translation_vector = calculate_rotation(results.face_landmarks, self.pcf, image.shape) 

            # draw the face mesh 
            drawing_utils.draw_landmarks(
                image=image, # image_black
                landmark_list=results.face_landmarks,
                connections=holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles
                .get_default_face_mesh_tesselation_style()
                )

            # draw the face contours
            drawing_utils.draw_landmarks(
                image=image, # image_black
                landmark_list=results.face_landmarks,
                connections=holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles
                .get_default_face_mesh_contours_style()
                )
                
            image = Drawing.draw_landmark_point(results.face_landmarks.landmark[468], image, color = (0, 0, 255))
            image = Drawing.draw_landmark_point(results.face_landmarks.landmark[473], image, color = (0, 255, 0))
        
            # calculate and set all the blendshapes    
            self.blendshape_calulator.calculate_blendshapes(
                self.live_link, metric_landmarks[0:3].T, results.face_landmarks.landmark)
            # calculate the head rotation out of the pose matrix
            eulerAngles = transforms3d.euler.mat2euler(pose_transform_mat)
            pitch = -eulerAngles[0]
            yaw = eulerAngles[1]
            roll = eulerAngles[2]
            self.live_link.set_blendshape(
                BlendShape.HeadPitch, pitch)
            self.live_link.set_blendshape(
                BlendShape.HeadRoll, roll)
            self.live_link.set_blendshape(
                BlendShape.HeadYaw, yaw)
            
        if results.left_hand_landmarks:
            left_hand_landmarks = results.left_hand_landmarks.landmark

            self.lhand_calculate._value_store(
                self.live_link, left_hand_landmarks
            )

            drawing_utils.draw_landmarks(
                    image=image, # image_black
                    landmark_list=results.left_hand_landmarks,
                    connections=holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style()
                )
            
        
        if results.right_hand_landmarks:
            right_hand_landmarks = results.right_hand_landmarks.landmark

            self.rhand_calculate._value_store(
                self.live_link, right_hand_landmarks
            )

            drawing_utils.draw_landmarks(
                    image=image, # image_black
                    landmark_list=results.right_hand_landmarks,
                    connections=holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style()
                )

        # Debug format settings
        white_bg = 0 * np.ones(shape=[720, 720, 3], dtype=np.uint8) # size of Debug Windows height*width
        text_coordinates = [25, 25]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        color = (0, 255, 0)

            
        if self.show_image:
            # cv2.imshow('MediaPipe Pose Estimate', cv2.flip(image.astype('uint8'), 1))
            # cv2.imshow('MediaPipe Pose Estimate', image.astype('uint8'))
            # cv2.imshow('Detection Camera', image_black)
            cv2.imshow('Detection Camera', image)
            if self.show_debug:
                for shape in BlendShape:
                    shape_debug_text = f'{shape.name}: {self.live_link.get_blendshape(BlendShape(shape.value)):.3f}'
                    cv2.putText(img=white_bg, text=shape_debug_text, org=tuple(text_coordinates), fontFace=font, fontScale=font_scale, color=color, thickness=1)
                    text_coordinates[1] += 20
                    if shape.value == 30: #start new column
                        text_coordinates = [300, 25]
                    # if shape.value == 61: #start new column
                    #     text_coordinates = [600, 25]
                    # if shape.value == 92: #start new column
                    #     text_coordinates = [900, 25]

                cv2.imshow('Debug', white_bg)

            if cv2.waitKey(5) & 0xFF == 27:
                return False
            elif cv2.getWindowProperty('Detection Camera', cv2.WND_PROP_VISIBLE) < 1:
                return False

        with self.lock:
            self.got_new_data = True
            self.network_data = self.live_link.encode()
        
        return True