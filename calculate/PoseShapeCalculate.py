import math
import numpy as np
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from livelink.pylivelink import PyLiveLink, BlendShape
from mediapipe.python.solutions import holistic
from typing import Optional
import time

class PoseshapeCalculator():

    def __init__(self) -> None:
        self.prev_deg = {}
        self.prev_deg_polar = {}
        self.prev_deg_azim = {}
        # self.smooth_deg = 30    # default = 30
        # self.accident_deg = 150    # default = 150

    def _value_store(self, live_link_pose: PyLiveLink, normalized_landmarks: RepeatedCompositeFieldContainer, landmark: Optional[RepeatedCompositeFieldContainer] = None):
        self.live_link_pose = live_link_pose
        self.landmarks_world = normalized_landmarks
        self.default_landmarks = landmark
        self._calculate_pose_landmarks()

    def calculate_angle(self, first_point3d, second_point3d, joint_name, smooth_deg):

        dX = first_point3d[0]-second_point3d[0]
        dY = first_point3d[1]-second_point3d[1] 
        dZ = first_point3d[2]-second_point3d[2]
                
        polar_angle = math.degrees(math.atan2(dZ, dX))   
        azimuthal_angle = math.degrees(math.atan2(math.sqrt(dZ*dZ + dX*dX), dY))+180

        if polar_angle < 0:
            polar_angle = 360 + polar_angle
        if azimuthal_angle < 0:
            azimuthal_angle = 360 + azimuthal_angle

        # first run
        if joint_name+"_x" not in self.prev_deg:
            self.prev_deg[joint_name+"_x"] = polar_angle
        if joint_name+"_y" not in self.prev_deg:
            self.prev_deg[joint_name+"_y"] = azimuthal_angle

        # smoothness
        dif_deg = self.prev_deg[joint_name+"_x"] - polar_angle  # this solution will get unsteady value from angle detection from camera
        if abs(dif_deg) > smooth_deg: # this solution will bring true direction of value
            self.prev_deg[joint_name+"_x"] = self.prev_deg[joint_name+"_x"] - smooth_deg*(dif_deg/abs(dif_deg))
        else:
            self.prev_deg[joint_name+"_x"] = polar_angle

        dif_deg = self.prev_deg[joint_name+"_y"] - azimuthal_angle
        if abs(dif_deg) > smooth_deg:
            self.prev_deg[joint_name+"_y"] = self.prev_deg[joint_name+"_y"] - smooth_deg*(dif_deg/abs(dif_deg))
        else:
            self.prev_deg[joint_name+"_y"] = azimuthal_angle

    def _calculate_position_xyz(self, point, joint_name, smooth_deg):
        
        _x = (point[0]-0.5)*10
        _y = (point[1]-0.5)*10
        _z = point[2]*10000

        # first run
        if joint_name+"_x" not in self.prev_deg:
            self.prev_deg[joint_name+"_x"] = _x
        if joint_name+"_y" not in self.prev_deg:
            self.prev_deg[joint_name+"_y"] = _y
        if joint_name+"_z" not in self.prev_deg:
            self.prev_deg[joint_name+"_z"] = _z

        #smoothness
        dif_dis = self.prev_deg[joint_name+"_x"] - _x
        if abs(dif_dis) > smooth_deg:
            self.prev_deg[joint_name+"_x"] = self.prev_deg[joint_name+"_x"] - smooth_deg*(dif_dis/abs(dif_dis))
        else:
            self.prev_deg[joint_name+"_x"] = _x

        dif_dis = self.prev_deg[joint_name+"_y"] - _y
        if abs(dif_dis) > smooth_deg:
            self.prev_deg[joint_name+"_y"] = self.prev_deg[joint_name+"_y"] - smooth_deg*(dif_dis/abs(dif_dis))
        else:
            self.prev_deg[joint_name+"_y"] = _y

        dif_dis = self.prev_deg[joint_name+"_z"] - _z
        if abs(dif_dis) > smooth_deg:
            self.prev_deg[joint_name+"_z"] = self.prev_deg[joint_name+"_z"] - smooth_deg*(dif_dis/abs(dif_dis))
        else:
            self.prev_deg[joint_name+"_z"] = _z



    def calculate_angle_matrix(self, a, b, smooth_deg):
        A = a
        B = b
        D = A-B
        # atan2(D[2], D[0]) z , x
        # atan2(sqrt(D[2]**2+D[0]**2), D[1])+pi  z ,x , y
        polar = []
        azimuthal = []
        dif_deg_polar = []
        dif_deg_azim = []
        # print(np.shape(D))
        # print(D.shape[1])
        # print("Check ", D[i, 1])
        for i in range(D.shape[0]):
            polar.append(np.degrees(np.arctan2(D[i, 2], D[i, 0])))
            azimuthal.append(np.degrees(np.arctan2(np.linalg.norm(D[i, 2]-D[i, 0]), D[i, 1]))+180)
            if polar[i] < 0:
                polar[i] = 360 + polar[i]
            if azimuthal[i] < 0:
                azimuthal[i] = 360 + azimuthal[i]

            # first run
            if i not in self.prev_deg_polar:
                self.prev_deg_polar[i] = polar[i]
            if i not in self.prev_deg_azim:
                self.prev_deg_azim[i] = azimuthal[i]

            # smoothness                        
            dif_deg_polar.append(self.prev_deg_polar[i] - polar[i])                       # this solution will get unsteady value from angle detection from camera
            if abs(dif_deg_polar[i]) > smooth_deg:                                  # this solution will bring true direction of value
                self.prev_deg_polar[i] = self.prev_deg_polar[i] - smooth_deg*(dif_deg_polar[i]/abs(dif_deg_polar[i]))
            else:
               self.prev_deg_polar[i] = polar[i]

            dif_deg_azim.append(self.prev_deg_azim[i] - azimuthal[i])                       # this solution will get unsteady value from angle detection from camera
            if abs(dif_deg_azim[i]) > smooth_deg:                                  # this solution will bring true direction of value
                self.prev_deg_azim[i] = self.prev_deg_azim[i] - smooth_deg*(dif_deg_azim[i]/abs(dif_deg_azim[i]))
            else:
               self.prev_deg_azim[i] = azimuthal[i]
    
    def _calculate_pose_landmarks(self):

        try:
            ####!! Experiment !!#####
            # Get coordinates
            # https://google.github.io/mediapipe/solutions/pose#pose-landmark-model-blazepose-ghum-3d

            # shoulder_value
            shoulder_left_xyz = [self.landmarks_world[holistic.PoseLandmark.LEFT_SHOULDER.value].x, self.landmarks_world[holistic.PoseLandmark.LEFT_SHOULDER.value].y, self.landmarks_world[holistic.PoseLandmark.LEFT_SHOULDER.value].z]
            shoulder_right_xyz = [self.landmarks_world[holistic.PoseLandmark.RIGHT_SHOULDER.value].x, self.landmarks_world[holistic.PoseLandmark.RIGHT_SHOULDER.value].y, self.landmarks_world[holistic.PoseLandmark.RIGHT_SHOULDER.value].z]

            # elbow_value
            elbow_left_xyz = [self.landmarks_world[holistic.PoseLandmark.LEFT_ELBOW.value].x, self.landmarks_world[holistic.PoseLandmark.LEFT_ELBOW.value].y, self.landmarks_world[holistic.PoseLandmark.LEFT_ELBOW.value].z]
            elbow_right_xyz = [self.landmarks_world[holistic.PoseLandmark.RIGHT_ELBOW.value].x, self.landmarks_world[holistic.PoseLandmark.RIGHT_ELBOW.value].y, self.landmarks_world[holistic.PoseLandmark.RIGHT_ELBOW.value].z]

            # wrist_value
            wrist_left_xyz = [self.landmarks_world[holistic.PoseLandmark.LEFT_WRIST.value].x, self.landmarks_world[holistic.PoseLandmark.LEFT_WRIST.value].y, self.landmarks_world[holistic.PoseLandmark.LEFT_WRIST.value].z]
            wrist_right_xyz = [self.landmarks_world[holistic.PoseLandmark.RIGHT_WRIST.value].x, self.landmarks_world[holistic.PoseLandmark.RIGHT_WRIST.value].y, self.landmarks_world[holistic.PoseLandmark.RIGHT_WRIST.value].z]

            # hip_value
            hip_left_xyz = [self.landmarks_world[holistic.PoseLandmark.LEFT_HIP.value].x, self.landmarks_world[holistic.PoseLandmark.LEFT_HIP.value].y, 0.2-(self.landmarks_world[holistic.PoseLandmark.LEFT_HIP.value].z)/(10)]
            hip_right_xyz = [self.landmarks_world[holistic.PoseLandmark.RIGHT_HIP.value].x, self.landmarks_world[holistic.PoseLandmark.RIGHT_HIP.value].y, 0.2-(self.landmarks_world[holistic.PoseLandmark.RIGHT_HIP.value].z)/(10)]

            hip_left_xyz_n = [self.landmarks_world[holistic.PoseLandmark.LEFT_HIP.value].x, self.landmarks_world[holistic.PoseLandmark.LEFT_HIP.value].y, self.landmarks_world[holistic.PoseLandmark.LEFT_HIP.value].z]
            hip_right_xyz_n = [self.landmarks_world[holistic.PoseLandmark.RIGHT_HIP.value].x, self.landmarks_world[holistic.PoseLandmark.RIGHT_HIP.value].y, self.landmarks_world[holistic.PoseLandmark.RIGHT_HIP.value].z]
            
            # knee_value
            knee_left_xyz = [self.landmarks_world[holistic.PoseLandmark.LEFT_KNEE.value].x, self.landmarks_world[holistic.PoseLandmark.LEFT_KNEE.value].y, self.landmarks_world[holistic.PoseLandmark.LEFT_KNEE.value].z]
            knee_right_xyz = [self.landmarks_world[holistic.PoseLandmark.RIGHT_KNEE.value].x, self.landmarks_world[holistic.PoseLandmark.RIGHT_KNEE.value].y, self.landmarks_world[holistic.PoseLandmark.RIGHT_KNEE.value].z]

            # ankle_value
            ankle_left_xyz = [self.landmarks_world[holistic.PoseLandmark.LEFT_ANKLE.value].x, self.landmarks_world[holistic.PoseLandmark.LEFT_ANKLE.value].y, self.landmarks_world[holistic.PoseLandmark.LEFT_ANKLE.value].z]
            ankle_right_xyz = [self.landmarks_world[holistic.PoseLandmark.RIGHT_ANKLE.value].x, self.landmarks_world[holistic.PoseLandmark.RIGHT_ANKLE.value].y, self.landmarks_world[holistic.PoseLandmark.RIGHT_ANKLE.value].z]

            # foot_value
            foot_left_index_xyz = [self.landmarks_world[holistic.PoseLandmark.LEFT_FOOT_INDEX.value].x, self.landmarks_world[holistic.PoseLandmark.LEFT_FOOT_INDEX.value].y, self.landmarks_world[holistic.PoseLandmark.LEFT_FOOT_INDEX.value].z]
            foot_right_index_xyz = [self.landmarks_world[holistic.PoseLandmark.RIGHT_FOOT_INDEX.value].x, self.landmarks_world[holistic.PoseLandmark.RIGHT_FOOT_INDEX.value].y, self.landmarks_world[holistic.PoseLandmark.RIGHT_FOOT_INDEX.value].z]

            #! default value landmark !#
            d_hip_left_xyz = [self.default_landmarks[holistic.PoseLandmark.LEFT_HIP.value].x, self.default_landmarks[holistic.PoseLandmark.LEFT_HIP.value].y,  self.default_landmarks[holistic.PoseLandmark.LEFT_HIP.value].z]
            d_hip_right_xyz = [self.default_landmarks[holistic.PoseLandmark.RIGHT_HIP.value].x, self.default_landmarks[holistic.PoseLandmark.RIGHT_HIP.value].y, self.default_landmarks[holistic.PoseLandmark.RIGHT_HIP.value].z]

            ## Experiment ##
            aa = np.matrix([shoulder_left_xyz, wrist_left_xyz,
                            elbow_right_xyz, elbow_right_xyz,
                            knee_left_xyz, knee_right_xyz,
                            knee_left_xyz, knee_right_xyz,
                            ankle_left_xyz, ankle_right_xyz,
                            shoulder_right_xyz, hip_right_xyz_n])

            bb = np.matrix([elbow_left_xyz, elbow_left_xyz,
                            shoulder_right_xyz, wrist_right_xyz,
                            hip_left_xyz, hip_right_xyz,
                            ankle_left_xyz, ankle_right_xyz,
                            foot_left_index_xyz, foot_right_index_xyz,
                            shoulder_left_xyz, hip_left_xyz_n])

            self.calculate_angle_matrix(aa,bb, smooth_deg=15) # default 30
            self.live_link_pose.set_blendshape(BlendShape.ShoulderLeftAngle_z, self.prev_deg_polar[0])
            self.live_link_pose.set_blendshape(BlendShape.ShoulderLeftAngle_x, self.prev_deg_azim[0])
            self.live_link_pose.set_blendshape(BlendShape.ElbowLeftAngle_z, self.prev_deg_polar[1])
            self.live_link_pose.set_blendshape(BlendShape.ElbowLeftAngle_x, self.prev_deg_azim[1])
            self.live_link_pose.set_blendshape(BlendShape.ShoulderRightAngle_z, self.prev_deg_polar[2])
            self.live_link_pose.set_blendshape(BlendShape.ShoulderRightAngle_x, self.prev_deg_azim[2])
            self.live_link_pose.set_blendshape(BlendShape.ElbowRightAngle_z, self.prev_deg_polar[3])
            self.live_link_pose.set_blendshape(BlendShape.ElbowRightAngle_x, self.prev_deg_azim[3])
            self.live_link_pose.set_blendshape(BlendShape.ThighLeftAngle_z, self.prev_deg_polar[4])
            self.live_link_pose.set_blendshape(BlendShape.ThighLeftAngle_x, self.prev_deg_azim[4])
            self.live_link_pose.set_blendshape(BlendShape.ThighRightAngle_z, self.prev_deg_polar[5])
            self.live_link_pose.set_blendshape(BlendShape.ThighRightAngle_x, self.prev_deg_azim[5])
            self.live_link_pose.set_blendshape(BlendShape.ShinLeftAngle_z, self.prev_deg_polar[6])
            self.live_link_pose.set_blendshape(BlendShape.ShinLeftAngle_x, self.prev_deg_azim[6])
            self.live_link_pose.set_blendshape(BlendShape.ShinRightAngle_z, self.prev_deg_polar[7])
            self.live_link_pose.set_blendshape(BlendShape.ShinRightAngle_x, self.prev_deg_azim[7])
            self.live_link_pose.set_blendshape(BlendShape.FootLeftAngle_x, self.prev_deg_polar[8])
            self.live_link_pose.set_blendshape(BlendShape.FootLeftAngle_y, self.prev_deg_azim[8])
            self.live_link_pose.set_blendshape(BlendShape.FootRightAngle_x, self.prev_deg_polar[9])
            self.live_link_pose.set_blendshape(BlendShape.FootRightAngle_y, self.prev_deg_azim[9])
            self.live_link_pose.set_blendshape(BlendShape.BodyShoulderAngle_y, self.prev_deg_polar[10])
            self.live_link_pose.set_blendshape(BlendShape.BodyShoulderAngle_x, self.prev_deg_azim[10])
            self.live_link_pose.set_blendshape(BlendShape.BodyHipAngle_y, self.prev_deg_polar[11])
            self.live_link_pose.set_blendshape(BlendShape.BodyHipAngle_x, self.prev_deg_azim[11])

            center_shoulder = (np.array(shoulder_left_xyz)+np.array(shoulder_right_xyz))/2
            center_hip = (np.array(hip_left_xyz)+np.array(hip_left_xyz))/2
            self.calculate_angle(center_hip, center_shoulder, "BodyPitch", smooth_deg=15)
            self.live_link_pose.set_blendshape(BlendShape.BodyAnglePitch_z, self.prev_deg["BodyPitch_y"])

            d_center_hip = (np.array(d_hip_right_xyz)+np.array(d_hip_left_xyz))/2
            self._calculate_position_xyz(d_center_hip, "Walk", smooth_deg=0.8)
            self.live_link_pose.set_blendshape(BlendShape.Walk_x, self.prev_deg["Walk_x"])
            self.live_link_pose.set_blendshape(BlendShape.Walk_y, self.prev_deg["Walk_y"])
            self.live_link_pose.set_blendshape(BlendShape.Walk_z, self.prev_deg["Walk_z"])

        except Exception as e:
            print(e)