import math
import numpy as np
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from livelink.pylivelink import PyLiveLink, BlendShape
from mediapipe.python.solutions import holistic
from typing import Optional

class RightHandCalculate():

    def __init__(self) -> None:
        self.prev_deg = {}

    def _value_store(self, live_link_rhand: PyLiveLink, normalized_landmarks: RepeatedCompositeFieldContainer, image: Optional[np.ndarray] = None):
        self.live_link_rhand = live_link_rhand
        self.landmark_rhand = normalized_landmarks
        self.image = image
        self._calculate_rhand_landmarks()

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


    def _calculate_rhand_landmarks(self):
        try:
            # Get coordinates

            ###########################  THUMB FINGER ################################################
            wrist_thumb_right_xyz = [self.landmark_rhand[holistic.HandLandmark.WRIST.value].x, self.landmark_rhand[holistic.HandLandmark.WRIST.value].y, self.landmark_rhand[holistic.HandLandmark.WRIST.value].z]

            thumb_cmc_right_xyz = [self.landmark_rhand[holistic.HandLandmark.THUMB_CMC.value].x, self.landmark_rhand[holistic.HandLandmark.THUMB_CMC.value].y, self.landmark_rhand[holistic.HandLandmark.THUMB_CMC.value].z]

            thumb_mcp_right_xyz = [self.landmark_rhand[holistic.HandLandmark.THUMB_MCP.value].x, self.landmark_rhand[holistic.HandLandmark.THUMB_MCP.value].y, self.landmark_rhand[holistic.HandLandmark.THUMB_MCP.value].z]

            thumb_ip_right_xyz = [self.landmark_rhand[holistic.HandLandmark.THUMB_IP.value].x, self.landmark_rhand[holistic.HandLandmark.THUMB_IP.value].y, self.landmark_rhand[holistic.HandLandmark.THUMB_IP.value].z]

            thumb_tip_right_xyz = [self.landmark_rhand[holistic.HandLandmark.THUMB_TIP.value].x, self.landmark_rhand[holistic.HandLandmark.THUMB_TIP.value].y, self.landmark_rhand[holistic.HandLandmark.THUMB_TIP.value].z]

            ###########################  Index FINGER ##################################################
            wrist_index_right_xyz = [self.landmark_rhand[holistic.HandLandmark.WRIST.value].x, self.landmark_rhand[holistic.HandLandmark.WRIST.value].y, self.landmark_rhand[holistic.HandLandmark.WRIST.value].z]

            index_finger_mcp_right_xyz = [self.landmark_rhand[holistic.HandLandmark.INDEX_FINGER_MCP.value].x, self.landmark_rhand[holistic.HandLandmark.INDEX_FINGER_MCP.value].y, self.landmark_rhand[holistic.HandLandmark.INDEX_FINGER_MCP.value].z]

            index_finger_pip_right_xyz = [self.landmark_rhand[holistic.HandLandmark.INDEX_FINGER_PIP.value].x, self.landmark_rhand[holistic.HandLandmark.INDEX_FINGER_PIP.value].y, self.landmark_rhand[holistic.HandLandmark.INDEX_FINGER_PIP.value].z]

            index_finger_dip_right_xyz = [self.landmark_rhand[holistic.HandLandmark.INDEX_FINGER_DIP.value].x, self.landmark_rhand[holistic.HandLandmark.INDEX_FINGER_DIP.value].y, self.landmark_rhand[holistic.HandLandmark.INDEX_FINGER_DIP.value].z]

            index_finger_tip_right_xyz = [self.landmark_rhand[holistic.HandLandmark.INDEX_FINGER_TIP.value].x, self.landmark_rhand[holistic.HandLandmark.INDEX_FINGER_TIP.value].y, self.landmark_rhand[holistic.HandLandmark.INDEX_FINGER_TIP.value].z]

            ###########################  Middle FINGER ##################################################
            wrist_middle_right_xyz = [self.landmark_rhand[holistic.HandLandmark.WRIST.value].x, self.landmark_rhand[holistic.HandLandmark.WRIST.value].y, self.landmark_rhand[holistic.HandLandmark.WRIST.value].z]

            middle_finger_mcp_right_xyz = [self.landmark_rhand[holistic.HandLandmark.MIDDLE_FINGER_MCP.value].x, self.landmark_rhand[holistic.HandLandmark.MIDDLE_FINGER_MCP.value].y, self.landmark_rhand[holistic.HandLandmark.MIDDLE_FINGER_MCP.value].z]

            middle_finger_pip_right_xyz = [self.landmark_rhand[holistic.HandLandmark.MIDDLE_FINGER_PIP.value].x, self.landmark_rhand[holistic.HandLandmark.MIDDLE_FINGER_PIP.value].y, self.landmark_rhand[holistic.HandLandmark.MIDDLE_FINGER_PIP.value].z]

            middle_finger_dip_right_xyz = [self.landmark_rhand[holistic.HandLandmark.MIDDLE_FINGER_DIP.value].x, self.landmark_rhand[holistic.HandLandmark.MIDDLE_FINGER_DIP.value].y, self.landmark_rhand[holistic.HandLandmark.MIDDLE_FINGER_DIP.value].z]

            middle_finger_tip_right_xyz = [self.landmark_rhand[holistic.HandLandmark.MIDDLE_FINGER_TIP.value].x, self.landmark_rhand[holistic.HandLandmark.MIDDLE_FINGER_TIP.value].y, self.landmark_rhand[holistic.HandLandmark.MIDDLE_FINGER_TIP.value].z]

            ###########################  Ring FINGER ##################################################
            wrist_ring_right_xyz = [self.landmark_rhand[holistic.HandLandmark.WRIST.value].x, self.landmark_rhand[holistic.HandLandmark.WRIST.value].y, self.landmark_rhand[holistic.HandLandmark.WRIST.value].z]

            ring_finger_mcp_right_xyz = [self.landmark_rhand[holistic.HandLandmark.RING_FINGER_MCP.value].x, self.landmark_rhand[holistic.HandLandmark.RING_FINGER_MCP.value].y, self.landmark_rhand[holistic.HandLandmark.RING_FINGER_MCP.value].z]

            ring_finger_pip_right_xyz = [self.landmark_rhand[holistic.HandLandmark.RING_FINGER_PIP.value].x, self.landmark_rhand[holistic.HandLandmark.RING_FINGER_PIP.value].y, self.landmark_rhand[holistic.HandLandmark.RING_FINGER_PIP.value].z]

            ring_finger_dip_right_xyz = [self.landmark_rhand[holistic.HandLandmark.RING_FINGER_DIP.value].x, self.landmark_rhand[holistic.HandLandmark.RING_FINGER_DIP.value].y, self.landmark_rhand[holistic.HandLandmark.RING_FINGER_DIP.value].z]

            ring_finger_tip_right_xyz = [self.landmark_rhand[holistic.HandLandmark.RING_FINGER_TIP.value].x, self.landmark_rhand[holistic.HandLandmark.RING_FINGER_TIP.value].y, self.landmark_rhand[holistic.HandLandmark.RING_FINGER_TIP.value].z]

            ###########################  Pinky FINGER ##################################################
            wrist_pinky_right_xyz = [self.landmark_rhand[holistic.HandLandmark.WRIST.value].x, self.landmark_rhand[holistic.HandLandmark.WRIST.value].y, self.landmark_rhand[holistic.HandLandmark.WRIST.value].z]

            pinky_mcp_right_xyz = [self.landmark_rhand[holistic.HandLandmark.PINKY_MCP.value].x, self.landmark_rhand[holistic.HandLandmark.PINKY_MCP.value].y, self.landmark_rhand[holistic.HandLandmark.PINKY_MCP.value].z]

            pinky_pip_right_xyz = [self.landmark_rhand[holistic.HandLandmark.PINKY_PIP.value].x, self.landmark_rhand[holistic.HandLandmark.PINKY_PIP.value].y, self.landmark_rhand[holistic.HandLandmark.PINKY_PIP.value].z]

            pinky_dip_right_xyz = [self.landmark_rhand[holistic.HandLandmark.PINKY_DIP.value].x, self.landmark_rhand[holistic.HandLandmark.PINKY_DIP.value].y, self.landmark_rhand[holistic.HandLandmark.PINKY_DIP.value].z]

            pinky_tip_right_xyz = [self.landmark_rhand[holistic.HandLandmark.PINKY_TIP.value].x, self.landmark_rhand[holistic.HandLandmark.PINKY_TIP.value].y, self.landmark_rhand[holistic.HandLandmark.PINKY_TIP.value].z]

            ############### WristThumbRightAngle ###############
            self.calculate_angle(wrist_thumb_right_xyz, thumb_cmc_right_xyz, "WristThumbRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.WristRightThumb_x, self.prev_deg["WristThumbRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.WristRightThumb_y, self.prev_deg["WristThumbRightAngle_y"])

            ############### ThumbCMCRightAngle ###############
            self.calculate_angle(thumb_cmc_right_xyz, thumb_mcp_right_xyz, "ThumbCMCRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.ThumbCMCRightAngle_x, self.prev_deg["ThumbCMCRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.ThumbCMCRightAngle_y, self.prev_deg["ThumbCMCRightAngle_y"])

            ############### ThumbMCPRightAngle ###############
            self.calculate_angle(thumb_mcp_right_xyz, thumb_ip_right_xyz, "ThumbMCPRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.ThumbMCPRightAngle_x, self.prev_deg["ThumbMCPRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.ThumbMCPRightAngle_y, self.prev_deg["ThumbMCPRightAngle_y"])

            ############### ThumbIPRightAngle ###############
            self.calculate_angle(thumb_ip_right_xyz, thumb_tip_right_xyz, "ThumbIPRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.ThumbIPRightAngle_x, self.prev_deg["ThumbIPRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.ThumbIPRightAngle_y, self.prev_deg["ThumbIPRightAngle_y"])
            
            ############### WristIndexRightAngle ###############
            self.calculate_angle(wrist_index_right_xyz, index_finger_mcp_right_xyz, "WristIndexRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.WristRightIndex_x, self.prev_deg["WristIndexRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.WristRightIndex_y, self.prev_deg["WristIndexRightAngle_y"])

            ############### IndexFingerMCPRightAngle ###############
            self.calculate_angle(index_finger_mcp_right_xyz, index_finger_pip_right_xyz, "IndexFingerMCPRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.IndexFingerMCPRightAngle_x, self.prev_deg["IndexFingerMCPRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.IndexFingerMCPRightAngle_y, self.prev_deg["IndexFingerMCPRightAngle_y"])

            ############### IndexFingerPIPRightAngle ###############
            self.calculate_angle(index_finger_pip_right_xyz, index_finger_dip_right_xyz, "IndexFingerPIPRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.IndexFingerPIPRightAngle_x, self.prev_deg["IndexFingerPIPRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.IndexFingerPIPRightAngle_y, self.prev_deg["IndexFingerPIPRightAngle_y"])

            ############### IndexFingerDIPRightAngle ###############
            self.calculate_angle(index_finger_dip_right_xyz, index_finger_tip_right_xyz, "IndexFingerDIPRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.IndexFingerDIPRightAngle_x, self.prev_deg["IndexFingerDIPRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.IndexFingerDIPRightAngle_y, self.prev_deg["IndexFingerDIPRightAngle_y"])

            ############### WristMiddleRightAngle ###############
            self.calculate_angle(wrist_middle_right_xyz, middle_finger_mcp_right_xyz, "WristMiddleRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.WristRightMiddle_x, self.prev_deg["WristMiddleRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.WristRightMiddle_y, self.prev_deg["WristMiddleRightAngle_y"])

            ############### MiddleFingerMCPRightAngle ###############
            self.calculate_angle(middle_finger_mcp_right_xyz, middle_finger_pip_right_xyz, "MiddleFingerMCPRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.MiddleFingerMCPRightAngle_x, self.prev_deg["MiddleFingerMCPRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.MiddleFingerMCPRightAngle_y, self.prev_deg["MiddleFingerMCPRightAngle_y"])

            ############### MiddleFingerPIPRightAngle ###############
            self.calculate_angle(middle_finger_pip_right_xyz, middle_finger_dip_right_xyz, "MiddleFingerPIPRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.MiddleFingerPIPRightAngle_x, self.prev_deg["MiddleFingerPIPRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.MiddleFingerPIPRightAngle_y, self.prev_deg["MiddleFingerPIPRightAngle_y"])

            ############### MiddleFingerDIPRightAngle ###############
            self.calculate_angle(middle_finger_dip_right_xyz, middle_finger_tip_right_xyz, "MiddleFingerDIPRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.MiddleFingerDIPRightAngle_x, self.prev_deg["MiddleFingerDIPRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.MiddleFingerDIPRightAngle_y, self.prev_deg["MiddleFingerDIPRightAngle_y"])

            ############### WristRingRightAngle ###############
            self.calculate_angle(wrist_ring_right_xyz, ring_finger_mcp_right_xyz, "WristRingRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.WristRightRing_x, self.prev_deg["WristRingRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.WristRightRing_y, self.prev_deg["WristRingRightAngle_y"])

            ############### RingFingerMCPRightAngle ###############
            self.calculate_angle(ring_finger_mcp_right_xyz, ring_finger_pip_right_xyz, "RingFingerMCPRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.RingFingerMCPRightAngle_x, self.prev_deg["RingFingerMCPRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.RingFingerMCPRightAngle_y, self.prev_deg["RingFingerMCPRightAngle_y"])

            ############### RingFingerPIPRightAngle ###############
            self.calculate_angle(ring_finger_pip_right_xyz, ring_finger_dip_right_xyz, "RingFingerPIPRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.RingFingerPIPRightAngle_x, self.prev_deg["RingFingerPIPRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.RingFingerPIPRightAngle_y, self.prev_deg["RingFingerPIPRightAngle_y"])

            ############### RingFingerDIPRightAngle ###############
            self.calculate_angle(ring_finger_dip_right_xyz, ring_finger_tip_right_xyz, "RingFingerDIPRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.RingFingerDIPRightAngle_x, self.prev_deg["RingFingerDIPRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.RingFingerDIPRightAngle_y, self.prev_deg["RingFingerDIPRightAngle_y"])
            
            ############### WristPinkyRightAngle ###############
            self.calculate_angle(wrist_pinky_right_xyz, pinky_mcp_right_xyz, "WristPinkyRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.WristRightPinky_x, self.prev_deg["WristPinkyRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.WristRightPinky_y, self.prev_deg["WristPinkyRightAngle_y"])

            ############### PinkyFingerMCPRightAngle ###############
            self.calculate_angle(pinky_mcp_right_xyz, pinky_pip_right_xyz, "PinkyMCPRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.PinkyMCPRightAngle_x, self.prev_deg["PinkyMCPRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.PinkyMCPRightAngle_y, self.prev_deg["PinkyMCPRightAngle_y"])

            ############### PinkyFingerPIPRightAngle ###############
            self.calculate_angle(pinky_pip_right_xyz, pinky_dip_right_xyz, "PinkyPIPRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.PinkyPIPRightAngle_x, self.prev_deg["PinkyPIPRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.PinkyPIPRightAngle_y, self.prev_deg["PinkyPIPRightAngle_y"])

            ############### PinkyFingerDIPRightAngle ###############
            self.calculate_angle(pinky_dip_right_xyz, pinky_tip_right_xyz, "PinkyDIPRightAngle", smooth_deg=30)
            self.live_link_rhand.set_blendshape(BlendShape.PinkyDIPRightAngle_x, self.prev_deg["PinkyDIPRightAngle_x"])
            self.live_link_rhand.set_blendshape(BlendShape.PinkyDIPRightAngle_y, self.prev_deg["PinkyDIPRightAngle_y"])

        except Exception as e:
            print(e)