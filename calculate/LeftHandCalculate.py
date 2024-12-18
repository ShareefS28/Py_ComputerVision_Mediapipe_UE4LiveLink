import math
import numpy as np
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from livelink.pylivelink import PyLiveLink, BlendShape
from mediapipe.python.solutions import holistic
from typing import Optional

class LeftHandCalculate():

    def __init__(self) -> None:
        self.prev_deg = {}

    def _value_store(self, live_link_lhand: PyLiveLink, normalized_landmarks: RepeatedCompositeFieldContainer, image: Optional[np.ndarray] = None):
        self.live_link_lhand = live_link_lhand
        self.landmark_lhand = normalized_landmarks
        self.image = image
        self._calculate_lhand_landmarks()

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


    def _calculate_lhand_landmarks(self):
        try:
                # Get coordinates

            ###########################  THUMB FINGER ################################################
            wrist_thumb_left_xyz = [self.landmark_lhand[holistic.HandLandmark.WRIST.value].x, self.landmark_lhand[holistic.HandLandmark.WRIST.value].y, self.landmark_lhand[holistic.HandLandmark.WRIST.value].z]

            thumb_cmc_left_xyz = [self.landmark_lhand[holistic.HandLandmark.THUMB_CMC.value].x, self.landmark_lhand[holistic.HandLandmark.THUMB_CMC.value].y, self.landmark_lhand[holistic.HandLandmark.THUMB_CMC.value].z]

            thumb_mcp_left_xyz = [self.landmark_lhand[holistic.HandLandmark.THUMB_MCP.value].x, self.landmark_lhand[holistic.HandLandmark.THUMB_MCP.value].y, self.landmark_lhand[holistic.HandLandmark.THUMB_MCP.value].z]

            thumb_ip_left_xyz = [self.landmark_lhand[holistic.HandLandmark.THUMB_IP.value].x, self.landmark_lhand[holistic.HandLandmark.THUMB_IP.value].y, self.landmark_lhand[holistic.HandLandmark.THUMB_IP.value].z]

            thumb_tip_left_xyz = [self.landmark_lhand[holistic.HandLandmark.THUMB_TIP.value].x, self.landmark_lhand[holistic.HandLandmark.THUMB_TIP.value].y, self.landmark_lhand[holistic.HandLandmark.THUMB_TIP.value].z]

            ###########################  Index FINGER ##################################################
            wrist_index_left_xyz = [self.landmark_lhand[holistic.HandLandmark.WRIST.value].x, self.landmark_lhand[holistic.HandLandmark.WRIST.value].y, self.landmark_lhand[holistic.HandLandmark.WRIST.value].z]

            index_finger_mcp_left_xyz = [self.landmark_lhand[holistic.HandLandmark.INDEX_FINGER_MCP.value].x, self.landmark_lhand[holistic.HandLandmark.INDEX_FINGER_MCP.value].y, self.landmark_lhand[holistic.HandLandmark.INDEX_FINGER_MCP.value].z]

            index_finger_pip_left_xyz = [self.landmark_lhand[holistic.HandLandmark.INDEX_FINGER_PIP.value].x, self.landmark_lhand[holistic.HandLandmark.INDEX_FINGER_PIP.value].y, self.landmark_lhand[holistic.HandLandmark.INDEX_FINGER_PIP.value].z]

            index_finger_dip_left_xyz = [self.landmark_lhand[holistic.HandLandmark.INDEX_FINGER_DIP.value].x, self.landmark_lhand[holistic.HandLandmark.INDEX_FINGER_DIP.value].y, self.landmark_lhand[holistic.HandLandmark.INDEX_FINGER_DIP.value].z]

            index_finger_tip_left_xyz = [self.landmark_lhand[holistic.HandLandmark.INDEX_FINGER_TIP.value].x, self.landmark_lhand[holistic.HandLandmark.INDEX_FINGER_TIP.value].y, self.landmark_lhand[holistic.HandLandmark.INDEX_FINGER_TIP.value].z]

            ###########################  Middle FINGER ##################################################
            wrist_middle_left_xyz = [self.landmark_lhand[holistic.HandLandmark.WRIST.value].x, self.landmark_lhand[holistic.HandLandmark.WRIST.value].y, self.landmark_lhand[holistic.HandLandmark.WRIST.value].z]

            middle_finger_mcp_left_xyz = [self.landmark_lhand[holistic.HandLandmark.MIDDLE_FINGER_MCP.value].x, self.landmark_lhand[holistic.HandLandmark.MIDDLE_FINGER_MCP.value].y, self.landmark_lhand[holistic.HandLandmark.MIDDLE_FINGER_MCP.value].z]

            middle_finger_pip_left_xyz = [self.landmark_lhand[holistic.HandLandmark.MIDDLE_FINGER_PIP.value].x, self.landmark_lhand[holistic.HandLandmark.MIDDLE_FINGER_PIP.value].y, self.landmark_lhand[holistic.HandLandmark.MIDDLE_FINGER_PIP.value].z]

            middle_finger_dip_left_xyz = [self.landmark_lhand[holistic.HandLandmark.MIDDLE_FINGER_DIP.value].x, self.landmark_lhand[holistic.HandLandmark.MIDDLE_FINGER_DIP.value].y, self.landmark_lhand[holistic.HandLandmark.MIDDLE_FINGER_DIP.value].z]

            middle_finger_tip_left_xyz = [self.landmark_lhand[holistic.HandLandmark.MIDDLE_FINGER_TIP.value].x, self.landmark_lhand[holistic.HandLandmark.MIDDLE_FINGER_TIP.value].y, self.landmark_lhand[holistic.HandLandmark.MIDDLE_FINGER_TIP.value].z]

            ###########################  Ring FINGER ##################################################
            wrist_ring_left_xyz = [self.landmark_lhand[holistic.HandLandmark.WRIST.value].x, self.landmark_lhand[holistic.HandLandmark.WRIST.value].y, self.landmark_lhand[holistic.HandLandmark.WRIST.value].z]

            ring_finger_mcp_left_xyz = [self.landmark_lhand[holistic.HandLandmark.RING_FINGER_MCP.value].x, self.landmark_lhand[holistic.HandLandmark.RING_FINGER_MCP.value].y, self.landmark_lhand[holistic.HandLandmark.RING_FINGER_MCP.value].z]

            ring_finger_pip_left_xyz = [self.landmark_lhand[holistic.HandLandmark.RING_FINGER_PIP.value].x, self.landmark_lhand[holistic.HandLandmark.RING_FINGER_PIP.value].y, self.landmark_lhand[holistic.HandLandmark.RING_FINGER_PIP.value].z]

            ring_finger_dip_left_xyz = [self.landmark_lhand[holistic.HandLandmark.RING_FINGER_DIP.value].x, self.landmark_lhand[holistic.HandLandmark.RING_FINGER_DIP.value].y, self.landmark_lhand[holistic.HandLandmark.RING_FINGER_DIP.value].z]

            ring_finger_tip_left_xyz = [self.landmark_lhand[holistic.HandLandmark.RING_FINGER_TIP.value].x, self.landmark_lhand[holistic.HandLandmark.RING_FINGER_TIP.value].y, self.landmark_lhand[holistic.HandLandmark.RING_FINGER_TIP.value].z]

            ###########################  Pinky FINGER ##################################################
            wrist_pinky_left_xyz = [self.landmark_lhand[holistic.HandLandmark.WRIST.value].x, self.landmark_lhand[holistic.HandLandmark.WRIST.value].y, self.landmark_lhand[holistic.HandLandmark.WRIST.value].z]

            pinky_mcp_left_xyz = [self.landmark_lhand[holistic.HandLandmark.PINKY_MCP.value].x, self.landmark_lhand[holistic.HandLandmark.PINKY_MCP.value].y, self.landmark_lhand[holistic.HandLandmark.PINKY_MCP.value].z]

            pinky_pip_left_xyz = [self.landmark_lhand[holistic.HandLandmark.PINKY_PIP.value].x, self.landmark_lhand[holistic.HandLandmark.PINKY_PIP.value].y, self.landmark_lhand[holistic.HandLandmark.PINKY_PIP.value].z]

            pinky_dip_left_xyz = [self.landmark_lhand[holistic.HandLandmark.PINKY_DIP.value].x, self.landmark_lhand[holistic.HandLandmark.PINKY_DIP.value].y, self.landmark_lhand[holistic.HandLandmark.PINKY_DIP.value].z]

            pinky_tip_left_xyz = [self.landmark_lhand[holistic.HandLandmark.PINKY_TIP.value].x, self.landmark_lhand[holistic.HandLandmark.PINKY_TIP.value].y, self.landmark_lhand[holistic.HandLandmark.PINKY_TIP.value].z]

            ############### WristThumbLeftAngle ###############
            self.calculate_angle(wrist_thumb_left_xyz, thumb_cmc_left_xyz, "WristThumbLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.WristLeftThumb_x, self.prev_deg["WristThumbLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.WristLeftThumb_y, self.prev_deg["WristThumbLeftAngle_y"])

            ############### ThumbCMCLeftAngle ###############
            self.calculate_angle(thumb_cmc_left_xyz, thumb_mcp_left_xyz, "ThumbCMCLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.ThumbCMCLeftAngle_x, self.prev_deg["ThumbCMCLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.ThumbCMCLeftAngle_y, self.prev_deg["ThumbCMCLeftAngle_y"])

            ############### ThumbMCPLeftAngle ###############
            self.calculate_angle(thumb_mcp_left_xyz, thumb_ip_left_xyz, "ThumbMCPLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.ThumbMCPLeftAngle_x, self.prev_deg["ThumbMCPLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.ThumbMCPLeftAngle_y, self.prev_deg["ThumbMCPLeftAngle_y"])

            ############### ThumbIPLeftAngle ###############
            self.calculate_angle(thumb_ip_left_xyz, thumb_tip_left_xyz, "ThumbIPLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.ThumbIPLeftAngle_x, self.prev_deg["ThumbIPLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.ThumbIPLeftAngle_y, self.prev_deg["ThumbIPLeftAngle_y"])
            
            ############### WristIndexLeftAngle ###############
            self.calculate_angle(wrist_index_left_xyz, index_finger_mcp_left_xyz, "WristIndexLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.WristLeftIndex_x, self.prev_deg["WristIndexLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.WristLeftIndex_y, self.prev_deg["WristIndexLeftAngle_y"])

            ############### IndexFingerMCPLeftAngle ###############
            self.calculate_angle(index_finger_mcp_left_xyz, index_finger_pip_left_xyz, "IndexFingerMCPLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.IndexFingerMCPLeftAngle_x, self.prev_deg["IndexFingerMCPLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.IndexFingerMCPLeftAngle_y, self.prev_deg["IndexFingerMCPLeftAngle_y"])

            ############### IndexFingerPIPLeftAngle ###############
            self.calculate_angle(index_finger_pip_left_xyz, index_finger_dip_left_xyz, "IndexFingerPIPLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.IndexFingerPIPLeftAngle_x, self.prev_deg["IndexFingerPIPLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.IndexFingerPIPLeftAngle_y, self.prev_deg["IndexFingerPIPLeftAngle_y"])

            ############### IndexFingerDIPLeftAngle ###############
            self.calculate_angle(index_finger_dip_left_xyz, index_finger_tip_left_xyz, "IndexFingerDIPLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.IndexFingerDIPLeftAngle_x, self.prev_deg["IndexFingerDIPLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.IndexFingerDIPLeftAngle_y, self.prev_deg["IndexFingerDIPLeftAngle_y"])

            ############### WristMiddleLeftAngle ###############
            self.calculate_angle(wrist_middle_left_xyz, middle_finger_mcp_left_xyz, "WristMiddleLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.WristLeftMiddle_x, self.prev_deg["WristMiddleLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.WristLeftMiddle_y, self.prev_deg["WristMiddleLeftAngle_y"])

            ############### MiddleFingerMCPLeftAngle ###############
            self.calculate_angle(middle_finger_mcp_left_xyz, middle_finger_pip_left_xyz, "MiddleFingerMCPLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.MiddleFingerMCPLeftAngle_x, self.prev_deg["MiddleFingerMCPLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.MiddleFingerMCPLeftAngle_y, self.prev_deg["MiddleFingerMCPLeftAngle_y"])

            ############### MiddleFingerPIPLeftAngle ###############
            self.calculate_angle(middle_finger_pip_left_xyz, middle_finger_dip_left_xyz, "MiddleFingerPIPLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.MiddleFingerPIPLeftAngle_x, self.prev_deg["MiddleFingerPIPLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.MiddleFingerPIPLeftAngle_y, self.prev_deg["MiddleFingerPIPLeftAngle_y"])

            ############### MiddleFingerDIPLeftAngle ###############
            self.calculate_angle(middle_finger_dip_left_xyz, middle_finger_tip_left_xyz, "MiddleFingerDIPLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.MiddleFingerDIPLeftAngle_x, self.prev_deg["MiddleFingerDIPLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.MiddleFingerDIPLeftAngle_y, self.prev_deg["MiddleFingerDIPLeftAngle_y"])

            ############### WristRingLeftAngle ###############
            self.calculate_angle(wrist_ring_left_xyz, ring_finger_mcp_left_xyz, "WristRingLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.WristLeftRing_x, self.prev_deg["WristRingLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.WristLeftRing_y, self.prev_deg["WristRingLeftAngle_y"])

            ############### RingFingerMCPLeftAngle ###############
            self.calculate_angle(ring_finger_mcp_left_xyz, ring_finger_pip_left_xyz, "RingFingerMCPLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.RingFingerMCPLeftAngle_x, self.prev_deg["RingFingerMCPLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.RingFingerMCPLeftAngle_y, self.prev_deg["RingFingerMCPLeftAngle_y"])

            ############### RingFingerPIPLeftAngle ###############
            self.calculate_angle(ring_finger_pip_left_xyz, ring_finger_dip_left_xyz, "RingFingerPIPLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.RingFingerPIPLeftAngle_x, self.prev_deg["RingFingerPIPLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.RingFingerPIPLeftAngle_y, self.prev_deg["RingFingerPIPLeftAngle_y"])

            ############### RingFingerDIPLeftAngle ###############
            self.calculate_angle(ring_finger_dip_left_xyz, ring_finger_tip_left_xyz, "RingFingerDIPLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.RingFingerDIPLeftAngle_x, self.prev_deg["RingFingerDIPLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.RingFingerDIPLeftAngle_y, self.prev_deg["RingFingerDIPLeftAngle_y"])
            
            ############### WristPinkyLeftAngle ###############
            self.calculate_angle(wrist_pinky_left_xyz, pinky_mcp_left_xyz, "WristPinkyLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.WristLeftPinky_x, self.prev_deg["WristPinkyLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.WristLeftPinky_y, self.prev_deg["WristPinkyLeftAngle_y"])

            ############### PinkyFingerMCPLeftAngle ###############
            self.calculate_angle(pinky_mcp_left_xyz, pinky_pip_left_xyz, "PinkyMCPLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.PinkyMCPLeftAngle_x, self.prev_deg["PinkyMCPLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.PinkyMCPLeftAngle_y, self.prev_deg["PinkyMCPLeftAngle_y"])

            ############### PinkyFingerPIPLeftAngle ###############
            self.calculate_angle(pinky_pip_left_xyz, pinky_dip_left_xyz, "PinkyPIPLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.PinkyPIPLeftAngle_x, self.prev_deg["PinkyPIPLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.PinkyPIPLeftAngle_y, self.prev_deg["PinkyPIPLeftAngle_y"])

            ############### PinkyFingerDIPLeftAngle ###############
            self.calculate_angle(pinky_dip_left_xyz, pinky_tip_left_xyz, "PinkyDIPLeftAngle", smooth_deg=30)
            self.live_link_lhand.set_blendshape(BlendShape.PinkyDIPLeftAngle_x, self.prev_deg["PinkyDIPLeftAngle_x"])
            self.live_link_lhand.set_blendshape(BlendShape.PinkyDIPLeftAngle_y, self.prev_deg["PinkyDIPLeftAngle_y"])

        except Exception as e:
            print(e)