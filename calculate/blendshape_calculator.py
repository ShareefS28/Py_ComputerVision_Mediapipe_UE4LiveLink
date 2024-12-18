import math
import numpy as np
from livelink.pylivelink import PyLiveLink, BlendShape
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from customs.blendshape_config import BlendShapeConfig

class BlendshapeCalculator():
    """ BlendshapeCalculator class
    
    This class calculates the blendshapes from the given landmarks.
    """

    def __init__(self) -> None:
        self.blend_shape_config = BlendShapeConfig()        
        
    def calculate_blendshapes(self, live_link_face: PyLiveLink, metric_landmarks: np.ndarray, normalized_landmarks: RepeatedCompositeFieldContainer) -> None:
        """ Calculate the blendshapes from the given landmarks. 
        
        This function calculates the blendshapes from the given landmarks and stores them in the given live_link_face.
        
        Parameters
        ----------
        live_link_face : PyLiveLinkFace
            Index of the BlendShape to get the value from.
        metric_landmarks: np.ndarray
            The metric landmarks of the face in 3d.
        normalized_landmarks: RepeatedCompositeFieldContainer
            Output from the mediapipe process function for each face.
        
        Returns
        ----------
        None
        """
        self._live_link_face = live_link_face
        self._metric_landmarks = metric_landmarks
        self._normalized_landmarks = normalized_landmarks

        self._calculate_mouth_landmarks()
        self._calculate_eye_landmarks()
        
    def _get_landmark(self, index: int, use_normalized: bool = False) -> np.array:   
        """ Get the stored landmark from the given index.
        
        This function converts both the metric and normalized landmarks to a numpy array.

        Parameters
        ----------
        index : int
            Index of the point to get the landmark from.
        use_normalized: bool
            If true, the normalized landmarks are used. Otherwise the metric landmarks are used.
        
        Returns
        ----------
        np.array
            The landmark in a 3d numpy array.   
        """

        landmarks = self._metric_landmarks
        if use_normalized:
            landmarks = self._normalized_landmarks

        if type(landmarks) == np.ndarray:
            # is a 3d landmark
            x = landmarks[index][0]
            y = landmarks[index][1]             
            z = landmarks[index][2] 
            return np.array([x, y, z])
        else:           
            # is a normalized landmark
            x = landmarks[index].x #* self.image_width
            y = landmarks[index].y #* self.image_height
            z = landmarks[index].z #* self.image_height
            return np.array([x, y, z])      

    #  clamp value to 0 - 1 using the min and max values of the config
    def _remap(self, value, min, max):
        return (np.clip(value, min, max) - min) / (max - min)

    def _remap_blendshape(self, index: BlendShape, value: float):
        min, max = self.blend_shape_config.config.get(index)        
        return self._remap(value, min, max)  

    def _calculate_mouth_landmarks(self):
        upper_lip = self._get_landmark(self.blend_shape_config.CanonicalPpoints.upper_lip)
        lower_lip = self._get_landmark(self.blend_shape_config.CanonicalPpoints.lower_lip)

        mouth_corner_left = self._get_landmark(self.blend_shape_config.CanonicalPpoints.mouth_corner_left)
        mouth_corner_right = self._get_landmark(self.blend_shape_config.CanonicalPpoints.mouth_corner_right)
        lowest_chin = self._get_landmark(self.blend_shape_config.CanonicalPpoints.lowest_chin)
        nose_tip = self._get_landmark(self.blend_shape_config.CanonicalPpoints.nose_tip)
        upper_head = self._get_landmark(self.blend_shape_config.CanonicalPpoints.upper_head)

        mouth_width = math.dist(mouth_corner_left, mouth_corner_right)
        mouth_center = (upper_lip + lower_lip) / 2
        mouth_open_dist = math.dist(upper_lip, lower_lip)
        mouth_center_nose_dist = math.dist(mouth_center, nose_tip)

        jaw_nose_dist = math.dist(lowest_chin, nose_tip)
        head_height = math.dist(upper_head, lowest_chin)
        jaw_open_ratio = jaw_nose_dist / head_height

        jaw_open = self._remap_blendshape(
            BlendShape.JawOpen, jaw_open_ratio)
        self._live_link_face.set_blendshape(BlendShape.JawOpen, jaw_open)
    
        mouth_open = self._remap_blendshape(
            BlendShape.MouthClose, mouth_center_nose_dist - mouth_open_dist)
        self._live_link_face.set_blendshape(
            BlendShape.MouthClose, mouth_open)

        # TODO mouth open but teeth closed
        smile_left = upper_lip[1] - mouth_corner_left[1]
        smile_right = upper_lip[1] - mouth_corner_right[1]

        mouth_smile_left = 1 - \
            self._remap_blendshape(BlendShape.MouthSmileLeft, smile_left)

        self._live_link_face.set_blendshape(
            BlendShape.MouthSmileLeft, mouth_smile_left)

        # jaw only interesting on x yxis
        jaw_right_left = nose_tip[0] - lowest_chin[0]

        # TODO: this is not face rotation resistant
        self._live_link_face.set_blendshape(BlendShape.JawRight, self._remap_blendshape(
            BlendShape.JawRight, jaw_right_left))

        mouth_pucker = self._remap_blendshape(
            BlendShape.MouthPucker, mouth_width)
        self._live_link_face.set_blendshape(
            BlendShape.MouthPucker, 1 - mouth_pucker)

        # really hard to do this, mediapipe is not really moving here
        # right_under_eye = self._get_landmark(350)
        # nose_sneer_right_dist = math.dist(nose_tip, right_under_eye)
        # print(nose_sneer_right_dist)
        # same with cheek puff

    def _eye_lid_distance(self, eye_points):
        eye_width = math.dist(self._get_landmark(
            eye_points[0]), self._get_landmark(eye_points[1]))
        eye_outer_lid = math.dist(self._get_landmark(
            eye_points[2]), self._get_landmark(eye_points[5]))
        eye_mid_lid = math.dist(self._get_landmark(
            eye_points[3]), self._get_landmark(eye_points[6]))
        eye_inner_lid = math.dist(self._get_landmark(
            eye_points[4]), self._get_landmark(eye_points[7]))
        eye_lid_avg = (eye_outer_lid + eye_mid_lid + eye_inner_lid) / 3
        ratio = eye_lid_avg / eye_width
        return ratio

    def _calculate_eye_landmarks(self):
        # Adapted from Kalidokit, https://github.com/yeemachine/kalidokit/blob/main/src/FaceSolver/calcEyes.ts
        def get_eye_open_ration(points):
            eye_distance = self._eye_lid_distance(points)
            max_ratio = 0.285
            ratio = np.clip(eye_distance / max_ratio, 0, 2)
            return ratio

        eye_open_ratio_left = get_eye_open_ration(self.blend_shape_config.CanonicalPpoints.eye_left)
        eye_open_ratio_right = get_eye_open_ration(self.blend_shape_config.CanonicalPpoints.eye_right)
    
        blink_left = 1 - \
            self._remap_blendshape(
                BlendShape.EyeBlinkLeft, eye_open_ratio_left)
        blink_right = 1 - \
            self._remap_blendshape(
                BlendShape.EyeBlinkRight, eye_open_ratio_right)

        self._live_link_face.set_blendshape(
            BlendShape.EyeBlinkLeft, blink_left, True)
        self._live_link_face.set_blendshape(
            BlendShape.EyeBlinkRight, blink_right, True)

        self._live_link_face.set_blendshape(BlendShape.EyeWideLeft, self._remap_blendshape(
            BlendShape.EyeWideLeft, eye_open_ratio_left))
        self._live_link_face.set_blendshape(BlendShape.EyeWideRight, self._remap_blendshape(
            BlendShape.EyeWideRight, eye_open_ratio_right))

        squint_left = math.dist(
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.squint_left[0]), 
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.squint_left[1])
        )
        self._live_link_face.set_blendshape(
            BlendShape.EyeSquintLeft, 1 - self._remap_blendshape(BlendShape.EyeSquintLeft, squint_left))

        right_brow_lower = (
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.right_brow_lower[0]) +
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.right_brow_lower[1]) +
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.right_brow_lower[2])
        ) / 3
        right_brow_dist = math.dist(self._get_landmark(self.blend_shape_config.CanonicalPpoints.right_brow), right_brow_lower)

        left_brow_lower = (
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.left_brow_lower[0]) +
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.left_brow_lower[1]) +
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.left_brow_lower[2])
        ) / 3
        left_brow_dist = math.dist(self._get_landmark(self.blend_shape_config.CanonicalPpoints.left_brow), left_brow_lower)

        self._live_link_face.set_blendshape(
            BlendShape.BrowDownLeft, 1 - self._remap_blendshape(BlendShape.BrowDownLeft, left_brow_dist))
        
        self._live_link_face.set_blendshape(
            BlendShape.BrowDownRight, 1 - self._remap_blendshape(BlendShape.BrowDownRight, right_brow_dist))

        inner_brow = self._get_landmark(self.blend_shape_config.CanonicalPpoints.inner_brow)
        upper_nose = self._get_landmark(self.blend_shape_config.CanonicalPpoints.upper_nose)
        inner_brow_dist = math.dist(upper_nose, inner_brow)

        self._live_link_face.set_blendshape(BlendShape.BrowInnerUp, self._remap_blendshape(
            BlendShape.BrowInnerUp, inner_brow_dist))

        cheek_squint_left = math.dist(
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.cheek_squint_left[0]), 
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.cheek_squint_left[1])
        )

        cheek_squint_right = math.dist(
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.cheek_squint_right[0]), 
            self._get_landmark(self.blend_shape_config.CanonicalPpoints.cheek_squint_right[1])
        )

        self._live_link_face.set_blendshape(
            BlendShape.CheekSquintLeft, 1 - self._remap_blendshape(BlendShape.CheekSquintLeft, cheek_squint_left))
        self._live_link_face.set_blendshape(
            BlendShape.CheekSquintRight, 1 - self._remap_blendshape(BlendShape.CheekSquintRight, cheek_squint_right))

        # just use the same values for cheeksquint for nose sneer, mediapipe deosn't seem to have a separate value for nose sneer
        self._live_link_face.set_blendshape(
            BlendShape.NoseSneerLeft, self._live_link_face.get_blendshape(BlendShape.CheekSquintLeft))
        self._live_link_face.set_blendshape(
            BlendShape.NoseSneerRight, self._live_link_face.get_blendshape(BlendShape.CheekSquintRight))