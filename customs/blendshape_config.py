from livelink.pylivelink import BlendShape


class BlendShapeConfig:
        class CanonicalPpoints:

            # canoncial points mapped from the canoncial face model        
            # for better understanding of the points, see the canonical face model from mediapipe
            # https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

            eye_right = [33, 133, 160, 159, 158, 144, 145, 153]
            eye_left = [263, 362, 387, 386, 385, 373, 374, 380]

            #Experiment

            # eye_right = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
            # eye_left = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]

            # Left_iris = [473, 474, 475, 476, 477]
            #473-478
            # Right_iris = [468 ,469, 470, 471, 472]
            #468-472

            # Left_iris = 468
            # Right_iris = 473
            
            #Experiment

            head = [10, 152]
            nose_tip = 1
            upper_lip = 13
            lower_lip = 14
            upper_outer_lip = 12
            mouth_corner_left = 291
            mouth_corner_right = 61
            lowest_chin = 152
            upper_head = 10
            mouth_frown_left = 422
            mouth_frown_right = 202
            mouth_left_stretch = 287
            mouth_right_stretch = 57
            lowest_lip = 17
            under_lip = 18
            over_upper_lip = 164
            left_upper_press = [40, 80]
            left_lower_press = [88, 91]
            right_upper_press = [270, 310]
            right_lower_press = [318, 321]
            squint_left = [253, 450]
            squint_right = [23, 230]            
            right_brow = 27
            right_brow_lower = [53, 52, 65]
            left_brow = 257
            left_brow_lower = [283, 282, 295]
            inner_brow = 9
            upper_nose = 6
            cheek_squint_left = [359, 342]
            cheek_squint_right = [130, 113]

        # blend shape type, min and max value
        config = {
            BlendShape.EyeBlinkLeft : (0.50, 0.90), #
            # BlendShape.EyeLookDownLeft : (-0.4, 0.0), #
            # BlendShape.EyeLookInLeft : (-0.4, 0.0),
            # BlendShape.EyeLookOutLeft : (-0.4, 0.0),
            # BlendShape.EyeLookUpLeft : (-0.4, 0.0), # 
            # BlendShape.JawOpen : (0.53, 0.58), #            ## Found some ploblem when you open and roll your head value is not stable ## !! issue is angle of camera
            # BlendShape.JawOpen : (0.50, 0.55),            # depend on camera angle # work on sit
            # BlendShape.JawOpen : (0.45, 0.48),            # work on walk and sit from 09/14/2022 in my discord [angle position]
            BlendShape.JawOpen : (0.48, 0.50),              # 09/14/2022 17.16 now is working
            BlendShape.EyeSquintLeft : (0.37, 0.44),
            BlendShape.EyeWideLeft : (0.9, 1.2), #
            BlendShape.EyeBlinkRight : (0.50, 0.90), #
            # BlendShape.EyeLookDownRight : (-0.4, 0.0),
            # BlendShape.EyeLookInRight : (-0.4, 0.0), #
            # BlendShape.EyeLookOutRight : (-0.4, 0.0), #
            # BlendShape.EyeLookUpRight : (-0.4, 0.0),
            # BlendShape.EyeSquintRight : (0.37, 0.44),
            BlendShape.EyeWideRight : (0.9, 1.2), #
            # BlendShape.JawForward : (-0.4, 0.0),
            # BlendShape.JawLeft : (-0.4, 0.0),
            BlendShape.JawRight : (0.0, 0.4),
            # BlendShape.JawOpen : (0.50, 0.55), # default value JawOpen
            BlendShape.MouthClose : (1.5, 2.85),
            # BlendShape.MouthFunnel : (4.0, 4.8),
            BlendShape.MouthPucker : (3.46, 4.92), #
            # BlendShape.MouthLeft : (-3.4, -2.3),
            # BlendShape.MouthRight : ( 1.5, 3.0),
            BlendShape.MouthSmileLeft : (-0.25, 0.0), #
            # BlendShape.MouthSmileRight : (-0.25, 0.0), #
            # BlendShape.MouthFrownLeft : (0.4, 0.9),
            # BlendShape.MouthFrownRight : (0.4, 0.9),
            # BlendShape.MouthDimpleLeft : (-0.4, 0.0),
            # BlendShape.MouthDimpleRight : (-0.4, 0.0),
            # BlendShape.MouthStretchLeft : (-0.4, 0.0),
            # BlendShape.MouthStretchRight : (-0.4, 0.0),
            # BlendShape.MouthRollLower : (0.4, 0.7),
            # BlendShape.MouthRollUpper : (0.31, 0.34),
            # BlendShape.MouthShrugLower : (1.9, 2.3),
            # BlendShape.MouthShrugUpper : (1.4, 2.4),
            # BlendShape.MouthPressLeft : (0.4, 0.5),
            # BlendShape.MouthPressRight : (0.4, 0.5),
            # BlendShape.MouthLowerDownLeft : (1.7, 2.1),
            # BlendShape.MouthLowerDownRight : (1.7, 2.1),
            # BlendShape.MouthUpperUpLeft : (-0.4, 0.0),
            # BlendShape.MouthUpperUpRight : (-0.4, 0.0),
            BlendShape.BrowDownLeft : (1.0, 1.2), #
            BlendShape.BrowDownRight : (1.0, 1.2), #
            BlendShape.BrowInnerUp : (2.2, 2.6), #
            # BlendShape.BrowOuterUpLeft : (1.25, 1.5),
            # BlendShape.BrowOuterUpRight : (1.25, 1.5),
            # BlendShape.CheekPuff : (-0.4, 0.0),
            BlendShape.CheekSquintLeft : (0.55, 0.63),
            BlendShape.CheekSquintRight : (0.55, 0.63),
            # BlendShape.NoseSneerLeft : (-0.4, 0.0),
            # BlendShape.NoseSneerRight : (-0.4, 0.0),
            # BlendShape.TongueOut : (-0.4, 0.0),
            # BlendShape.HeadYaw : (-0.4, 0.0),
            # BlendShape.HeadPitch : (-0.4, 0.0),
            # BlendShape.HeadRoll : (-0.4, 0.0),
            # BlendShape.LeftEyeYaw : (-0.4, 0.0),
            # BlendShape.LeftEyePitch : (-0.4, 0.0),
            # BlendShape.LeftEyeRoll : (-0.4, 0.0),
            # BlendShape.RightEyeYaw : (-0.4, 0.0),
            # BlendShape.RightEyePitch : (-0.4, 0.0),
            # BlendShape.RightEyeRoll : (-0.4, 0.0), 
        }

       
