# import numpy as np
# import math
# import cv2


# def _get_position(result_pose_landmarks):

#     keypoints = []
#     _x = []
#     _y = []
#     _z = []
#     # _visibility = []

#     # keypoints = None

#     for _pose_landmarks in result_pose_landmarks:
#         keypoints.append({'x': _pose_landmarks.x,
#                           'y': _pose_landmarks.y,
#                           'z': _pose_landmarks.z,
#                         })
#     # print(keypoints[0]['x']) # Check-keypoint 'x', 'y', 'z', 'visibility'

#     for i in range(np.size(keypoints)):
#         _x.append(keypoints[i]['x'])
#         _y.append(keypoints[i]['y'])
#         _z.append(keypoints[i]['z'])


#     # landmarks_position = np.concatenate((_x,_y,_z,_visibility) , axis = 0).reshape((33,4))
#     landmarks_position = np.concatenate((_x,_y,_z) , axis = 0).reshape((33,3))
#     landmarks_position = landmarks_position.T
#     # print(landmarks['x'][i]) # index i 0-32 loop use 33
#     # print(landmarks)
#     # print(np.shape(keypoints))

#     # return landmarks_position , keypoints
#     return landmarks_position, keypoints


# def _calculate_angle(firstpoint, midpoint, endpoint):

#     firstpoint = np.array(firstpoint)
#     midpoint = np.array(midpoint)
#     endpoint = np.array(endpoint)

#     # use arctan
#     radians = np.arctan2(endpoint[1]-midpoint[1], endpoint[0]-midpoint[0]) - np.arctan2(firstpoint[1]-midpoint[1], firstpoint[0]-midpoint[0])
#     angle = np.absolute(radians*180.0/np.pi) # convert radiants to degrees

#     if angle > 180.0:
#         angle = 360 - angle

#     return angle

# def _calculate_circle_degrees(firstpoint, midpoint, endpoint):

#     firstpoint = np.array(firstpoint)
#     midpoint = np.array(midpoint)
#     endpoint = np.array(endpoint)

#     radians = np.arctan2(endpoint[1]-midpoint[1], endpoint[0]-midpoint[0]) - np.arctan2(firstpoint[1]-midpoint[1], firstpoint[0]-midpoint[0])
#     degrees = np.absolute(math.degrees(radians))

#     if degrees > 360.0:
#             degrees = 360 - degrees

#     return degrees



# def _puttext_onjoint(cv_image, calculate_angle_joint, where_joints):
#     cv2.putText(cv_image, str(calculate_angle_joint),
#                     tuple(np.multiply(where_joints, [640,480]).astype(int)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.5, (255,255,255), 1, cv2.LINE_AA
#                 )


#     # Test change Height Width camera
#     # cv2.putText(cv_image, str(calculate_angle_joint),
#     #     tuple(np.multiply(where_joints, [1280,720]).astype(int)),
#     #         cv2.FONT_HERSHEY_SIMPLEX, 
#     #         0.5, (255,255,255), 1, cv2.LINE_AA
#     #         )




















# # def _get_position(result_pose_landmarks):

# #     keypoints = []
# #     _x = []
# #     _y = []
# #     _z = []
# #     _visibility = []

# #     for _pose_landmarks in result_pose_landmarks:
# #         keypoints.append({'x': _pose_landmarks.x,
# #                           'y': _pose_landmarks.y,
# #                           'z': _pose_landmarks.z,
# #                           'visibility': _pose_landmarks.visibility,
# #                         })
# #     # print(keypoints[0]['x']) # Check-keypoint 'x', 'y', 'z', 'visibility'

# #     for i in range(np.size(keypoints)):
# #         _x.append(keypoints[i]['x'])
# #         _y.append(keypoints[i]['y'])
# #         _z.append(keypoints[i]['z'])
# #         _visibility.append(keypoints[i]['visibility'])

# #     landmarks_position = np.concatenate((_x,_y,_z,_visibility) , axis = 0).reshape((33,4))
# #     landmarks_position = landmarks_position.T
# #     # print(landmarks['x'][i]) # index i 0-32 loop use 33
# #     # print(landmarks)
# #     # print(np.shape(keypoints))

# #     return landmarks_position , keypoints






#     # def _calculate_angle(self, firstpoint, midpoint, endpoint):

#     #     firstpoint = np.array(firstpoint)
#     #     midpoint = np.array(midpoint)
#     #     endpoint = np.array(endpoint)

#     #     # use arctan
#     #     radians = np.arctan2(endpoint[1]-midpoint[1], endpoint[0]-midpoint[0]) - np.arctan2(firstpoint[1]-midpoint[1], firstpoint[0]-midpoint[0])
#     #     angle = np.absolute(radians*180.0/np.pi) # convert radiants to degrees

#     #     if angle > 180.0:
#     #         angle = 360 - angle

#     #     return angle

#     # def _calculate_circle_degrees(self, firstpoint, midpoint, endpoint):

#     #     firstpoint = np.array(firstpoint)
#     #     midpoint = np.array(midpoint)
#     #     endpoint = np.array(endpoint)

#     #     radians = np.arctan2(endpoint[1]-midpoint[1], endpoint[0]-midpoint[0]) - np.arctan2(firstpoint[1]-midpoint[1], firstpoint[0]-midpoint[0])
#     #     degrees = np.absolute(math.degrees(radians))

#     #     if degrees > 360.0:
#     #         degrees = 360 - degrees

#     #     return degrees



#     # def _puttext_onjoint(self, cv_image, calculate_angle_joint, where_joints):
#     #     cv2.putText(cv_image, str(calculate_angle_joint),
#     #                     tuple(np.multiply(where_joints, [640,480]).astype(int)),
#     #                     cv2.FONT_HERSHEY_SIMPLEX, 
#     #                     0.5, (255,255,255), 1, cv2.LINE_AA
#     #                 )


# # def azimuthAngle( x1, y1, x2, y2):
# #     angle = 0.0
# #     dx = x2 - x1
# #     dy = y2 - y1
# #     if x2 == x1:
# #         angle = math.pi / 2.0
# #         if y2 == y1 :
# #             angle = 0.0
# #         elif y2 < y1 :
# #             angle = 3.0 * math.pi / 2.0
# #     elif x2 > x1 and y2 > y1:
# #         angle = math.atan(dx / dy)
# #     elif x2 > x1 and y2 < y1 :
# #         angle = math.pi / 2 + math.atan(-dy / dx)
# #     elif x2 < x1 and y2 < y1 :
# #         angle = math.pi + math.atan(dx / dy)
# #     elif x2 < x1 and y2 > y1 :
# #         angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
# #     return (angle * 180 / math.pi)



# # def azimuthAngle(x, y):
# #     if x > 0:
# #         angle = np.arctan(y/x)
# #     elif x < 0 and y >= 0:
# #         angle = np.arctan(y/x)
# #     elif x < 0 and y < 0:
# #         angle = np.arctan(y/x)
# #     elif x == 0 and y > 0:
# #         angle = math.pi/2
# #     elif x == 0 and y < 0:
# #         angle = -math.pi/2
# #     else:
# #         print("undefined")
# #     return (angle * 180.0 / math.pi)