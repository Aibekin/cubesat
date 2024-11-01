# import cv2
# import math
# import numpy as np
# from scipy.spatial.transform import Rotation as R

# # def aruco_angle(rvecs, tvecs):
# #     rvec_matrix, _ = cv2.Rodrigues(rvecs[0])

# #     proj_matrix = np.hstack((rvec_matrix, tvecs[0]))
# #     euler_angle = cv2.decomposeProjectionMatrix(proj_matrix)[6] # [deg]
# #     # print(tvecs)
# #     # print("tvecs[0]"+str(tvecs[0])) 
# #     # print("tvecs[1][0]"+str(tvecs[1][0]))
# #     # print("tvecs[0][1]"+str(tvecs[0][1]))

# #     if len(tvecs) > 0:
# #         print("x : " + str(tvecs[0][0]))
# #         print("y : " + str(tvecs[0][1]))
# #         print("z : " + str(tvecs[0][2]))
# #         print("roll : " + str(euler_angle[0]))
# #         print("pitch: " + str(euler_angle[1]))
# #         print("yaw  : " + str(euler_angle[2]))


# # def calculate_angle(center_x, center_y, center_z, point_x, point_y, point_z):
# #     # Вычисляем вектор от центра к точке
# #     delta_x = point_x - center_x
# #     delta_y = point_y - center_y
# #     delta_z = point_z - center_z
# #     # Вычисляем угол в радианах
# #     roll_angle_rad = math.atan2(delta_y, delta_x)
# #     roll_angle_deg = math.degrees(roll_angle_rad)
    
# #     # Коррекция угла поворота относительно смещения (yaw offset)
# #     roll_angle_deg = (roll_angle_deg-111) % 360
# #     if roll_angle_deg > 180:
# #         roll_angle_deg = 360 - roll_angle_deg
    
# #     # Угол наклона (pitch) относительно оси Y
# #     pitch_angle_rad = math.atan2(delta_z, math.sqrt(delta_x**2 + delta_y**2))
# #     pitch_angle_deg = math.degrees(pitch_angle_rad)
# #     pitch_angle_deg = (pitch_angle_deg + 70) 
    
# #     # Угол рыскания (yaw) относительно оси XZ
# #     yaw_angle_rad = math.atan2(delta_z, delta_x)
# #     yaw_angle_deg = math.degrees(yaw_angle_rad)
# #     # yaw_angle_deg = (yaw_angle_deg + 117) % 360
# #     if yaw_angle_deg > 180:
# #         yaw_angle_deg -= 360
# #     elif yaw_angle_deg < -180:
# #         yaw_angle_deg += 360
# #     return roll_angle_deg, pitch_angle_deg, yaw_angle_deg

# # def estimate_center(ccx, ccy):
# #     top_left_x = ccx - 5
# #     top_left_y = ccy - 5
# #     top_right_x = ccx + 5
# #     top_right_y = ccy - 5 
# #     bottom_left_x = ccx - 5
# #     bottom_left_y = ccy + 5
# #     bottom_right_x = ccx + 5
# #     bottom_right_y = ccy + 5
# #     corners_my = np.array([
# #         [[top_left_x, top_left_y],    # Верхний левый угол
# #         [top_right_x, top_right_y],  # Верхний правый угол
# #         [bottom_right_x, bottom_right_y], # Нижний правый угол
# #         [bottom_left_x, bottom_left_y]]    # Нижний левый угол
# #     ], dtype=np.float32)
# #     return corners_my
# def find_center(corners):
#     [x1, y1] = corners[0][0][0]
#     [x2, y2] = corners[0][0][1]
#     [x3, y3] = corners[0][0][2]
#     [x4, y4] = corners[0][0][3]
    
#     x_center = (x1 + x2 + x3 + x4) / 4
#     y_center = (y1 + y2 + y3 + y4) / 4
#     return (x_center, y_center)

# def draw_axis(yaw, pitch, roll):

#     pitch = pitch * np.pi / 180
#     yaw = -(yaw * np.pi / 180)
#     roll = roll * np.pi / 180
    
#     print("roll : " + str(roll))
#     print("pitch: " + str(pitch))
#     print("yaw  : " + str(yaw))
   
# import joblib
# import time

# def draw_cube(image, front_points, back_points, color=(255, 0, 0), thickness=2):

#     for i in range(4):
#         cv2.line(image, tuple(front_points[i]), tuple(front_points[(i + 1) % 4]), color, thickness)
    
#     for i in range(4):
#         cv2.line(image, tuple(back_points[i]), tuple(back_points[(i + 1) % 4]), color, thickness)
#         cv2.circle(image, tuple(back_points[(i + 1) % 4]), 5, color, -1)

#     for i in range(4):
#         cv2.line(image, tuple(front_points[i]), tuple(back_points[i]), color, thickness)
#         cv2.circle(image, tuple(back_points[i]), 5, color, -1)

# def calc_ang(tvecs1, rvecs1):
#     rvec_matrix, _ = cv2.Rodrigues(rvecs1[0])
#     proj_matrix = np.hstack((rvec_matrix, tvecs1[0]))
#     euler_angle = cv2.decomposeProjectionMatrix(proj_matrix)[6] # [deg]
    
    
    
#     if len(tvecs1) > 0:
#         # print("x : " + str(tvecs1[0][0]))
#         # print("y : " + str(tvecs1[0][1]))
#         # print("z : " + str(tvecs1[0][2]))
#         # print("roll : " + str(euler_angle[0]))
#         # print("pitch: " + str(euler_angle[1]))
#         # print("yaw  : " + str(euler_angle[2]))
#         return euler_angle[0], euler_angle[1]

# def calc_angle_rotation(tvecs1, rvecs1):
#     transform_translation_x = tvecs1[0][0]
#     transform_translation_y = tvecs1[0][1]
#     transform_translation_z = tvecs1[0][2]

#     rotation_matrix = np.eye(4)
#     rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs1[0]))[0]
#     r = R.from_matrix(rotation_matrix[0:3, 0:3])
#     quat = r.as_quat()   
        
#     transform_rotation_x = quat[0] 
#     transform_rotation_y = quat[1] 
#     transform_rotation_z = quat[2] 
#     transform_rotation_w = quat[3] 
    
#     roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x, 
#                                                 transform_rotation_y, 
#                                                 transform_rotation_z, 
#                                                 transform_rotation_w)
    
#     roll_x = math.degrees(roll_x)
#     pitch_y = math.degrees(pitch_y)
#     yaw_z = math.degrees(yaw_z)
#     # roll_x = (roll_x + 360) % 360
#     # pitch_y = (pitch_y + 180) % 180
#     # yaw_z = (yaw_z + 90) % 90
#     # print("transform_translation_x: {}".format(transform_translation_x))
#     # print("transform_translation_y: {}".format(transform_translation_y))
#     # print("transform_translation_z: {}".format(transform_translation_z))
#     print(str(roll_x))
#     # print("pitch_y: {}".format(pitch_y))
#     # print("yaw_z: {}".format(yaw_z))

# def euler_from_quaternion(x, y, z, w):
#   t0 = +2.0 * (w * x + y * z)
#   t1 = +1.0 - 2.0 * (x * x + y * y)
#   roll_x = math.atan2(t0, t1)
      
#   t2 = +2.0 * (w * y - z * x)
#   t2 = +1.0 if t2 > +1.0 else t2
#   t2 = -1.0 if t2 < -1.0 else t2
#   pitch_y = math.asin(t2)
      
#   t3 = +2.0 * (w * z + x * y)
#   t4 = +1.0 - 2.0 * (y * y + z * z)
#   yaw_z = math.atan2(t3, t4)
      
#   return roll_x, pitch_y, yaw_z 


# def give_points(corners, index_id_1, index_id_2, index_id_3, index_id_4):

    
#     ccx_1 = corners[index_id_1][0][1]
    
#     ccx_2 = corners[index_id_2][0][0]
    
#     ccx_3 = corners[index_id_3][0][2]
   
#     ccx_4 = corners[index_id_4][0][3]
    
#     point = np.array([
#         [[ccx_2[0], ccx_2[1]],   
#         [ccx_1[0], ccx_1[1]], 
#         [ccx_3[0], ccx_3[1]], 
#         [ccx_4[0], ccx_4[1]]]
#     ], dtype=np.float32)
#     return point


# def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
#     marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
#                               [marker_size / 2, marker_size / 2, 0],
#                               [marker_size / 2, -marker_size / 2, 0],
#                               [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
#     trash = []
#     rvecs = []
#     tvecs = []
#     for c in corners:
#         nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
#         rvecs.append(R)
#         tvecs.append(t)
#         trash.append(nada)
#     return rvecs, tvecs, trash

