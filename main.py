
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import random
import funcs


def calc_ang(tvecs1, rvecs1):
    rvec_matrix, _ = cv2.Rodrigues(rvecs1[0])
    proj_matrix = np.hstack((rvec_matrix, tvecs1[0]))
    euler_angle = cv2.decomposeProjectionMatrix(proj_matrix)[6] # [deg]
    
    
    
    if len(tvecs) > 0:
        # print("x : " + str(tvecs1[0][0]))
        # print("y : " + str(tvecs1[0][1]))
        # print("z : " + str(tvecs1[0][2]))
        # print("roll : " + str(euler_angle[0]))
        # print("pitch: " + str(euler_angle[1]))
        # print("yaw  : " + str(euler_angle[2]))
        return euler_angle[0], euler_angle[1]

def calc_angle_rotation(tvecs1, rvecs1):
    transform_translation_x = tvecs1[0][0]
    transform_translation_y = tvecs1[0][1]
    transform_translation_z = tvecs1[0][2]

    rotation_matrix = np.eye(4)
    rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs1[0]))[0]
    r = R.from_matrix(rotation_matrix[0:3, 0:3])
    quat = r.as_quat()   
        
    transform_rotation_x = quat[0] 
    transform_rotation_y = quat[1] 
    transform_rotation_z = quat[2] 
    transform_rotation_w = quat[3] 
    
    roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x, 
                                                transform_rotation_y, 
                                                transform_rotation_z, 
                                                transform_rotation_w)
    
    roll_x = math.degrees(roll_x)
    pitch_y = math.degrees(pitch_y)
    yaw_z = math.degrees(yaw_z)
    # roll_x = (roll_x + 360) % 360
    # pitch_y = (pitch_y + 180) % 180
    # yaw_z = (yaw_z + 90) % 90
    # print("transform_translation_x: {}".format(transform_translation_x))
    # print("transform_translation_y: {}".format(transform_translation_y))
    # print("transform_translation_z: {}".format(transform_translation_z))
    print(str(roll_x))
    # print("pitch_y: {}".format(pitch_y))
    # print("yaw_z: {}".format(yaw_z))

def euler_from_quaternion(x, y, z, w):
  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + y * y)
  roll_x = math.atan2(t0, t1)
      
  t2 = +2.0 * (w * y - z * x)
  t2 = +1.0 if t2 > +1.0 else t2
  t2 = -1.0 if t2 < -1.0 else t2
  pitch_y = math.asin(t2)
      
  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  yaw_z = math.atan2(t3, t4)
      
  return roll_x, pitch_y, yaw_z 


def give_points(corners, index_id_1, index_id_2, index_id_3, index_id_4):

    
    ccx_1 = corners[index_id_1][0][1]
    
    ccx_2 = corners[index_id_2][0][0]
    
    ccx_3 = corners[index_id_3][0][2]
   
    ccx_4 = corners[index_id_4][0][3]
    
    point = np.array([
        [[ccx_2[0], ccx_2[1]],   
        [ccx_1[0], ccx_1[1]], 
        [ccx_3[0], ccx_3[1]], 
        [ccx_4[0], ccx_4[1]]]
    ], dtype=np.float32)
    return point


def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

# def my_estimatePoseSingleMarkers_3P(corners, central_point, marker_size, mtx, distortion):
#     # Определение 3D координат трех известных точек маркера
#     half_size = marker_size / 2
#     marker_points_3d = np.array([
#         [central_point[0] - half_size, central_point[1] + half_size, 0],  # Верхний левый угол
#         [central_point[0] + half_size, central_point[1] - half_size, 0],  # Верхний правый угол
#         [central_point[0] - half_size, central_point[1] - half_size, 0]
#     ], dtype=np.float32)

    
#     for c in corners:
#         # Используем только три угла
#         selected_corners = c[:3]

#         # Решение задачи PnP с тремя точками без начальных значений
#         success, rvec, tvec = cv2.solvePnP(marker_points_3d, selected_corners, mtx, distortion, flags=cv2.SOLVEPNP_SQPNP)
        
#         if success:
#             rvecs.append(rvec)
#             tvecs.append(tvec)
#         else:
#             print("Ошибка при решении PnP")

#     return rvecs, tvecs

def draw_axes_at_center_of_markers(frame, center_1, camera_matrix, dist_coeffs):
    if len(corners) < 3:
        print("Недостаточно маркеров для расчета центра.")
        return

    # Получаем 2D координаты углов для трех маркеров
    center = np.array([center_1[0], center_1[1], 0.0], dtype=np.float32).reshape((3,1))
    axis_length = 0.1  # Длина осей
    axes = np.array([[0.1, 0, 0],    # Ось X
                     [0, 0.1, 0],    # Ось Y
                     [0, 0, 0.1]],   # Ось Z
                    dtype=np.float32)
    axes += center.T
    # Проецируем 3D оси на 2D изображение
    image_points, _ = cv2.projectPoints(axes, np.zeros((3, 1), dtype=np.float32), center, camera_matrix, dist_coeffs)

    # Рисуем оси
    origin = tuple(image_points[0].flatten().astype(int))  # Начальная точка (центр)
    x_axis = tuple(image_points[1].flatten().astype(int))  # Ось X
    y_axis = tuple(image_points[2].flatten().astype(int))  # Ось Y
    z_axis = tuple(image_points[3].flatten().astype(int))  # Ось Z, но игнорируется при отрисовке

    # Рисуем оси на изображении
    cv2.line(frame, origin, x_axis[:2], (255, 0, 0), 2)  # Ось X - красная
    cv2.line(frame, origin, y_axis[:2], (0, 255, 0), 2)  # Ось Y - зеленая
    cv2.line(frame, origin, z_axis[:2], (0, 0, 255), 2)  # Ось Z - синяя (если н


aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)  # Подставь нужный словарь
parameters = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(0)

camera_matrix = np.array([[4577.7689562929327, 0, 341.35776797777669],
                          [0, 1845.0965778655157, 189.94836015559704],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([27.505983603416613, -651.00215745718265, -1.1253795213096234,
       -0.18015690540560628, 4713.9105972395546 ])

marker_size = 0.01 
virtual_size = 0.02
data = []
data_y = []
i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    (corners, ids, rejected) = detector.detectMarkers(gray)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    if ids is not None:
        rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
        indices_id_1 = np.where(ids == 1)[0]
        indices_id_2 = np.where(ids == 0)[0]
        indices_id_3 = np.where(ids == 5)[0]
        indices_id_4 = np.where(ids == 4)[0]

        if indices_id_1.size > 0 and indices_id_4.size > 0 and indices_id_3.size > 0 and indices_id_2.size > 0:

            index_id_1 = indices_id_1[0]
            index_id_2 = indices_id_2[0]
            index_id_3 = indices_id_3[0]
            index_id_4 = indices_id_4[0]
            point = give_points(corners, index_id_1, index_id_2, index_id_3, index_id_4)
            # x, y = funcs.find_center(corners)
            # print(x, y)
            # cv2.circle(frame, (int(x), int(y)), 5, (255,0,0), -1)
            lenBott = math.sqrt((point[0][2][0] - point[0][3][0])**2 + (point[0][2][1] - point[0][3][1])**2) / 2
            rvecs1, tvecs1, _ = my_estimatePoseSingleMarkers(point, virtual_size, camera_matrix, dist_coeffs)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs1[0], tvecs1[0], 0.01)
            cv2.drawContours(frame, [point.astype(np.int32)], -1, (160, 120, 255), thickness=2)
            

            # front_points = np.array([
            #     [point[0][0][0], point[0][0][1]],
            #     [point[0][1][0], point[0][1][1]], 
            #     [point[0][2][0], point[0][2][1]], 
            #     [point[0][3][0], point[0][3][1]]
            # ], dtype=np.int32)
            # back_square = front_points - [0, int(lenBott)]
            # draw_cube(frame, front_points, back_square)

            # # aruco_angle(rvecs1, tvecs1)
            # calc_angle_rotation(tvecs1, rvecs1) #та самая 
            
            # roll, pitch, yaw = calc_ang(tvecs1, rvecs1)
            
        elif indices_id_4.size > 0 and indices_id_3.size > 0 and indices_id_2.size > 0:
            
            index_id_2 = indices_id_2[0]
            index_id_3 = indices_id_3[0]
            index_id_4 = indices_id_4[0]

            ccx_2 = corners[index_id_2][0][0].astype(int)
            
            ccx_3 = corners[index_id_3][0][2].astype(int)
        
            ccx_4 = corners[index_id_4][0][3].astype(int)

            p1 = np.array([ccx_2[0], ccx_2[1]])  # Маркер 1
            p2 = np.array([ccx_3[0], ccx_3[1]])  # Маркер 2
            p3 = np.array([ccx_4[0], ccx_4[1]])  # Маркер 3

            # Середина между p1 и p2
            midpoint = (p1 + p2) / 2

            # Предсказанная точка как зеркальное отражение p3 относительно midpoint
            predicted_point = 2 * midpoint - p3
            points = np.array([
                [[ccx_2[0], ccx_2[1]],
                [predicted_point[0].astype(int), predicted_point[1].astype(int)],   
                [ccx_3[0], ccx_3[1]], 
                [ccx_4[0], ccx_4[1]]]
            ], dtype=np.float32)
            print([predicted_point[0], predicted_point[1]])
            rvec, tvec, _ = my_estimatePoseSingleMarkers(points, virtual_size, camera_matrix, dist_coeffs)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.01)

            # rvec, tvec = my_estimatePoseSingleMarkers_3P(point, [424.75, 362.0], 0.04, camera_matrix, dist_coeffs)
            # draw_axes_at_center_of_markers(frame, [424.75, 362.0], camera_matrix, dist_coeffs)
            # cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.01)
            cv2.drawContours(frame, [points.astype(np.int32)], -1, (160, 120, 255), thickness=2)

    cv2.imshow('ArUco Marker Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
