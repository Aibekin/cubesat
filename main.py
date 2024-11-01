import cv2
import numpy as np
import math

def estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
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
    return rvecs, tvecs

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

def find_center(corners):
    [x1, y1] = corners[0][0][0]
    [x2, y2] = corners[0][0][1]
    [x3, y3] = corners[0][0][2]
    [x4, y4] = corners[0][0][3]
    
    x_center = (x1 + x2 + x3 + x4) / 4
    y_center = (y1 + y2 + y3 + y4) / 4
    return (x_center, y_center)

cap = cv2.VideoCapture(0)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
camera_matrix = np.array([[4577.7689562929327, 0, 341.35776797777669],
                          [0, 1845.0965778655157, 189.94836015559704],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([27.505983603416613, -651.00215745718265, -1.1253795213096234,
                        -0.18015690540560628, 4713.9105972395546])

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    
    if ids is not None:
        rvecs, tvecs = estimatePoseSingleMarkers(corners, 0.1, camera_matrix, dist_coeffs)
        
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

            lenBott = math.sqrt((point[0][2][0] - point[0][3][0])**2 + (point[0][2][1] - point[0][3][1])**2) / 2

            rvecs1, tvecs1 = estimatePoseSingleMarkers(point, 0.2, camera_matrix, dist_coeffs)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs1[0], tvecs1[0], 0.01)
            cv2.drawContours(frame, [point.astype(np.int32)], -1, (160, 120, 255), thickness=2)
            x, y = find_center(corners)
            print(x, y)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
