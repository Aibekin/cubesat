import cv2
import numpy as np
from funcs import calc_ang
import time 
import matplotlib.pyplot as plt
from collections import deque

marker_length = 0.03 
board_size = 0.10 
offset = board_size / 2  # Смещение для определения центра
buffer_size =1000000
roll_data = deque(maxlen=buffer_size)
pitch_data = deque(maxlen=buffer_size)
yaw_data = deque(maxlen=buffer_size)
time_data = deque(maxlen=buffer_size)

# Инициализация времени
start_time = time.time()
def estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs):
   
    if ids is None or len(ids) == 0:
        return False, None, None

    object_points = [] 
    image_points = []
    print(board.getIds())
    print(ids)
    # Итерируем по всем маркерам доски
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in ids:

            marker_idx = np.where(ids == marker_id)[0][0]
  
            obj_pts = board.getObjPoints()[marker_idx]

            img_pts = corners[i].reshape(-1, 2)

            object_points.append(obj_pts)
            image_points.append(img_pts)

    if len(object_points) == 0:
        return False, None, None

    object_points = np.concatenate(object_points, axis=0)
    image_points = np.concatenate(image_points, axis=0)

    retval, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs
    )
    return retval, rvec, tvec

def board_corn_id(marker_length, offset):
    board_corners = []
    board_ids = []

    side_ids = [
        [0, 1, 2, 3],  # Сторона 1
        [4, 5, 6, 7],  # Сторона 2
        [8, 9, 10, 11],  # Сторона 3
        [12, 13, 14, 15],  # Сторона 4
        [16, 17, 18, 19],  # Сторона 5
        [20, 21, 22, 23],  # Сторона 6
    ]

    # Координаты для каждой стороны
    side_offsets = [
        [0, 0, -offset],  # Передняя сторона
        [offset, 0, 0],  # Правая сторона
        [0, 0, offset],  # Задняя сторона
        [-offset, 0, 0],  # Левая сторона
        [0, -offset, 0],  # Верхняя сторона
        [0, offset, 0],  # Нижняя сторона
    ]

    for i, ids in enumerate(side_ids):
        x_offset, y_offset, z_offset = side_offsets[i]
        corners = [
            [x_offset - marker_length / 2, y_offset - marker_length / 2, z_offset],
            [x_offset + marker_length / 2, y_offset - marker_length / 2, z_offset],
            [x_offset + marker_length / 2, y_offset + marker_length / 2, z_offset],
            [x_offset - marker_length / 2, y_offset + marker_length / 2, z_offset],
        ]
        for j in range(4):  # Один маркер на каждой позиции
            board_corners.append(np.array(corners))
            board_ids.append(ids[j])

    # Преобразуем данные в формат OpenCV
    board_corners = np.array(board_corners, dtype=np.float32)
    board_ids = np.array(board_ids, dtype=np.int32)
    return board_corners, board_ids

camera_matrix = np.array([
                        [874.27826124, 0, 337.49038078],
                        [0, 867.79031673, 196.680340],
                        [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.array([0.57624005, -3.3464065, -0.01874216, 0.05337322, 2.87051004])

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)  # Подставь нужный словарь
parameters = cv2.aruco.DetectorParameters()

board_corners, ids = board_corn_id(marker_length, offset)
board = cv2.aruco.Board(board_corners, aruco_dict, ids)
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

cap = cv2.VideoCapture(0)
plt.ion()  # Включаем интерактивный режим
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle('Real-Time Roll, Pitch, Yaw')

axs[0].set_title('Roll')
axs[0].set_ylim(-180, 180)
axs[0].set_xlim(0, buffer_size)
axs[0].grid(True)
line_roll, = axs[0].plot([], [], 'r-')

axs[1].set_title('Pitch')
axs[1].set_ylim(-180, 180)
axs[1].set_xlim(0, buffer_size)
axs[1].grid(True)
line_pitch, = axs[1].plot([], [], 'g-')

axs[2].set_title('Yaw')
axs[2].set_ylim(-180, 180)
axs[2].set_xlim(0, buffer_size)
axs[2].grid(True)
line_yaw, = axs[2].plot([], [], 'b-')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
while True:
    
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids_mark, _ = detector.detectMarkers(gray)
    
    if ids_mark is not None:
        retval, rvec, tvec = estimatePoseBoard(corners, ids_mark, board, camera_matrix, dist_coeffs)
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.01)
        roll, pitch, yaw = calc_ang(tvec, rvec)
        current_time = time.time() - start_time
        time_data.append(current_time)
        roll_data.append(roll)
        pitch_data.append(pitch)
        yaw_data.append(yaw)
        
        # Обновляем графики
        line_roll.set_xdata(range(len(roll_data)))
        line_roll.set_ydata(roll_data)
        axs[0].set_xlim(max(0, len(roll_data)-buffer_size), len(roll_data))
        axs[0].relim()
        axs[0].autoscale_view(True, True, True)
        
        line_pitch.set_xdata(range(len(pitch_data)))
        line_pitch.set_ydata(pitch_data)
        axs[1].set_xlim(max(0, len(pitch_data)-buffer_size), len(pitch_data))
        axs[1].relim()
        axs[1].autoscale_view(True, True, True)
        
        line_yaw.set_xdata(range(len(yaw_data)))
        line_yaw.set_ydata(yaw_data)
        axs[2].set_xlim(max(0, len(yaw_data)-buffer_size), len(yaw_data))
        axs[2].relim()
        axs[2].autoscale_view(True, True, True)
        
        # Перерисовка графиков
        plt.pause(0.001)
    cv2.imshow("Video", frame)

    # Выход по клавише ESC
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
