import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Arc
from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator
import math

plt.style.use('ggplot')

def calculate_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return (round((x1+x2)/2,2), round((y1+y2)/2,2))

def calculate_translation_vectors(coordinates):
    t_vectors = []
    for i in range(1, len(coordinates)):
        dx = coordinates[i][0] - coordinates[i - 1][0]
        dy = coordinates[i][1] - coordinates[i - 1][1]
        t_vectors.append((dx, dy))

    return t_vectors

def calculate_rotation_angles(coordinates):
    r_angles = []
    for i in range(1, len(coordinates)):
        num = np.dot(coordinates[0], coordinates[i])
        den = math.sqrt(sum([coordinates[0][0] ** 2, coordinates[0][1] ** 2])) * math.sqrt(sum([coordinates[i][0] ** 2, coordinates[i][1] ** 2]))
        res = round(num/den, 2)
        ang = math.acos(res)
        ang_deg = round(math.degrees(ang),2)
        r_angles.append(ang_deg)

    return r_angles

def construct_rotation_matrix(angles):
    matrix = []
    for i in range(len(angles)):
        angle_in_rad = np.radians(angles[i])
        r_matrix = np.array([[np.cos(angle_in_rad), -np.sin(angle_in_rad)],
                                      [np.sin(angle_in_rad), np.cos(angle_in_rad)]])

        matrix.append(r_matrix)

    return matrix

def construct_homogeneous_matrix(angles, vectors):
    h_matrices = []
    for i in range(len(angles)):
        dx = vectors[i][0]
        dy = vectors[i][1]
        angle_in_rad = np.radians(angles[i])
        cos_theta = np.cos(angle_in_rad)
        sin_theta = np.sin(angle_in_rad)
        homogeneous_matrix = np.array([[cos_theta, -sin_theta, dx],  # [ R11  R12  Tx ]
                                       [sin_theta, cos_theta, dy],  # [ R21  R22  Ty ]
                                       [0, 0, 1]])  # [  0    0    1 ]

        h_matrices.append(homogeneous_matrix)

    return h_matrices

model = YOLO('/Users/komalb/PycharmProjects/pythonProject2/runs/detect/train8/weights/best.pt')
confidence_threshold = 0.51


video_path = '/Users/komalb/Downloads/cups_folder/cup_in_motion.mp4'
cap = cv2.VideoCapture(video_path)

bboxes = []
video_frames = []

while True:
    ret, frame_data = cap.read()
    if not ret:
        break

    results = model.predict(frame_data, conf=confidence_threshold)

    if len(results[0].boxes)==0:
        continue

    bboxes.append(results[0].boxes[0].xyxyn[0].numpy())
    video_frames.append(frame_data)

cap.release()

centroids = [calculate_centroid(i) for i in bboxes]
rot_angles = calculate_rotation_angles(centroids)
trans_vectors = calculate_translation_vectors(centroids)
rotation_matrix = construct_rotation_matrix(rot_angles)
homogeneous_matrix = construct_homogeneous_matrix(rot_angles, trans_vectors)

c=0
fig, ax = plt.subplots()
image = video_frames[c]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def update(frame):
    global c
    c += 1

    if c>len(rot_angles):
        return

    image = video_frames[c]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_axes = np.array([[1, 0], [0, 1]])

    rot_matrix = rotation_matrix[c-1]
    new_axes = np.dot(rot_matrix, original_axes)

    ax.clear()
    ax.imshow(image, extent=[-1.5, 1.5, -1.5, 1.5], alpha=0.6)
    ax.quiver(0, 0, original_axes[0, 0], original_axes[1, 0], angles='xy', scale_units='xy', scale=1, color='red',
              label='Original Axes')
    ax.quiver(0, 0, original_axes[0, 1], original_axes[1, 1], angles='xy', scale_units='xy', scale=1, color='red')
    ax.quiver(0, 0, new_axes[0, 0], new_axes[1, 0], angles='xy', scale_units='xy', scale=1, color='blue',
              label='New Axes')
    ax.quiver(0, 0, new_axes[0, 1], new_axes[1, 1], angles='xy', scale_units='xy', scale=1, color='blue')

    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    angle_value = rot_angles[c-1]

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Axes Rotation: {angle_value} degrees')

    angle_x = 0.2 * np.cos(np.radians(angle_value / 2))
    angle_y = 0.2 * np.sin(np.radians(angle_value / 2))

    arc = Arc([0, 0], 0.4, 0.4, theta1=0, theta2=angle_value, color='black', linewidth=1.5)
    ax.add_patch(arc)
    ax.text(angle_x, angle_y, f"{angle_value:.2f}°", fontsize=12)
    angle_text = f"Pose: {angle_value:.2f} degrees"
    ax.text(-1, 1.2, angle_text, fontsize=12)

    ax.legend()


anim = animation.FuncAnimation(fig, update, frames=len(rot_angles), interval=1, repeat=False)
plt.show()



#Display the output video
'''
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame,conf=confidence_threshold)

    if len(results[0].boxes) == 0:
        continue

    for r in results:

        annotator = Annotator(frame)

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

    output = annotator.result()
    cv2.imshow('YOLO V8 Detection', output)
    cv2.waitKey(1)'''