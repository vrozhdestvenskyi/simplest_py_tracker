import numpy as np
import cv2

SRC_IM_PATH = r'C:\shaterniy\forest.jpg'
DST_VIDEO_PATH = r'C:\shaterniy\forest.avi'
DST_SHIFTS_PATH = r'C:\shaterniy\forest.txt'
DST_RES = np.array([640, 480])
STEP = 20
STD = 10
N_FRAMES = 100

np.random.seed(4)

im = cv2.imread(SRC_IM_PATH, cv2.IMREAD_UNCHANGED)

def is_valid(center, res, res_global):
    return np.all(center >= res / 2.0) and np.all(center + res / 2.0 < res_global)

shift_tl = np.array([STEP, STEP])
shift_br = im.shape[::-1][1:] - DST_RES - STEP

def is_valid(p):
    return np.all(p >= shift_tl) and np.all(p <= shift_br)

video = cv2.VideoWriter(DST_VIDEO_PATH, cv2.VideoWriter_fourcc(*'DIVX'), 30, DST_RES)
txt_file = open(DST_SHIFTS_PATH, 'w')
shift = shift_tl.copy()
direction = 1
n_frames = 0
while is_valid(shift) and n_frames < N_FRAMES:
    shift_noised = shift + np.random.normal(0, STD, 2)
    shift_noised = shift_noised.astype(int).clip(shift_tl, shift_br)

    video.write(im[shift_noised[1] : shift_noised[1] + DST_RES[1], shift_noised[0] : shift_noised[0] +  DST_RES[0]])
    txt_file.write('%d %d\n' % tuple(shift_noised))
    n_frames += 1

    shift_next = shift + np.array([STEP, 0]) * direction
    if not is_valid(shift_next):
        shift_next = shift + np.array([0, STEP])
        direction *= -1
    shift = shift_next
video.release()
txt_file.close()
