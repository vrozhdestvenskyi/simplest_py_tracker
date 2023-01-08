import numpy as np
import cv2

SRC_VIDEO_PATH = r'C:\shaterniy\forest.avi'
SRC_GT_PATH = r'C:\shaterniy\forest.txt'
GRID_SZ = 10
SYMMETRIC = True
LK_SETT = dict(
    winSize = (15, 15),
    maxLevel = 4,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))

video = cv2.VideoCapture(SRC_VIDEO_PATH)
txt_file = open(SRC_GT_PATH, 'r')

def read():
    if not video.isOpened():
        return False, None, None
    success, frame = video.read()
    if not success:
        return False, None, None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gt = np.array([int(w) for w in txt_file.readline().rstrip('\n').split(' ')])
    return success, frame, gt

def calc_ref_pts(im):
    shape = np.array(im.shape)
    n = GRID_SZ + 1
    step = shape // n
    left = (shape - (im.shape // step) * step) // 2
    grid = np.mgrid[left[1] : im.shape[1] + 1 : step[1], left[0] : im.shape[0] + 1 : step[0]][:, 1 : -1, 1 : -1]
    return grid.T.flatten().reshape(-1, 2)

def track(im_prev, im_next):
    pts_prev = calc_ref_pts(im_prev)
    pts_next = cv2.calcOpticalFlowPyrLK(im_prev, im_next, pts_prev.astype(np.float32), None, **LK_SETT)[0]
    return pts_next - pts_prev

success, im_prev, gt_prev = read()
if success:
    success, im_next, gt_next = read()
while success:
    prev2next = track(im_prev, im_next)
    if SYMMETRIC:
        next2prev = track(im_next, im_prev)
        prev2next = np.vstack((prev2next, -next2prev))
    print(np.median(prev2next, axis=0), gt_prev - gt_next)
    im_prev, gt_prev = im_next.copy(), gt_next
    success, im_next, gt_next = read()

video.release()
txt_file.close()
