# https://www.youtube.com/watch?v=1wUnPu4OvOA

import numpy as np
import cv2

img = cv2.imread('forrest.jpg', cv2.IMREAD_COLOR)
print(img.shape)


selected_pts = []


def mouse_cb(event, x, y, flags, param):
	global selected_pts
	if event == cv2.EVENT_LBUTTONUP:
		selected_pts.append([x, y])
		cv2.circle(img, (x, y), 10, (0, 255, 0), 3)


def select_points(image, points_num):
	global selected_pts
	selected_pts = []
	cv2.namedWindow('image')
	cv2.setMouseCallback('image', mouse_cb)

	while 1:
		cv2.imshow('image', image)
		k = cv2.waitKey(1)
		if k == 27 or len(selected_pts) >= points_num:
			break

	cv2.destroyAllWindows()
	return np.array(selected_pts, dtype=np.float32)


def test1():
	src_points = select_points(img, 3)
	ny, nx = img.shape[:2]
	dst_points = np.array(src_points[::-1], dtype=np.float32)
	affine_m = cv2.getAffineTransform(src_points, dst_points)

	ny, nx = img.shape[:2]


	warped_img = cv2.warpAffine(img, affine_m, (nx, ny))
	cv2.imshow('image', warped_img)
	cv2.waitKey()
	cv2.destroyAllWindows()


def test2():
	"""
	DOBRZE ROKUJE!!!!
	"""
	src_points = select_points(img, 4)

	ny, nx = img.shape[:2]
	x,y = int(nx*0.9), int(ny*0.9)

	dst_pts = np.array([[0,y], [0,0], [x,0], [x,y]], dtype=np.float32)
	persp = cv2.getPerspectiveTransform(src_points, dst_pts)
	
	warp = cv2.warpPerspective(img, persp, (x, y))
	cv2.imshow('result', warp)
	cv2.waitKey()
	cv2.destroyAllWindows()

test2()