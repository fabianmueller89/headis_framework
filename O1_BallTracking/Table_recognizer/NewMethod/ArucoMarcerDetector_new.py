import numpy as np
import copy
from copy import deepcopy
import cv2

# Positions of Markers on Table
#                L
#       |------------------|
#        __________________          __
#      /         ^ y        \         \
#     /          | -> x      \         \  B
# LL /_LT________|_________RT_\ RR     _\_
#   LC        LM   RM         RC

#               b
#           |-------|
#       _________________   ___
#      |     _______     |  _|_dh ___
#      |    |# ### #|    |         |
#      |    | # ##  |    |         | h
#      |    |###_#_#|    |        _|_
#      |                 |
#      |_________________|

DICT_MARKER = {'d_h': 0.03, 'b': 0.096, 'h': 0.096} # Distance from edge to center


class cArucoMarkerDetector():
    def __init__(self, marker_id_desired):
        markerSize = 5
        totalMarkers = 250
        self.__marker_dictionary = f'DICT_{markerSize}X{markerSize}_{totalMarkers}'
        self.marker_id_desired = marker_id_desired
        self.__status_dict_init = {'aruco_marcers_detected': False, 'aruco_marcer_id_found': False}
        self.status_dict = copy.deepcopy(self.__status_dict_init)


    def get_correspondences_from_frame(self, frame):
        self.status_dict = copy.deepcopy(self.__status_dict_init)
        marker_corners_img_all, marker_id_all_lst = self.__get_marker_corners_from_frame(frame)

        if self.status_dict['aruco_marcer_id_found']:
            marker_corners_world, marker_corners_img = self.__get_arucomarker_point_correspondences(marker_corners_img_all, marker_id_all_lst)
            return marker_corners_world, marker_corners_img
        else:
            return None, None

    def __get_marker_corners_from_frame(self, frame):
        """
        Calculate corner point in image coordinates of aruco markers and determine marker ids
        :param frame: frame of image
        :return:
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        key = getattr(cv2.aruco, self.__marker_dictionary)
        arucoDict = cv2.aruco.Dictionary_get(key)
        arucoParam = cv2.aruco.DetectorParameters_create()
        marker_corners_img, marker_ids, marker_rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)

        if isinstance(marker_corners_img, list) and isinstance(marker_ids, np.ndarray):
            self.status_dict['aruco_marcers_detected'] = True

            if self.marker_id_desired in marker_ids:
                self.status_dict['aruco_marcer_id_found'] = True

        return marker_corners_img, marker_ids

    def __get_arucomarker_point_correspondences(self, marker_corners_img_lst, marker_id_lst):
        marker_corner_world = None
        marker_corners_img = None
        # Aruco Marker starts with left top corner and then clockwise
        for marker_corners_img_cand, marker_id in zip(marker_corners_img_lst, marker_id_lst):
            if marker_id == self.marker_id_desired:
                b, h = DICT_MARKER['b'] / 2, DICT_MARKER['h'] / 2
                marker_corner_world = np.array([[-b, -h, 0], [-b, h, 0], [b, h, 0], [b, -h, 0]])
                marker_corners_img = deepcopy(marker_corners_img_cand[0,:,:])
        return marker_corner_world, marker_corners_img

if __name__ == '__main__':
    video_path = "C:/Users/Fabian/Documents/Headis Mainz/UniprojektHeadisData/VideoDataAruco/20211114_170239.mp4"
    detect_obj = cArucoMarkerDetector(marker_id_desired=3)

    ### Read video
    video_cap = cv2.VideoCapture(video_path)
    #video_cap = cv2.VideoCapture(0)
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, start_frame = video_cap.read()
    if not success:
        print('could not read video. Check video name')
    else:
        while True:
            success, img = video_cap.read()
            X_world, X_img = detect_obj.get_correspondences_from_frame(img)

            print('X_w', X_world)
            print('X_img', X_img)

            cv2.imshow('img', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
    video_cap.release()
    cv2.destroyAllWindows()