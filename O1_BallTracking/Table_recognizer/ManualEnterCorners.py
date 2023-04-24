import cv2
import numpy as np
from O1_BallTracking.Table_recognizer.TableDetectorClass import cTableDetector

### Manual Determination of table corners in image (WiSe2021 Projektseminar)

class cManualDetermination(cTableDetector):
    def __init__(self):
        pass

    def get_coordinate_by_mouse_click(img, window_name=None):
        """
        gets a position within an image by mouse. First opens the picture then you can click on the desired location with
        the mouse, then this method will return the x,y coordinates
        :param img: the image
        :param window_name: name of the window on which to klick
        :return: (x, y)
        """

        def get_mouse_coo(event, x, y, flags, param):

            if event == cv.EVENT_LBUTTONDOWN:
                param.set_values(True, x, y)

        # I couldn't find a better solution to save the local variables of the callback function
        class MouseCallbackVariableStorage:
            def __init__(self):
                self.values = False, -1, -1

            def set_values(self, c, x, y):
                self.values = c, x, y

        object_tuple = MouseCallbackVariableStorage()
        if window_name is None:
            window_name = 'temp'
            cv2.imshow(window_name, img)

        cv2.setMouseCallback(window_name, get_mouse_coo, object_tuple)
        while not object_tuple.values[0]:
            cv2.waitKey(1)
        if window_name is None:
            cv2.destroyWindow(window_name)
        return object_tuple.values[1], object_tuple.values[2]

    def get_homographie_matrix():
        print('now select the table corners')
        cv2.imshow('select table corners', current_image)
        table_corners = np.zeros((2, 4))
        for i in range(4):
            x, y = self.get_coordinate_by_mouse_click(current_image, 'select table corners')
            table_corners[0, i] = x
            table_corners[1, i] = y
        cv2.destroyWindow('select table corners')

        # get camera
        cam = data_IO.load_cam(cam_name)
        table_world = np.load(path_table_corners / '{}_world.npy'.format(setup_name))
        R, t = one_view_reconstruction.calc_cam_pos(table_corners, table_world, cam['K'])
        G = np.vstack((np.hstack((R, t.reshape(3, 1))), np.array([0, 0, 0, 1]).reshape(1, 4)))
        return G