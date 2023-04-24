import cv2
import numpy as np
#from O1_BallTracking.BallPreDetection.CVBlobDetector import cCVBlobDetector
#from O1_BallTracking.BallPreDetection.CNNPredetector import cCNNDetector

from O1_BallTracking.Table_recognizer.ArucoMarkerDetector import cArucoMarkerDetector
#from O1_BallTracking.Table_recognizer.CV_TableDetector import cCV_TableDetector
#from O1_BallTracking.Table_recognizer.ManualEnterCorners import cManualDetermination
from O1_BallTracking.Table_recognizer.ZhangMethod import cZhangMethod

#from O1_BallTracking.WorldCoord_Reconstruction.EllipsesParasOptimizer import cEllipseReconstructor
#from O1_BallTracking.WorldCoord_Reconstruction.DeepthOptimizer import cDeepthOptimizer

### Ball Tracker

class cBallTracker():
    def __init__(self):
        self.table_recognizer = cZhangMethod()
        self.predetector = None
        self.threeD_reconstructor = None

    def show_world_coordinate_system(self, P_mat, img, color = None):
        scale = 0.1 # arrow lengths in m
        origin = P_mat[:,-1].astype(int)[:-1]
        x_axis = P_mat.dot(np.array([scale, 0.0, 0.0 ,1.0])).astype(int)[:-1]
        y_axis = P_mat.dot(np.array([0.0, scale, 0.0 ,1.0])).astype(int)[:-1]
        z_axis = P_mat.dot(np.array([0.0, 0.0, scale ,1.0])).astype(int)[:-1]
        if isinstance(color, tuple):
            cv2.arrowedLine(img, origin, x_axis, color, 3, 8, 0, 0.1)
            cv2.arrowedLine(img, origin, y_axis, color, 3, 8, 0, 0.1)
            cv2.arrowedLine(img, origin, z_axis, color, 3, 8, 0, 0.1)
        else:
            cv2.arrowedLine(img, origin, x_axis, (255,0,0), 3, 8, 0, 0.1)
            cv2.arrowedLine(img, origin, y_axis, (0,255,0), 3, 8, 0, 0.1)
            cv2.arrowedLine(img, origin, z_axis, (0,0,255), 3, 8, 0, 0.1)


    def get_3d_positions_from_video(self, video_path):

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

                ### Table Recognizer => Transformation matrix from image to 3D
                P_mat_lin = self.table_recognizer.get_full_camera_matrix(img)
                #P_mat_nl = self.table_recognizer.get_full_camera_matrix(img, linear_b=False)

                if isinstance(P_mat_lin, np.ndarray):
                    ### Show world coordinate system on table
                    self.show_world_coordinate_system(P_mat_lin, img, (255,0,0))

                    ### Predetection => Localize ball in image

                    ### 3D reconstruction => Reconstruct 3D position from ball-image Snipet and transformation matrix


                cv2.imshow('img', img)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
        video_cap.release()
        cv2.destroyAllWindows()
        return None