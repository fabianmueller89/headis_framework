import cv2
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

def mouseclick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(img[y,x])



video_path = "C:/Users/Fabian/Documents/Headis Mainz/UniprojektHeadisData/VideoMarkerLines/20220111_211213.mp4"

fig3d = plt.figure()
ax = fig3d.add_subplot(111, projection = '3d')

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
        #Filter red parts
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        scale = 0.12
        width = int(img_hsv.shape[1] * scale)
        height = int(img_hsv.shape[0] * scale)
        img_rez = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        r = img_rez[:, :, 0].ravel()
        g = img_rez[:, :, 1].ravel()
        b = img_rez[:, :, 2].ravel()
        print(len(r))

        light_red = (110, 145, 100)
        dark_red = (160, 200, 200)

        mask = cv2.inRange(img_hsv, light_red, dark_red)
        img_hsv_rez = cv2.resize(img_hsv, (width, height), interpolation=cv2.INTER_AREA)
        mask_rez = cv2.inRange(img_hsv_rez, light_red, dark_red)

        #mask_rez = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
        mask_bin = (np.array(mask_rez.ravel()) > 0)
        rm = np.array(r) * mask_bin
        gm = np.array(g) * mask_bin
        bm = np.array(b) * mask_bin

        cv2.imshow('img', mask)
        cv2.setMouseCallback('img', mouseclick)

        #ax.scatter(r, g, b, c='r', alpha=0.1)
        #ax.scatter(rm, gm, bm, c='b', alpha=0.1)
        #ax.set_xlabel('r')
        #ax.set_ylabel('g')
        #ax.set_zlabel('b')
        #plt.show()

        #cv2.waitKey()
        if cv2.waitKey(1) == 27:
            break
video_cap.release()
cv2.destroyAllWindows()
