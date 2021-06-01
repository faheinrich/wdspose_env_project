import cv2
import time

while(True):
    
    try:
        print("reading test.jpg")
        img = cv2.imread("workspace/test.jpg")
        img = cv2.resize(img, (img.shape[1]*3, img.shape[0]*3))
        cv2.imshow("3dpose_estimation", img)
        if cv2.waitKey(1) & 0xFF == 27: # use ESC to quit
            break
    except:
        print("failed")
    
    time.sleep(0.1)
