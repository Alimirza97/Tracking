import numpy as np
import time
import math
import cv2
from random import randrange
import Pose_Detection.PoseDetectionModel as pdm

def abc(number):
    if number < 0:
        number *= -1
    return number

def main():
    pTime = 0
    wCam, hCam = 960, 540
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    detector = pdm.PoseDetection(min_detection_confidence=0.7)
    while cap.isOpened():
        success, img = cap.read()
        if success:
            img = cv2.flip(img, 1)
            img = detector.FindPose(img, draw=False)
            lmList = detector.FindPosition(img)
            img[np.where((img != [0, 0, 0]).all(axis=2))] = [0, 0, 0]
            if lmList:
                #for lm in lmList:
                    #cv2.circle(img, (lm[1], lm[2]), 5, (255, 0, 255), cv2.FILLED)

                b_one_x, b_one_y = (lmList[12][1] + lmList[11][1]) // 2, (lmList[12][2] + lmList[11][2]) // 2
                b_two_x, b_two_y = (lmList[24][1] + lmList[23][1]) // 2, (lmList[24][2] + lmList[23][2]) // 2

            # -----------------------------------------Head--------------------------------------------------------------------------------------------------------------------------------------
            # -----------------------------------------Eyes--------------------------------------------------------------------------------------------------------------------------------------
                cv2.circle(img, (lmList[5][1], lmList[5][2]), 5, (randrange(256), randrange(256), randrange(256)), cv2.FILLED)
                cv2.circle(img, (lmList[2][1], lmList[2][2]), 5, (randrange(256), randrange(256), randrange(256)), cv2.FILLED)
                #cv2.line(img, (lmList[6][1], lmList[6][2]), (lmList[4][1], lmList[4][2]), (randrange(256), randrange(256), randrange(256)), 2)
                #cv2.line(img, (lmList[1][1], lmList[1][2]), (lmList[3][1], lmList[3][2]), (randrange(256), randrange(256), randrange(256)), 2)
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # ------------------------------------------Ear--------------------------------------------------------------------------------------------------------------------------------------
                #cv2.line(img, (lmList[8][1], lmList[8][2]), (lmList[10][1], lmList[10][2]), (randrange(256), randrange(256), randrange(256)), 2)
                #cv2.line(img, (lmList[9][1], lmList[9][2]), (lmList[7][1], lmList[7][2]), (randrange(256), randrange(256), randrange(256)), 2)
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                length = math.hypot(lmList[8][1] - lmList[7][1], lmList[8][2] - lmList[7][2])
                center_coordinates = (lmList[0][1], lmList[0][2])
                axesLength, angle, startAngle, endAngle, color, thickness = (int(length), int(length/2)), lmList[8][2]-lmList[7][2], 0, 360, (randrange(256), randrange(256), randrange(256)), 2
                cv2.ellipse(img, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness)
                cv2.line(img, (lmList[10][1], lmList[10][2]), (lmList[9][1], lmList[9][2]), (randrange(256), randrange(256), randrange(256)), 2)
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #----------------------------------------Left Hand-----------------------------------------------------------------------------------------------------------------------------------
                cv2.line(img, (lmList[12][1], lmList[12][2]), (lmList[14][1], lmList[14][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[14][1], lmList[14][2]), (lmList[16][1], lmList[16][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[16][1], lmList[16][2]), (lmList[18][1], lmList[18][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[16][1], lmList[16][2]), (lmList[20][1], lmList[20][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[16][1], lmList[16][2]), (lmList[22][1], lmList[22][2]), (randrange(256), randrange(256), randrange(256)), 2)
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #----------------------------------------Right Hand----------------------------------------------------------------------------------------------------------------------------------
                cv2.line(img, (lmList[11][1], lmList[11][2]), (lmList[13][1], lmList[13][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[13][1], lmList[13][2]), (lmList[15][1], lmList[15][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[15][1], lmList[15][2]), (lmList[17][1], lmList[17][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[15][1], lmList[15][2]), (lmList[19][1], lmList[19][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[15][1], lmList[15][2]), (lmList[21][1], lmList[21][2]), (randrange(256), randrange(256), randrange(256)), 2)
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #-------------------------------------------Body-------------------------------------------------------------------------------------------------------------------------------------
                cv2.line(img, (lmList[12][1], lmList[12][2]), (lmList[11][1], lmList[11][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[12][1], lmList[12][2]), (lmList[24][1], lmList[24][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[11][1], lmList[11][2]), (lmList[23][1], lmList[23][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[24][1], lmList[24][2]), (lmList[23][1], lmList[23][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[12][1], lmList[12][2]), (lmList[23][1], lmList[23][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[11][1], lmList[11][2]), (lmList[24][1], lmList[24][2]), (randrange(256), randrange(256), randrange(256)), 2)
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # ---------------------------------------Left Foot-----------------------------------------------------------------------------------------------------------------------------------
                cv2.line(img, (lmList[24][1], lmList[24][2]), (lmList[26][1], lmList[26][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[26][1], lmList[26][2]), (lmList[28][1], lmList[28][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[28][1], lmList[28][2]), (lmList[30][1], lmList[30][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[28][1], lmList[28][2]), (lmList[32][1], lmList[32][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[30][1], lmList[30][2]), (lmList[32][1], lmList[32][2]), (randrange(256), randrange(256), randrange(256)), 2)
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # ---------------------------------------Right Foot----------------------------------------------------------------------------------------------------------------------------------
                cv2.line(img, (lmList[23][1], lmList[23][2]), (lmList[25][1], lmList[25][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[25][1], lmList[25][2]), (lmList[27][1], lmList[27][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[27][1], lmList[27][2]), (lmList[29][1], lmList[29][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[27][1], lmList[27][2]), (lmList[31][1], lmList[31][2]), (randrange(256), randrange(256), randrange(256)), 2)
                cv2.line(img, (lmList[29][1], lmList[29][2]), (lmList[31][1], lmList[31][2]), (randrange(256), randrange(256), randrange(256)), 2)
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # ---------------------------------------FPS Calculate-----------------------------------------
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'{int(fps)}', (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
        # ---------------------------------------------------------------------------------------------
            cv2.imshow("Image", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print("Unsuccess")

    print("End")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
