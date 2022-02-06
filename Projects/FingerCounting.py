import cv2
import time
import os
import Hand_Tracking.HandTrackingModel as htm

def main():
    pTime = 0
    wCam, hCam = 960, 540
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    detector = htm.HandDetector(maxHands=1, detectionCon=0.75)
    folderPath = "../Images/Fingers"
    myList = os.listdir(folderPath)
    #print(myList)
    overlayList = []
    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        #print(f'{folderPath}/{imPath}')
        overlayList.append(image)
    overlayList.pop()
    #print(len(overlayList))
    tipIds = [4, 8, 12, 16, 20]
    while cap.isOpened():
        success, img = cap.read()
        if success:
            img = detector.FindMultiHands(img)
            lmList, bbox = detector.FindPosition(img)
            if lmList:
                #print(lmList)
                fingers = []
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                for id in range(1, 5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                if lmList[tipIds[0]][2] < lmList[tipIds[1]][2] and \
                        lmList[tipIds[0]][2] < lmList[tipIds[2]][2] and \
                        lmList[tipIds[0]][2] < lmList[tipIds[3]][2] and \
                        lmList[tipIds[0]][2] < lmList[tipIds[4]][2] and \
                        lmList[0][2] < lmList[17][2]:
                    totalFinger = 6
                else:
                    if lmList[tipIds[0]][2] > lmList[0][2]:
                        totalFinger = 7
                    else:
                        totalFinger = fingers.count(1)

                #print(fingers)
                #totalFinger = fingers.count(1)
                print(totalFinger)

                h, w, c = overlayList[totalFinger].shape
                img[0:h, 0:w] = overlayList[totalFinger]

                cv2.rectangle(img, (20, 180), (110, 270), (0, 255, 0), cv2.FILLED)
                if totalFinger != 6 and totalFinger != 7:
                    cv2.putText(img, str(totalFinger), (40, 250), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 10)
                elif totalFinger == 6:
                    cv2.putText(img, "Up", (35, 240), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 5)
                elif totalFinger == 7:
                    cv2.putText(img, "Down", (22, 240), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        # -------------------------------------FPS Calculate-----------------------------------------
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'{int(fps)}', (880, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        # -------------------------------------------------------------------------------------------
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