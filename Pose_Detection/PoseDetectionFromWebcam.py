import cv2
import time
import PoseDetectionModel as pdm

def main():
    pTime = 0
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    detector = pdm.PoseDetection(min_detection_confidence=0.7)
    while cap.isOpened():
        success, img = cap.read()
        if success:
            img = detector.FindPose(img)
            #body = detector.FindPoseWithoutBody(img, draw=False)
            lmList = detector.FindPosition(img)
            if lmList:
                cv2.line(img, (lmList[12][1], lmList[12][2]), (lmList[11][1], lmList[11][2]), (0, 255, 0), 2)
                #print(lmList)
            # -------------------------------------FPS Calculate-----------------------------------------
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'{int(fps)}', (580, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
            # -------------------------------------------------------------------------------------------
            cv2.imshow("Pose Detection", img)
            #cv2.imshow("Pose Detection Body", body)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print("Unsuccess")

    print("End")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()