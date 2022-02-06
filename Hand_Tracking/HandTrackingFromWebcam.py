import cv2
import time
import HandTrackingModel as htm

def main():
    pTime = 0
    wCam, hCam = 960, 540
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    detector = htm.HandDetector(detectionCon=0.7)
    #detector.set_random_color()
    while cap.isOpened():
        success, img = cap.read()
        if success:
            hand_img = detector.FindHandsWithoutHands(img, rectangle=True, multi_rectangle=True)
            # hand_img = detector.FindHandsWithoutHands(img, rectangle=True, multi_rectangle=False)
            # lmList, bbox = detector.FindPosition(img)
            # if lmList and bbox:
            # print(lmList, bbox)
            detector.set_connection_drawing_spec((255, 255, 0), 2, 2)
            detector.set_landmark_Drawing_spec((255, 0, 0), 2, 2)
            detector.set_rectangle_drawing_spec((0, 255, 0), 2)

            """
            lmList = detector.FindPosition(img)
            if len(lmList) != 0:
                for lm in lmList:
                    cv2.circle(img, (lm[1], lm[2]), 5, (0, 255, 0), cv2.FILLED)
                print(lmList)
            """
            # -----------FPS Calculate-----------------------------------------------------
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            # -----------------------------------------------------------------------------
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow("Tracking Hand Image", hand_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Unsuccessful")
    print("End")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()