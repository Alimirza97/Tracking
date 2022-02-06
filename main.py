import cv2
import time
import Hand_Tracking.HandTrackingModel as htm


def main():
    pTime = 0
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    detector = htm.HandDetector(detectionCon=0.7)
    detector.set_connection_drawing_spec((0, 255, 0), 2, 2)
    detector.set_landmark_Drawing_spec((0, 255, 0), 2, 2)
    detector.set_rectangle_drawing_spec((255, 255, 0), 2)
    while cap.isOpened():
        success, img = cap.read()
        if success:
            hand_img = detector.FindMultiHands(img, rectangle=False, multi_rectangle=True)
            #hand_img = detector.FindHandsWithoutHands(img, draw=True, rectangle=True, multi_rectangle=False)
            lmList, RbBox, LbBox = detector.FindAllPosition(img)
            handType=""
            if lmList and (RbBox or LbBox):
                if RbBox and LbBox:
                    xmin = min(LbBox[0], RbBox[0])
                    ymin = min(LbBox[1], RbBox[1])
                    xmax = max(LbBox[2], RbBox[2])
                    ymax = max(LbBox[3], RbBox[3])
                    if lmList[17][1] < lmList[5][1]:
                        handType = "Right"
                    else:
                        handType = "Left"
                    cv2.putText(img, handType, (xmin + 10, ymin - 10), cv2.FONT_HERSHEY_PLAIN,
                                detector.landmark_drawing_spec.thickness, detector.landmark_drawing_spec.color,
                                detector.landmark_drawing_spec.thickness)

                    if lmList[38][1] < lmList[26][1]:
                        handType = "Right"
                    else:
                        handType = "Left"
                    cv2.putText(img, handType, (xmax - 100, ymin - 10), cv2.FONT_HERSHEY_PLAIN,
                                detector.landmark_drawing_spec.thickness, detector.landmark_drawing_spec.color,
                                detector.landmark_drawing_spec.thickness)
                else:
                    if RbBox:
                        if lmList[17][1] < lmList[5][1]:
                            handType = "Right"
                        else:
                            handType = "Left"
                        cv2.putText(img, handType, (RbBox[0] + 10, RbBox[1] - 10), cv2.FONT_HERSHEY_PLAIN, detector.landmark_drawing_spec.thickness, detector.landmark_drawing_spec.color, detector.landmark_drawing_spec.thickness)
                    if LbBox:
                        if lmList[38][1] < lmList[26][1]:
                            handType = "Right"
                        else:
                            handType = "Left"
                        cv2.putText(img, handType, (LbBox[0] + 10, LbBox[1] - 10), cv2.FONT_HERSHEY_PLAIN, detector.landmark_drawing_spec.thickness, detector.landmark_drawing_spec.color, detector.landmark_drawing_spec.thickness)
            # -----------FPS Calculate-----------------------------------------------------
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            # -----------------------------------------------------------------------------
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