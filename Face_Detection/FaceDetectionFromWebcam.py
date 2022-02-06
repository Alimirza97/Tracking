import cv2
import time
import FaceDetectionModel as fdm

def main():
    pTime = 0
    wCam, hCam = 960, 540
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    detection = fdm.FaceDetection(min_detection_confidence=0.6)
    detection.SetDrawingSettings((255, 255, 0), 1)
    while cap.isOpened():
        success, img = cap.read()
        if success:
            img = detection.FindFace(img, custom_rectangle=True)
            #bbox = detection.FindPosition(img)
            #print(bbox)
            #bbox = detection.FindCoordinate()
            #print(bbox)
        # -----------FPS Calculate-----------------------------------------------------
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
        # -----------------------------------------------------------------------------
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow("Face Image", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Unsuccessful")
    print("End")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()