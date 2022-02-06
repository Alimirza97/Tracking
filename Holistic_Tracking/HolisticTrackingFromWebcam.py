import cv2
import time
import HolisticTrackingModel as htm

def main():
        pTime = 0
        wCam, hCam = 960, 540
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
        detector = htm.HolisticTracking(min_detection_confidence=0.7)
        while cap.isOpened():
            success, img = cap.read()
            if success:
                img = detector.AllTrackingWithoutBackground(img)
                #body = detector.AllTrackingWithoutBackground(img)
                """
                rHand = detector.FindRightHand(img)
                lHand = detector.FindLeftHand(img)
                pose = detector.FindPose(img)
                face = detector.FindFace(img)
                """
                # -------------------------------------FPS Calculate-----------------------------------------
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(img, f'{int(fps)}', (580, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                """
                cv2.putText(lHand, f'{int(fps)}', (580, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(rHand, f'{int(fps)}', (580, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(pose, f'{int(fps)}', (580, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(face, f'{int(fps)}', (580, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                """
                # -------------------------------------------------------------------------------------------
                cv2.imshow("Holistic Tracking", img)
                #cv2.imshow("Holistic Tracking Without Body", body)
                """
                cv2.imshow("Left Hand", lHand)
                cv2.imshow("Right Hand", rHand)
                cv2.imshow("Pose", pose)
                cv2.imshow("Face", face)
                """

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                print("Unsuccess")

        print("End")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()