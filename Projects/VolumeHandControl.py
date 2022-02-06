import cv2
import time
import Hand_Tracking.HandTrackingModel as htm
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def main():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    volRange = volume.GetVolumeRange()
    minVol = volRange[0]
    maxVol = volRange[1]
    volBar = 400
    volPer = 0
    area = 0
    color = (255, 0, 0)
    pTime = 0
    wCam, hCam = 960, 540
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    detector = htm.HandDetector(maxHands=1, detectionCon=0.7)
    while cap.isOpened():
        success, img = cap.read()
        if success == True:
            #Find Hand
            img = detector.FindMultiHands(img, rectangle=True)
            lmList, bbox = detector.FindPosition(img)
            if lmList:
                cv2.circle(img, (lmList[20][1], lmList[20][2]), 15, (255, 0, 255), cv2.FILLED)
                # Filter based on size
                area = (bbox[2] - bbox[0]) * (bbox[3]-bbox[1]) // 100
                #print(area)
                if 250 < area < 1000:
                    #print("Yes")
                    # Find Distance between index and Thumb
                    img, length, lineInfo = detector.FindDistance(img, 4, 8)

                    # Convert Valume
                    volBar = np.interp(length, [50, 200], [400, 150])
                    volPer = np.interp(length, [50, 200], [0, 100])

                    # Reduce Resolution to make it smoother
                    smoothness = 10
                    volPer = smoothness * round(volPer / smoothness)

                    # Check finger up
                    fingers = detector.FingersUp(img)
                    #print(fingers)

                    # If pinkt is Down set Volume
                    if fingers:
                        if not fingers[4]:
                            volume.SetMasterVolumeLevelScalar(volPer/100, None)
                            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                            cv2.circle(img, (lmList[20][1], lmList[20][2]), 15, (0, 255, 0), cv2.FILLED)
                            color = (0, 255, 0)
                        else:
                            color = (255 ,0, 0)
                # Drawings
                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
                cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
                cv2.putText(img, f'Vol Set: {int(cVol)}', (700, 50), cv2.FONT_HERSHEY_COMPLEX, 1, color, 3)
            # -----------FPS Calculate-----------------------------------------------------
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            # -----------------------------------------------------------------------------
            cv2.imshow("Tracking Image", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Unsuccessful")
    print("End")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()