import cv2
import numpy as np
import mediapipe as mp
import time
import math
from random import randrange

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.results = tuple()

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.landmark_drawing_spec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        self.connection_drawing_spec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        self.rectangle_color = (0, 255, 0)
        self.rectangle_thickness = 2

    def set_landmark_Drawing_spec(self, color=(0, 255, 0), thickness=2, circle_radius=2):
        self.landmark_drawing_spec = self.mpDraw.DrawingSpec(color=color, thickness=thickness, circle_radius=circle_radius)

    def set_connection_drawing_spec(self, color=(0, 255, 0), thickness=2, circle_radius=2):
        self.connection_drawing_spec = self.mpDraw.DrawingSpec(color=color, thickness=thickness, circle_radius=circle_radius)

    def set_rectangle_drawing_spec(self, color=(0, 255, 0), thickness=2):
        self.rectangle_color = color
        self.rectangle_thickness = thickness

    def set_random_color(self):
        self.landmark_drawing_spec = self.mpDraw.DrawingSpec(color=(randrange(256), randrange(256), randrange(256)), thickness=2, circle_radius=2)
        self.connection_drawing_spec = self.mpDraw.DrawingSpec(color=(randrange(256), randrange(256), randrange(256)), thickness=2, circle_radius=2)
        self.rectangle_color = (randrange(256), randrange(256), randrange(256))


    def FindMultiHands(self, img, draw=True, rectangle=False, multi_rectangle=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        lmList, RbBox, LbBox = self.FindAllPosition(img)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        image=img,
                        landmark_list=handLms,
                        connections=self.mpHands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.landmark_drawing_spec,
                        connection_drawing_spec=self.connection_drawing_spec
                    )
                    if rectangle:
                        if LbBox:
                            xmin = min(LbBox[0], RbBox[0])
                            ymin = min(LbBox[1], RbBox[1])
                            xmax = max(LbBox[2], RbBox[2])
                            ymax = max(LbBox[3], RbBox[3])
                            if multi_rectangle:
                                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), self.rectangle_color, self.rectangle_thickness)
                            else:
                                cv2.rectangle(img, (RbBox[0], RbBox[1]), (RbBox[2], RbBox[3]), self.rectangle_color, self.rectangle_thickness)
                                cv2.rectangle(img, (LbBox[0], LbBox[1]), (LbBox[2], LbBox[3]), self.rectangle_color, self.rectangle_thickness)
                        else:
                            cv2.rectangle(img, (RbBox[0], RbBox[1]), (RbBox[2], RbBox[3]), self.rectangle_color, self.rectangle_thickness)
        return img

    def FindHandsWithoutHands(self, img, draw=True, rectangle=False, multi_rectangle=False, white=False):
        img = self.FindMultiHands(img, draw=draw, rectangle=rectangle, multi_rectangle=multi_rectangle)
        if white:
            Wmask = (img[:, :, 0:3] == [0, 255, 0]).all(2)
            Wmask = (Wmask * 255).astype(np.uint8)
            return Wmask
        else:
            img[np.where((img != [0, 0, 0]).all(axis=2))] = [0, 0, 0]
            return img

    def FindPosition(self, img, handNo = 0):
        xList, yList, lmList, bBox = [], [], [], []
        list = self.FindCoordinate(handNo)
        if list:
            h, w, c = img.shape
            for item in list:
                cx, cy = int(item[1] * w), int(item[2] * h)
                xList.append(cx)
                yList.append(cy)
                lmList.append([item[0], cx, cy])
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bBox = [xmin - 20, ymin - 20, xmax + 20, ymax + 20]
        return lmList, bBox

    def __FindOnList(self, count, lmList):
        for item in lmList:
            if item[0] == count:
                return False
        return True

    def FindAllPosition(self, img):
        lmList, RxList, RyList, RbBox, LxList, LyList, LbBox = [], [], [], [], [], [], []
        if self.results:
            if self.results.multi_hand_landmarks:
                for myHand in self.results.multi_hand_landmarks:
                    for id, lm in enumerate(myHand.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if self.__FindOnList(id, lmList):
                            RxList.append(cx)
                            RyList.append(cy)
                            lmList.append([id, cx, cy])
                        else:
                            LxList.append(cx)
                            LyList.append(cy)
                            lmList.append([21 + id, cx, cy])

                xmin, xmax = min(RxList), max(RxList)
                ymin, ymax = min(RyList), max(RyList)
                RbBox = xmin - 20, ymin - 20, xmax + 20, ymax + 20
                if LxList and LyList:
                    xmin, xmax = min(LxList), max(LxList)
                    ymin, ymax = min(LyList), max(LyList)
                    LbBox = xmin - 20, ymin - 20, xmax + 20, ymax + 20
        return lmList, RbBox, LbBox

    def FindCoordinate(self, handNo = 0):
        lmList = []
        if self.results:
            if self.results.multi_hand_landmarks:
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    lmList.append([id, lm.x, lm.y, lm.z])
        return lmList

    def FingersUp(self, img):
        fingers = []
        tipIds = [4, 8, 12, 16, 20]
        lmList, bbox = self.FindPosition(img)
        if lmList:
            # Thumb
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # 4 Fingers
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def FindDistance(self, img, lenmark1=0, lenmark2=0, draw=True):
        lmList, bbox = self.FindPosition(img)
        if lmList:
            x1, y1 = lmList[lenmark1][1], lmList[lenmark1][2]
            x2, y2 = lmList[lenmark2][1], lmList[lenmark2][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if draw:
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
        else:
            length, x1, y1, x2, y2, cx, cy = 0, 0, 0, 0, 0, 0, 0
        return img, length, [x1, y1, x2, y2, cx, cy]

    def HandType(self, img):
        lmList, RbBox, LbBox = self.FindAllPosition(img)
        if lmList:
            if lmList[17][1] < lmList[5][1]:
                return "Right"
            else:
                return "Left"

def main():
    pTime = 0
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    detector = HandDetector(detectionCon=0.7)
    detector.set_connection_drawing_spec((0, 255, 0), 2, 2)
    detector.set_landmark_Drawing_spec((0, 255, 0), 2, 2)
    detector.set_rectangle_drawing_spec((255, 255, 0), 2)
    while cap.isOpened():
        success, img = cap.read()
        if success:
            hand_img = detector.FindMultiHands(img, rectangle=True, multi_rectangle=True)
            #hand_img = detector.FindHandsWithoutHands(img, draw=True, rectangle=True, multi_rectangle=False)
            lmList, RbBox, LbBox = detector.FindAllPosition(img)
            handType=""
            if lmList and (RbBox or LbBox):
                if RbBox and LbBox:
                    xmin = min(LbBox[0], RbBox[0])
                    ymin = min(LbBox[1], RbBox[1])
                    xmax = max(LbBox[2], RbBox[2])

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