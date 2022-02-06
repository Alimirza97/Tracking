import cv2
import mediapipe as mp
import numpy as np
import time

class HolisticTracking():
    def __init__(self,
                 static_image_mode=False,
                 upper_body_only=False,
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.upper_body_only = upper_body_only
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(self.static_image_mode, self.upper_body_only, self.smooth_landmarks, self.min_detection_confidence, self.min_tracking_confidence)

        self.landmark_drawing_spec_left_hand = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        self.connection_drawing_spec_left_hand = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

        self.landmark_drawing_spec_right_hand = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        self.connection_drawing_spec_right_hand = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

        self.landmark_drawing_spec_face = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        self.connection_drawing_spec_face = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

        self.landmark_drawing_spec_pose = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        self.connection_drawing_spec_pose = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

#------------------------------------------------------------Left Hand---------------------------------------------------------------------
    def set_landmark_drawing_spec_left_hand(self, color=(0, 255, 0), thickness=2, circle_radius=2):
        self.landmark_drawing_spec_left_hand = self.mp_drawing.DrawingSpec(color=color, thickness=thickness, circle_radius=circle_radius)

    def set_connection_drawing_spec_left_hand(self, color=(0, 255, 0), thickness=2, circle_radius=2):
        self.connection_drawing_spec_left_hand = self.mp_drawing.DrawingSpec(color=color, thickness=thickness, circle_radius=circle_radius)
#-------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------Right Hand---------------------------------------------------------------------
    def set_landmark_drawing_spec_right_hand(self, color=(0, 255, 0), thickness=2, circle_radius=2):
        self.landmark_drawing_spec_right_hand = self.mp_drawing.DrawingSpec(color=color, thickness=thickness, circle_radius=circle_radius)

    def set_connection_drawing_spec_right_hand(self, color=(0, 255, 0), thickness=2, circle_radius=2):
        self.connection_drawing_spec_right_hand = self.mp_drawing.DrawingSpec(color=color, thickness=thickness, circle_radius=circle_radius)
#-------------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------Face------------------------------------------------------------------------
    def set_landmark_drawing_spec_face(self, color=(0, 255, 0), thickness=1, circle_radius=1):
        self.landmark_drawing_spec_face = self.mp_drawing.DrawingSpec(color=color, thickness=thickness, circle_radius=circle_radius)

    def set_connection_drawing_spec_face(self, color=(0, 255, 0), thickness=1, circle_radius=1):
        self.connection_drawing_spec_face = self.mp_drawing.DrawingSpec(color=color, thickness=thickness, circle_radius=circle_radius)
#-------------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------Pose------------------------------------------------------------------------
    def set_landmark_drawing_spec_pose(self, color=(0, 255, 0), thickness=2, circle_radius=2):
        self.landmark_drawing_spec_pose = self.mp_drawing.DrawingSpec(color=color, thickness=thickness, circle_radius=circle_radius)

    def set_connection_drawing_spec_pose(self, color=(0, 255, 0), thickness=2, circle_radius=2):
        self.connection_drawing_spec_pose = self.mp_drawing.DrawingSpec(color=color, thickness=thickness, circle_radius=circle_radius)
#-------------------------------------------------------------------------------------------------------------------------------------------

    def FindLeftHand(self, img, draw=True, rectangle=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.holistic.process(imgRGB)
        if self.results.left_hand_landmarks:
            if draw:
                self.mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=self.results.left_hand_landmarks,
                    connections=self.mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.landmark_drawing_spec_left_hand,
                    connection_drawing_spec=self.connection_drawing_spec_left_hand
                )
                if rectangle:
                    lmList, bBox = self.FindLeftHandPosition(img)
                    cv2.rectangle(img, (bBox[0], bBox[1]), (bBox[2], bBox[3]), self.landmark_drawing_spec_left_hand.color, self.landmark_drawing_spec_left_hand.thickness)
        return img

    def FindRightHand(self, img, draw=True, rectangle=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.holistic.process(imgRGB)
        if self.results.right_hand_landmarks:
            if draw:
                self.mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=self.results.right_hand_landmarks,
                    connections=self.mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.landmark_drawing_spec_right_hand,
                    connection_drawing_spec=self.connection_drawing_spec_right_hand
                )
                if rectangle:
                    lmList, bBox = self.FindRightHandPosition(img)
                    cv2.rectangle(img, (bBox[0], bBox[1]), (bBox[2], bBox[3]), self.landmark_drawing_spec_right_hand.color, self.landmark_drawing_spec_right_hand.thickness)
        return img

    def FindFace(self, img, draw=True, rectangle=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.holistic.process(imgRGB)
        if self.results.face_landmarks:
            if draw:
                self.mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=self.results.face_landmarks,
                    connections=self.mp_holistic.FACE_CONNECTIONS,
                    landmark_drawing_spec=self.landmark_drawing_spec_face,
                    connection_drawing_spec=self.connection_drawing_spec_face
                )
                if rectangle:
                    lmList, bBox = self.FindFacePosition(img)
                    cv2.rectangle(img, (bBox[0], bBox[1]), (bBox[2], bBox[3]), self.landmark_drawing_spec_face.color, self.landmark_drawing_spec_face.thickness)
        return img

    def FindPose(self, img, draw=True, rectangle=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.holistic.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=self.results.pose_landmarks,
                    connections=self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.landmark_drawing_spec_pose,
                    connection_drawing_spec=self.connection_drawing_spec_pose
                )
                if rectangle:
                    lmList, bBox = self.FindPosePosition(img)
                    cv2.rectangle(img, (bBox[0], bBox[1]), (bBox[2], bBox[3]), self.landmark_drawing_spec_pose.color, self.landmark_drawing_spec_pose.thickness)
        return img

    def AllTracking(self, img, draw=True, face=True, face_rectangle=False, left_hand=True, left_hand_rectangle=False, right_hand=True, right_hand_rectangle=False, pose=True, pose_rectangle=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.holistic.process(imgRGB)
        if self.results.face_landmarks or self.results.pose_landmarks or self.results.right_hand_landmarks or self.results.left_hand_landmarks:
            if draw:
                if pose:
                    self.mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=self.results.pose_landmarks,
                        connections=self.mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.landmark_drawing_spec_pose,
                        connection_drawing_spec=self.connection_drawing_spec_pose
                    )
                    if pose_rectangle:
                        lmList, bBox = self.FindPosePosition(img)
                        if lmList and bBox:
                            cv2.rectangle(img, (bBox[0], bBox[1]), (bBox[2], bBox[3]),
                                          self.landmark_drawing_spec_pose.color, self.landmark_drawing_spec_pose.thickness)
                if face:
                    self.mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=self.results.face_landmarks,
                        connections=self.mp_holistic.FACE_CONNECTIONS,
                        landmark_drawing_spec=self.landmark_drawing_spec_face,
                        connection_drawing_spec=self.connection_drawing_spec_face
                    )
                    if face_rectangle:
                        lmList, bBox = self.FindFacePosition(img)
                        if lmList and bBox:
                            cv2.rectangle(img, (bBox[0], bBox[1]), (bBox[2], bBox[3]),
                                          self.landmark_drawing_spec_face.color, self.landmark_drawing_spec_face.thickness)
                if right_hand:
                    self.mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=self.results.right_hand_landmarks,
                        connections=self.mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.landmark_drawing_spec_right_hand,
                        connection_drawing_spec=self.connection_drawing_spec_right_hand
                    )
                    if right_hand_rectangle:
                        lmList, bBox = self.FindRightHandPosition(img)
                        if lmList and bBox:
                            cv2.rectangle(img, (bBox[0], bBox[1]), (bBox[2], bBox[3]),
                                          self.landmark_drawing_spec_right_hand.color, self.landmark_drawing_spec_right_hand.thickness)
                if left_hand:
                    self.mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=self.results.left_hand_landmarks,
                        connections=self.mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.landmark_drawing_spec_left_hand,
                        connection_drawing_spec=self.connection_drawing_spec_left_hand
                    )
                    if left_hand_rectangle:
                        lmList, bBox = self.FindLeftHandPosition(img)
                        if lmList and bBox:
                            cv2.rectangle(img, (bBox[0], bBox[1]), (bBox[2], bBox[3]),
                                          self.landmark_drawing_spec_left_hand.color, self.landmark_drawing_spec_left_hand.thickness)
        return img

    def AllTrackingWithoutBackground(self, img, draw=True, face=True, face_rectangle=False,
                                     left_hand=True, left_hand_rectangle=False, right_hand=True,
                                     right_hand_rectangle=False, pose=True, pose_rectangle=False, white=False):
        img = self.AllTracking(img, draw=draw, face=face, face_rectangle=face_rectangle, left_hand=left_hand,
                               left_hand_rectangle=left_hand_rectangle, right_hand=right_hand,
                               right_hand_rectangle=right_hand_rectangle, pose=pose, pose_rectangle=pose_rectangle)
        if white:
            Wmask = (img[:, :, 0:3] == [0, 255, 0]).all(2)
            Wmask = (Wmask * 255).astype(np.uint8)
            return Wmask
        else:
            img[np.where((img != [0, 0, 0]).all(axis=2))] = [0, 0, 0]
            return img

    def FindLeftHandCoordinate(self):
        lmList = []
        if self.results.left_hand_landmarks:
            myLeftHand = self.results.left_hand_landmarks
            myLeftHand = str(myLeftHand)
            list = myLeftHand.split('landmark')
            for i in range(1, len(list)):
                degisken = list[i].split('{')[1].split('}')[0].split(':')
                x = degisken[1].split(' ')[1].split('\n')[0].split(' ')[0]
                y = degisken[2].split(' ')[1].split('\n')[0].split(' ')[0]
                z = degisken[3].split(' ')[1].split('\n')[0].split(' ')[0]
                lmList.append([i - 1, float(x), float(y), float(z)])
        return lmList

    def FindLeftHandPosition(self, img):
        xList, yList, lmList, bBox = [], [], [], []
        list = self.FindLeftHandCoordinate()
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

    def FindRightHandCoordinate(self):
        lmList = []
        if self.results.right_hand_landmarks:
            myRightHand = self.results.right_hand_landmarks
            myRightHand = str(myRightHand)
            list = myRightHand.split('landmark')
            for i in range(1, len(list)):
                degisken = list[i].split('{')[1].split('}')[0].split(':')
                x = degisken[1].split(' ')[1].split('\n')[0].split(' ')[0]
                y = degisken[2].split(' ')[1].split('\n')[0].split(' ')[0]
                z = degisken[3].split(' ')[1].split('\n')[0].split(' ')[0]
                lmList.append([i - 1, float(x), float(y), float(z)])
        return lmList

    def FindRightHandPosition(self, img):
        xList, yList, lmList, bBox = [], [], [], []
        list = self.FindRightHandCoordinate()
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

    def FindFaceCoordinate(self):
        lmList = []
        if self.results.face_landmarks:
            myFace = self.results.face_landmarks
            myFace = str(myFace)
            list = myFace.split('landmark')
            for i in range(1, len(list)):
                degisken = list[i].split('{')[1].split('}')[0].split(':')
                x = degisken[1].split(' ')[1].split('\n')[0].split(' ')[0]
                y = degisken[2].split(' ')[1].split('\n')[0].split(' ')[0]
                z = degisken[3].split(' ')[1].split('\n')[0].split(' ')[0]
                lmList.append([i - 1, float(x), float(y), float(z)])
        return lmList

    def FindFacePosition(self, img):
        xList, yList, lmList, bBox = [], [], [], []
        list = self.FindFaceCoordinate()
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

    def FindPoseCoordinate(self):
        lmList = []
        if self.results.pose_landmarks:
            myPose = self.results.pose_landmarks
            myPose = str(myPose)
            list = myPose.split('landmark')
            for i in range(1, len(list)):
                degisken = list[i].split('{')[1].split('}')[0].split(':')
                x = degisken[1].split(' ')[1].split('\n')[0].split(' ')[0]
                y = degisken[2].split(' ')[1].split('\n')[0].split(' ')[0]
                z = degisken[3].split(' ')[1].split('\n')[0].split(' ')[0]
                lmList.append([i - 1, float(x), float(y), float(z)])
        return lmList

    def FindPosePosition(self, img):
        lmList = []
        list = self.FindPoseCoordinate()
        if list:
            h, w, c = img.shape
            for item in list:
                cx, cy = int(item[1] * w), int(item[2] * h)
                lmList.append([item[0], cx, cy])
        return lmList


def main():
        pTime = 0
        wCam, hCam = 640, 480
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
        detector = HolisticTracking(min_detection_confidence=0.7)
        detector.set_landmark_drawing_spec_left_hand((0, 255, 0), 1, 1)
        detector.set_connection_drawing_spec_left_hand((0, 255, 0), 1, 1)
        detector.set_landmark_drawing_spec_right_hand((255, 255, 0), 1, 1)
        detector.set_connection_drawing_spec_right_hand((255, 255, 0), 1, 1)
        detector.set_landmark_drawing_spec_face((255, 0, 255), 1, 1)
        detector.set_connection_drawing_spec_face((255, 0, 255), 1, 1)
        #detector.set_landmark_drawing_spec_pose((0, 255, 255), 1, 1)
        #detector.set_connection_drawing_spec_pose((0, 255, 255), 1, 1)
        while cap.isOpened():
            success, img = cap.read()
            if success:
                #img = detector.AllTracking(img, pose=False)
                img = detector.AllTrackingWithoutBackground(img, pose=False)
                """
                lmlistLeft, LbBox = detector.FindLeftHandPosition(img)
                lmlistRight, RbBox = detector.FindRightHandPosition(img)
                lmlistFace, faceBbox = detector.FindFacePosition(img)
                lmlistPose, poseBbox = detector.FindPosePosition(img)
                
                xmin, ymin, xmax, ymax = -1, -1, -1, -1
                if LbBox and RbBox and faceBbox:
                    xmin = min(LbBox[0], RbBox[0], faceBbox[0])
                    ymin = min(LbBox[1], RbBox[1], faceBbox[1])
                    xmax = max(LbBox[2], RbBox[2], faceBbox[2])
                    ymax = max(LbBox[3], RbBox[3], faceBbox[3])
                elif LbBox and faceBbox:
                    xmin = min(LbBox[0], faceBbox[0])
                    ymin = min(LbBox[1], faceBbox[1])
                    xmax = max(LbBox[2], faceBbox[2])
                    ymax = max(LbBox[3], faceBbox[3])
                elif RbBox and faceBbox:
                    xmin = min(RbBox[0], faceBbox[0])
                    ymin = min(RbBox[1], faceBbox[1])
                    xmax = max(RbBox[2], faceBbox[2])
                    ymax = max(RbBox[3], faceBbox[3])
                elif LbBox and RbBox:
                    xmin = min(LbBox[0], RbBox[0])
                    ymin = min(LbBox[1], RbBox[1])
                    xmax = max(LbBox[2], RbBox[2])
                    ymax = max(LbBox[3], RbBox[3])
                elif LbBox:
                    xmin = LbBox[0]
                    ymin = LbBox[1]
                    xmax = LbBox[2]
                    ymax = LbBox[3]
                elif RbBox:
                    xmin = RbBox[0]
                    ymin = RbBox[1]
                    xmax = RbBox[2]
                    ymax = RbBox[3]
    
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                
                if lmlistLeft:
                    cv2.rectangle(img, (LbBox[0], LbBox[1]), (LbBox[2], LbBox[3]), (255, 0, 0), 2)
                    #for lm in lmlistLeft:
                        #cv2.circle(img, (lm[1], lm[2]), 5, (255, 0, 255), cv2.FILLED)
                    #print(lmlistLeft)
                if lmlistRight:
                    cv2.rectangle(img, (RbBox[0], RbBox[1]), (RbBox[2], RbBox[3]), (0, 0, 255), 2)
                    #for lm in lmlistRight:
                        #cv2.circle(img, (lm[1], lm[2]), 5, (255, 255, 0), cv2.FILLED)
                    #print(lmlistRight)
                
                if lmlistFace:
                    for lm in lmlistFace:
                        cv2.circle(img, (lm[1], lm[2]), 1, (0, 255, 255), cv2.FILLED)
                    #print(lmlistFace)
                if lmlistPose:
                    for lm in lmlistPose:
                        cv2.circle(img, (lm[1], lm[2]), 15, (0, 255, 255), cv2.FILLED)
                    #print(lmlistPose)
                """
                #body = detector.AllTrackingWithoutBackground(img, draw=False)
                """
                detector.set_landmark_drawing_spec_left_hand((255, 0, 0), 2, 2)
                detector.set_connection_drawing_spec_left_hand((255, 0, 0), 2, 2)

                detector.set_landmark_drawing_spec_right_hand((255, 0, 255), 2, 2)
                detector.set_connection_drawing_spec_right_hand((255, 0, 255), 2, 2)

                detector.set_landmark_drawing_spec_face((255, 255, 0), 2, 2)
                detector.set_connection_drawing_spec_face((255, 255, 0), 2, 2)

                detector.set_landmark_drawing_spec_pose((0, 0, 255), 2, 2)
                detector.set_connection_drawing_spec_pose((0, 0, 255), 2, 2)
                """
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