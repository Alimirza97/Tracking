import cv2
import mediapipe as mp
import numpy as np
import time

class PoseDetection():
    def __init__(self, static_image_mode=False,
                 upper_body_only=False,
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.upper_body_only = upper_body_only
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.results = tuple()
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.static_image_mode,
                                     upper_body_only=self.upper_body_only,
                                     smooth_landmarks=self.smooth_landmarks,
                                     min_detection_confidence=self.min_detection_confidence,
                                     min_tracking_confidence=self.min_tracking_confidence)

        self.landmark_drawing_spec_pose = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        self.connection_drawing_spec_pose = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)


    def set_landmark_drawing_spec_pose(self, color=(0, 255, 0), thickness=2, circle_radius=2):
        self.landmark_drawing_spec_pose = self.mpDraw.DrawingSpec(color=color, thickness=thickness, circle_radius=circle_radius)

    def set_connection_drawing_spec_pose(self, color=(0, 255, 0), thickness=2, circle_radius=2):
        self.connection_drawing_spec_pose = self.mpDraw.DrawingSpec(color=color, thickness=thickness, circle_radius=circle_radius)

    def FindPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    image=img,
                    landmark_list=self.results.pose_landmarks,
                    connections=self.mpPose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.landmark_drawing_spec_pose,
                    connection_drawing_spec=self.connection_drawing_spec_pose
                )
        return img

    def FindPoseWithoutBody(self, img, draw=True, white=False):
        img = self.FindPose(img, draw=draw)
        if white:
            Wmask = (img[:, :, 0:3] == [0, 255, 0]).all(2)
            Wmask = (Wmask * 255).astype(np.uint8)
            return Wmask
        else:
            img[np.where((img != [0, 0, 0]).all(axis=2))] = [0, 0, 0]
            return img

    def FindPosition(self, img):
        lmList = []
        list = self.FindCoordinate()
        if list:
            for item in list:
                h, w, c = img.shape
                cx, cy = int(item[1] * w), int(item[2] * h)
                lmList.append([item[0], cx, cy])
        return lmList

    def FindCoordinate(self):
        lmList = []
        if self.results:
            if self.results.pose_landmarks:
                for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    lmList.append([id, lm.x, lm.y, lm.z])
        return lmList

def main():
    pTime = 0
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    detector = PoseDetection(min_detection_confidence=0.7)
    while cap.isOpened():
        success, img = cap.read()
        if success:
            img = detector.FindPose(img)
            lmlist = detector.FindPosition(img)
            print(lmlist)
            #body = detector.FindPoseWithoutBody(img, draw=False)

            detector.set_connection_drawing_spec_pose((255, 0, 0), 2, 2)
            detector.set_landmark_drawing_spec_pose((255, 0, 0), 2, 2)
            """
            lmList = detector.FindPosition(img)
            if lmList:
                print(lmList)
            """
            # -------------------------------------FPS Calculate-----------------------------------------
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'{int(fps)}', (580, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
            # -------------------------------------------------------------------------------------------
            cv2.imshow("Pose Detection", img)
            #cv2.imshow("Pose Detection Without Body", body)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print("Unsuccess")

    print("End")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()