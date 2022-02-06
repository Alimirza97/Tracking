import cv2
import numpy as np
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, mode=False, numFaces=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.numFaces = numFaces
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.results = tuple()

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.faces = self.mp_face_mesh.FaceMesh(self.mode, self.numFaces, self.detectionCon, self.trackCon)

        self.landmark_drawing_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        self.connection_drawing_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

    def set_landmark_Drawing_spec(self, color=(0, 255, 0), thickness=1, circle_radius=1):
        self.landmark_drawing_spec = self.mp_drawing.DrawingSpec(color=color, thickness=thickness, circle_radius=circle_radius)

    def set_connection_drawing_spec(self, color=(0, 255, 0), thickness=1, circle_radius=1):
        self.connection_drawing_spec = self.mp_drawing.DrawingSpec(color=color, thickness=thickness, circle_radius=circle_radius)


    def FindFace(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False
        self.results = self.faces.process(imgRGB)
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACE_CONNECTIONS,
                        landmark_drawing_spec=self.landmark_drawing_spec,
                        connection_drawing_spec=self.connection_drawing_spec)
        return img

    def FindFaceWithoutFace(self, img, draw=True, white=False):
        img = self.FindFace(img, draw=draw)
        if white:
            Wmask = (img[:, :, 0:3] == [0, 255, 0]).all(2)
            Wmask = (Wmask * 255).astype(np.uint8)
            return Wmask
        else:
            img[np.where((img != [0, 0, 0]).all(axis=2))] = [0, 0, 0]
            return img

    def FindCoordinate(self):
        lmList = []
        if self.results:
            if self.results.multi_face_landmarks:
                myFace = self.results.multi_face_landmarks
                myFace = str(myFace)
                list = myFace.split('landmark')
                for i in range(1, len(list)):
                    degisken = list[i].split('{')[1].split('}')[0].split(':')
                    x = degisken[1].split(' ')[1].split('\n')[0].split(' ')[0]
                    y = degisken[2].split(' ')[1].split('\n')[0].split(' ')[0]
                    z = degisken[3].split(' ')[1].split('\n')[0].split(' ')[0]
                    lmList.append([i-1, float(x), float(y), float(z)])
        return lmList

    def FindPosition(self, img):
        list = self.FindCoordinate()
        lmList = []
        if list:
            h, w, c = img.shape
            for item in list:
                cx, cy = int(item[1] * w), int(item[2] * h)
                lmList.append([item[0], cx, cy])
        return lmList

def main():
    pTime = 0
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    detector = FaceDetector(detectionCon=0.7)
    while cap.isOpened():
        success, img = cap.read()
        if success:
            img = detector.FindFaceWithoutFace(img)
            lmList = detector.FindPosition(img)
            print(lmList)
            detector.set_connection_drawing_spec((255, 255, 0), 2, 2)
            detector.set_landmark_Drawing_spec((255, 0, 0), 2, 2)
            # -----------FPS Calculate-----------------------------------------------------
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            # -----------------------------------------------------------------------------
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow("Face Mesh Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Unsuccessful")
    print("End")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()