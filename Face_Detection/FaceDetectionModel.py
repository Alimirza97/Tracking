import cv2
import time
import mediapipe as mp

class FaceDetection():
    def __init__(self,  min_detection_confidence=0.5):
        self.min_detection_confidence = min_detection_confidence
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.min_detection_confidence)
        self.drawing_color = (255, 0, 255)
        self.drawing_thickness = 2
        self.results = tuple()

    def SetDrawingSettings(self, color=(255, 0, 255), thickness=2):
        self.drawing_color = color
        self.drawing_thickness = thickness

    def FindFace(self, img, draw=True, custom_rectangle=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                if draw:
                    if custom_rectangle:
                        bBoxs = self.FindPosition(img)
                        for bBox in bBoxs:
                            img = self.__fancyDraw(img, bBox)
                    else:
                        self.mpDraw.draw_detection(img, detection)
        return img

    def __fancyDraw(self, img, bbox):
        x, y ,w, h = bbox[2]
        x1, y1 = x + w, y + h
        len = int(w / 5)
        # Rectangle
        cv2.rectangle(img, bbox[2], self.drawing_color, self.drawing_thickness)
        # Top Left x, y
        cv2.line(img, (x, y), (x + len, y), self.drawing_color, self.drawing_thickness + 4)
        cv2.line(img, (x, y), (x, y + len), self.drawing_color, self.drawing_thickness + 4)
        # Top Right x1, y
        cv2.line(img, (x1, y), (x1 - len, y), self.drawing_color, self.drawing_thickness + 4)
        cv2.line(img, (x1, y), (x1, y + len), self.drawing_color, self.drawing_thickness + 4)
        # Bottom Left x, y1
        cv2.line(img, (x, y1), (x + len, y1), self.drawing_color, self.drawing_thickness + 4)
        cv2.line(img, (x, y1), (x, y1 - len), self.drawing_color, self.drawing_thickness + 4)
        # Bottom Right x1, y1
        cv2.line(img, (x1, y1), (x1 - len, y1), self.drawing_color, self.drawing_thickness + 4)
        cv2.line(img, (x1, y1), (x1, y1 - len), self.drawing_color, self.drawing_thickness + 4)

        cv2.putText(img, f'{bbox[1]}%', (bbox[2][0], bbox[2][1] - 20),
                    cv2.FONT_HERSHEY_PLAIN, 2, self.drawing_color, self.drawing_thickness)
        return img

    def FindPosition(self, img):
        bBoxs = []
        bBoxsC = self.FindCoordinate()
        if bBoxsC:
            for bboxc in bBoxsC:
                bBoxc = bboxc[2]
                h, w, c = img.shape
                bBox = [int(bBoxc[0] * w), int(bBoxc[1] * h), \
                       int(bBoxc[2] * w), int(bBoxc[3] * h)]
                score = int(bboxc[1] * 100)
                bBoxs.append([bboxc[0], score, bBox])
        return bBoxs

    def FindCoordinate(self):
        bBox = []
        if self.results:
            if self.results.detections:
                for id, detection in enumerate(self.results.detections):
                    bBox.append([detection.label_id[0], detection.score[0],
                                 [
                                     detection.location_data.relative_bounding_box.xmin,
                                     detection.location_data.relative_bounding_box.ymin,
                                     detection.location_data.relative_bounding_box.width,
                                     detection.location_data.relative_bounding_box.height
                                 ]])
        return bBox

def main():
    pTime = 0
    wCam, hCam = 960, 540
    cap = cv2.VideoCapture("../Videos/video.mkv", cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    detection = FaceDetection(min_detection_confidence=0.6)
    detection.SetDrawingSettings((255, 255, 0), 1)
    while cap.isOpened():
        success, img = cap.read()
        if success:
            img = detection.FindFace(img, custom_rectangle=True)
            bbox = detection.FindPosition(img)
            print(bbox)
            bbox = detection.FindCoordinate()
            print(bbox)
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


