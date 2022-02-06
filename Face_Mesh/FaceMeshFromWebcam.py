import cv2
import FaceMeshModel as fm
import time

def main():
  pTime = 0
  wCam, hCam = 640, 480
  cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
  detector = fm.FaceDetector(detectionCon=0.7)
  while cap.isOpened():
    success, img = cap.read()
    if success == True:

      cv2.imshow("Img", img)

      img = detector.FindFace(img)
      """
      lmlistPosition = detector.FindPosition(img)
      if lmlistPosition:
        for lm in lmlistPosition:
          cv2.circle(img, (lm[1], lm[2]), 1, (0, 255, 0), cv2.FILLED)
        # print(lmlistPosition)
      """
      # -----------FPS Calculate-----------------------------------------------------
      cTime = time.time()
      fps = 1 / (cTime - pTime)
      pTime = cTime
      # -----------------------------------------------------------------------------
      cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
      cv2.imshow("Face Mesh Image", img)
      mask = detector.FindFaceWithoutFace(img)
      res = detector.FindFaceWithoutFace(img, green = False)
      cv2.imshow("Face Mesh Green", mask)
      cv2.imshow("Face Mesh White", res)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
      print("Unsuccessful")
  print("End")
  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()