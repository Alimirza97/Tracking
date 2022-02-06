import cv2
import HandTrackingModel as htm
import os

class HandTrackingImage():
    def __init__(self, inputDir = "", outputDir = "", extention = ".png"):
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.extention = extention

        self.detector = htm.HandDetector(mode=True, detectionCon=0.7)
        self.CreateList()

    def CreateList(self):
        self.fileList = []
        print(self.inputDir)
        for root, dirs, files in os.walk(str(self.inputDir), topdown=False):
            for name in files:
                if name.endswith(self.extention):
                    fullName = self.inputDir + name
                    print(fullName)
                    self.fileList.append(fullName)

    def SaveFilesWithHand(self):
        for file in self.fileList:
            img = cv2.imread(file, 1)
            img = self.detector.FindMultiHands(img)
            tmp = file.split("/")
            fileName = tmp[len(tmp) - 1].split(".")[0]
            full_name = self.outputDir + fileName + "_hd.png"
            cv2.imwrite(full_name, img)

    def SaveFilesWithoutHand(self, green=True):
        for file in self.fileList:
            img = cv2.imread(file, 1)
            green_img = self.detector.FindHandsWithoutHands(img)
            white_img = self.detector.FindHandsWithoutHands(img, green=False)
            tmp = file.split("/")
            fileName = tmp[len(tmp) - 1].split(".")[0]
            if green:
                full_name = self.outputDir + fileName + "_hd_green.png"
                cv2.imwrite(full_name, green_img)
            else:
                full_name = self.outputDir + fileName + "_hd_white.png"
                cv2.imwrite(full_name, white_img)

    def DetectionFilesWithHand(self):
        for file in self.fileList:
            img = cv2.imread(file, 1)
            img = self.detector.FindMultiHands(img)
            tmp = file.split("/")
            fileName = tmp[len(tmp) - 1].split(".")[0]
            full_name = fileName + "_hd"
            cv2.imshow(full_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def DetectionFilesWithoutHand(self, green=True):
        for file in self.fileList:
            img = cv2.imread(file, 1)
            img = self.detector.FindMultiHands(img)
            green_img = self.detector.FindHandsWithoutHands(img)
            white_img = self.detector.FindHandsWithoutHands(img, green=False)
            tmp = file.split("/")
            fileName = tmp[len(tmp) - 1].split(".")[0]
            full_name = fileName + "_hd"
            if green:
                cv2.imshow(full_name, green_img)
            else:
                cv2.imshow(full_name, white_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    inDir = 'C:/Users/Elimirze/Videos/iVCam/'
    outDir = 'C:/Users/Elimirze/Videos/iVCam/'
    ext = ".jpg"
    HDFI = HandTrackingImage(inputDir=inDir, outputDir=outDir, extention = ext)
    HDFI.SaveFilesWithoutHand()

if __name__ == "__main__":
    main()