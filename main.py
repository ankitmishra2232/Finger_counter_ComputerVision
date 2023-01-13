import time
import cv2
import os
import HandTrackingModule as htm
import math

wCam, hCam = 1280,720

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)


folderPath = "fingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList=[]

for impath in myList:
    image = cv2.imread(f'{folderPath}/{impath}')
    # print(f'{folderPath}/{impath}')
    overlayList.append(image)

print(len(overlayList))
pTime =0

detector = htm.handDetector(detectionCon=0.75)

tipIds =[4,8,12,16,20]

while True:
    success, img = cap.read()
    img = detector.findHands(img,draw=False)
    lmList = detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        # print(lmList[8][2]-lmList[6][2])
        fingers =[]
        #thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            # print("index finger is open")
            fingers.append(1)
        else:
            fingers.append(0)

        #four Fingures
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                # print("index finger is open")
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingures = fingers.count(1)
        print(totalFingures)




        h,w,c = overlayList[totalFingures].shape
        img[0:h,0:w] = overlayList[totalFingures]


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (1050, 37), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.putText(img, "use your 'Right' hand", (50, 637), cv2.FONT_HERSHEY_PLAIN, 3,
                (25, 0, 255), 3)

    cv2.imshow("img",img)
    cv2.waitKey(1)