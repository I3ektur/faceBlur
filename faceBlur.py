import cv2
from cvzone.FaceDetectionModule import FaceDetector

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


detector = FaceDetector(minDetectionCon=0.55)

while True:
    success, img = cap.read(1)
    img, bboxs = detector.findFaces(img, draw=True)

    if bboxs:
        for i, bbox in enumerate(bboxs):
            x, y, w, h = bbox['bbox']
            if x < 0: x = 0
            if y < 0: y = 0
            if i >= 5:  # Limit to 3 windows
                break
            x, y, w, h = bbox['bbox']
            imgCrop = img[y:y+h, x:x+w]
            imgBlur = cv2.blur(imgCrop,(55,55))
            img[y:y+h, x:x+w] = imgBlur
            #cv2.imshow(f'Image Cropped {i}', imgCrop)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
