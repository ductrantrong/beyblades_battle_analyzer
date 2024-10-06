import cv2

cap = cv2.VideoCapture("battle.mp4")
cnt = 0

while True:
    end_signal, frame = cap.read()

    if end_signal == False:
        break

    if cnt >= 80 and cnt % 10 == 0:
        cv2.imwrite(f"data/{cnt}.png", frame)

    cnt += 1
