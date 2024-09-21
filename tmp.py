import cv2

path = "/Users/personal/Desktop/cogVideoExperiments/2Btests/prompt_134_seed_713626_16fps.mp4"

cap = cv2.VideoCapture(path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
ret, frame = cap.read()
cv2.imwrite("tmpMonk.jpg", frame)
cap.release()
