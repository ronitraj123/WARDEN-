# scripts/roi_calibrate.py

import cv2

VIDEO_PATH = r'C:\Users\ronit\OneDrive\Desktop\WARDEN-\datasets\barrier_test_clip.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)
ret, img = cap.read()  # grab first frame, or seek to a specific timestamp for a clearer frame
cap.release()

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: ({x}, {y})")
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.line(img, points[-2], points[-1], (0, 255, 0), 2)
        cv2.imshow('Click ROI corners (press q when done)', img)

cv2.imshow('Click ROI corners (press q when done)', img)
cv2.setMouseCallback('Click ROI corners (press q when done)', click_event)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("\nFinal polygon points:")
print(points)