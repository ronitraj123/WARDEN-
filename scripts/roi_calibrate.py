# scripts/roi_calibrate.py

import cv2

REFERENCE_IMAGE = r'C:\Users\ronit\OneDrive\Desktop\WARDEN-\datasets\objects\objects_v2\valid\images\frame_0081_jpg.rf.343d48ba0fac2e9e31553b31447e9c61.jpg'  # pick a clear, representative frame
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: ({x}, {y})")
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.line(img, points[-2], points[-1], (0, 255, 0), 2)
        cv2.imshow('Click ROI corners (press q when done)', img)

img = cv2.imread(REFERENCE_IMAGE)
cv2.imshow('Click ROI corners (press q when done)', img)
cv2.setMouseCallback('Click ROI corners (press q when done)', click_event)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("\nFinal polygon points:")
print(points)