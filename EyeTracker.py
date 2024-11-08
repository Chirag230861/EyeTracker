import cv2
import mediapipe as mp
import pyautogui
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
BLINK_THRESHOLD = 0.004
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            if id == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)
        left_eye_top, left_eye_bottom = landmarks[145], landmarks[159]
        right_eye_top, right_eye_bottom = landmarks[374], landmarks[386]
        for landmark in [left_eye_top, left_eye_bottom, right_eye_top, right_eye_bottom]:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        left_blink_distance = left_eye_top.y - left_eye_bottom.y
        right_blink_distance = right_eye_top.y - right_eye_bottom.y
        if left_blink_distance < BLINK_THRESHOLD and right_blink_distance < BLINK_THRESHOLD:
            pyautogui.click(button='left') 
            print("Left-click triggered by both eyes blinking")
            pyautogui.sleep(1)  
    cv2.imshow('Eye Controlled Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
