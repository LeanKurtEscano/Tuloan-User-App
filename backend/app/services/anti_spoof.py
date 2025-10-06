import cv2

class AntiSpoof:
    def __init__(self):
        pass

    def detect_blink_opencv(self, image_path: str):
        """
        Detect blinks using OpenCV's Haar Cascade for eyes.
        Returns: (face_detected, eyes_open, left_ear, right_ear, num_eyes_detected)
        """
        try:
        
            img = cv2.imread(image_path)
            if img is None:
                print("  ❌ Could not read image")
                return False, True, 0.0, 0.0, 0

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            faces_list = list(faces) if len(faces) > 0 else []

            if len(faces_list) == 0:
                print("  ❌ No face detected")
                return False, True, 0.0, 0.0, 0

            # Get the largest face
            (x, y, w, h) = max(faces_list, key=lambda face: face[2] * face[3])
            x, y, w, h = int(x), int(y), int(w), int(h)
        

            # Region of interest for eyes (upper half of face)
            roi_y_end = int(y + h * 0.6)
            roi_gray = gray[y:roi_y_end, x:x+w]

            # Try multiple detection strategies
            eyes_list = []

            # Strategy 1: Very lenient detection
            eyes1 = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,
                minNeighbors=2,
                minSize=(10, 10)
            )
            if len(eyes1) > 0:
                eyes_list = list(eyes1)
             

            # Strategy 2: Alternative parameters if first fails
            if len(eyes_list) == 0:
                eyes2 = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(20, 20)
                )
                if len(eyes2) > 0:
                    eyes_list = list(eyes2)
       
            # Strategy 3: Histogram equalization
            if len(eyes_list) == 0:
                roi_enhanced = cv2.equalizeHist(roi_gray)
                eyes3 = eye_cascade.detectMultiScale(
                    roi_enhanced,
                    scaleFactor=1.05,
                    minNeighbors=2,
                    minSize=(10, 10)
                )
                if len(eyes3) > 0:
                    eyes_list = list(eyes3)
                   

            num_eyes = len(eyes_list)
           

            # Blink logic based on number of detected eyes
            if num_eyes >= 2:
                eyes_sorted = sorted(eyes_list, key=lambda e: e[0])
                left_eye = eyes_sorted[0]
                right_eye = eyes_sorted[1]

                left_ratio = float(left_eye[3]) / float(left_eye[2]) if left_eye[2] > 0 else 0.0
                right_ratio = float(right_eye[3]) / float(right_eye[2]) if right_eye[2] > 0 else 0.0

                print(f"  👁️ Both eyes visible - Ratios: L={left_ratio:.3f}, R={right_ratio:.3f}")
                return True, True, left_ratio, right_ratio, num_eyes  # Eyes open

            elif num_eyes == 1:
                eye = eyes_list[0]
                ratio = float(eye[3]) / float(eye[2]) if eye[2] > 0 else 0.0
                print(f"  ⚠️ Only 1 eye detected - Ratio: {ratio:.3f} - marking as CLOSED")
                return True, False, ratio, ratio, num_eyes  # Eyes closed

            else:
                print(f"  🔴 NO EYES DETECTED - Eyes are CLOSED!")
                return True, False, 0.0, 0.0, num_eyes  # Eyes closed

        except Exception as e:
            print(f"  ❌ Error in blink detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, True, 0.0, 0.0, 0
