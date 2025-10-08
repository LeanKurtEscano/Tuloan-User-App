import cv2

class FaceBlinkDetector:
    def __init__(self):
        # Load cascades once when the class is created
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        print("✅ Haar cascades loaded successfully")

    def detect_blink(self, image_path: str):
        """
        Detect blinks using pre-loaded OpenCV Haar Cascades.
        Returns: (face_detected, eyes_open, left_ear, right_ear, num_eyes_detected)
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                print("❌ Could not read image")
                return False, True, 0.0, 0.0, 0

            # Resize for faster processing
            max_dimension = 640
            h, w = img.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect face
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 4, minSize=(60, 60))
            if len(faces) == 0:
                print("❌ No face detected")
                return False, True, 0.0, 0.0, 0

            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            roi_gray = gray[y:int(y + h * 0.6), x:x+w]
            roi_enhanced = cv2.equalizeHist(roi_gray)

            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(
                roi_enhanced,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(int(w * 0.15), int(h * 0.1)),
                maxSize=(int(w * 0.4), int(h * 0.3))
            )
            num_eyes = len(eyes)

            if num_eyes >= 2:
                eyes_sorted = sorted(eyes, key=lambda e: e[0])[:2]
                left_eye, right_eye = eyes_sorted[0], eyes_sorted[1]
                left_ratio = float(left_eye[3]) / float(left_eye[2])
                right_ratio = float(right_eye[3]) / float(right_eye[2])
                return True, True, left_ratio, right_ratio, num_eyes

            elif num_eyes == 1:
                eye = eyes[0]
                ratio = float(eye[3]) / float(eye[2])
                return True, False, ratio, ratio, num_eyes

            else:
                return True, False, 0.0, 0.0, num_eyes

        except Exception as e:
            print(f"❌ Error in blink detection: {str(e)}")
            return False, True, 0.0, 0.0, 0
