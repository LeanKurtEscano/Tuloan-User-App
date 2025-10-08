import cv2

class AntiSpoof:
    def __init__(self):
        pass

    def detect_blink_opencv(image_path: str):
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

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            faces_list = list(faces) if len(faces) > 0 else []

            if len(faces_list) == 0:
                print("  ❌ No face detected")
                return False, True, 0.0, 0.0, 0

            (x, y, w, h) = max(faces_list, key=lambda face: face[2] * face[3])
            x, y, w, h = int(x), int(y), int(w), int(h)

            print(f"  ✅ Face detected at ({x},{y}) size {w}x{h}")

            roi_y_end = int(y + h * 0.6)
            roi_gray = gray[y:roi_y_end, x:x+w]

            eyes_list = []

            # Strategy 1: Very lenient detection
            eyes1 = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=2, minSize=(10, 10))
            if len(eyes1) > 0:
                eyes_list = list(eyes1)
                print(f"  👁️ Strategy 1: Found {len(eyes_list)} eyes")

            # Strategy 2: Alternative parameters
            if len(eyes_list) == 0:
                eyes2 = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
                if len(eyes2) > 0:
                    eyes_list = list(eyes2)
                    print(f"  👁️ Strategy 2: Found {len(eyes_list)} eyes")

            # Strategy 3: Enhanced with histogram equalization
            if len(eyes_list) == 0:
                roi_enhanced = cv2.equalizeHist(roi_gray)
                eyes3 = eye_cascade.detectMultiScale(roi_enhanced, scaleFactor=1.05, minNeighbors=2, minSize=(10, 10))
                if len(eyes3) > 0:
                    eyes_list = list(eyes3)
                    print(f"  👁️ Strategy 3 (enhanced): Found {len(eyes_list)} eyes")

            num_eyes = len(eyes_list)
            print(f"  👁️ Total eyes detected: {num_eyes}")

            if num_eyes >= 2:
                eyes_sorted = sorted(eyes_list, key=lambda e: e[0])
                left_eye = eyes_sorted[0]
                right_eye = eyes_sorted[1]

                left_ratio = float(left_eye[3]) / float(left_eye[2]) if left_eye[2] > 0 else 0.0
                right_ratio = float(right_eye[3]) / float(right_eye[2]) if right_eye[2] > 0 else 0.0

                print(f"  👁️ Both eyes visible - Ratios: L={left_ratio:.3f}, R={right_ratio:.3f}")
                return True, True, left_ratio, right_ratio, num_eyes

            elif num_eyes == 1:
                eye = eyes_list[0]
                ratio = float(eye[3]) / float(eye[2]) if eye[2] > 0 else 0.0
                print(f"  ⚠️ Only 1 eye detected - Ratio: {ratio:.3f} - marking as CLOSED")
                return True, False, ratio, ratio, num_eyes

            else:
                print(f"  🔴 NO EYES DETECTED - Eyes are CLOSED!")
                return True, False, 0.0, 0.0, num_eyes

        except Exception as e:
            print(f"  ❌ Error in blink detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, True, 0.0, 0.0, 0



def detect_head_pose(image_path: str):
    """
    Detect head pose and determine if the user is showing their left or right facial profile.
    Returns:
        (face_detected, head_direction, face_area, eye_count, is_frontal)
        head_direction ∈ {"frontal", "left_profile", "right_profile", "slight_turn"}
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Could not read image")
            return False, "unknown", 0, 0, False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Detect frontal faces with more lenient parameters
        frontal_faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,  # More sensitive
            minNeighbors=4,   # Lower threshold
            minSize=(30, 30)
        )
        
        # Detect profile faces - check both normal and flipped for left/right
        profile_faces_right = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )
        
        # Flip image to detect left profiles
        flipped_gray = cv2.flip(gray, 1)
        profile_faces_left = profile_cascade.detectMultiScale(
            flipped_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )

        # Determine which detection to use
        all_detections = []
        
        if len(frontal_faces) > 0:
            face = max(frontal_faces, key=lambda f: f[2] * f[3])
            all_detections.append(("frontal", face, face[2] * face[3]))
        
        if len(profile_faces_right) > 0:
            face = max(profile_faces_right, key=lambda f: f[2] * f[3])
            all_detections.append(("right_profile", face, face[2] * face[3]))
        
        if len(profile_faces_left) > 0:
            face = max(profile_faces_left, key=lambda f: f[2] * f[3])
            # Convert coordinates back from flipped image
            x, y, w, h = face
            face = (width - x - w, y, w, h)
            all_detections.append(("left_profile", face, face[2] * face[3]))

        if not all_detections:
            print("❌ No face detected")
            return False, "unknown", 0, 0, False

        # Use the largest detection
        face_type, (x, y, w, h), face_area = max(all_detections, key=lambda d: d[2])
        print(f"✅ {face_type.replace('_', ' ').title()} detected - Area: {face_area}")

        # Crop ROI for eyes (upper 60% of face)
        x, y, w, h = int(x), int(y), int(w), int(h)
        roi_y_end = int(y + h * 0.6)
        roi_gray = gray[y:roi_y_end, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(15, 15)
        )
        eye_count = len(eyes)
        print(f"👁️ Eyes detected: {eye_count}")

        # Analyze face position in frame for additional profile detection
        face_center_x = x + w / 2
        image_center_x = width / 2
        shift_ratio = (face_center_x - image_center_x) / (width / 2)

        # Determine head direction with multi-factor analysis
        is_frontal = False
        head_direction = "unknown"

        if eye_count >= 2 and face_type == "frontal":
            # Both eyes visible and frontal detector triggered
            head_direction = "frontal"
            is_frontal = True
            
        elif face_type == "left_profile" or (eye_count <= 1 and shift_ratio > 0.15):
            # Left profile detector OR single eye with rightward shift
            head_direction = "left_profile"
            is_frontal = False
            
        elif face_type == "right_profile" or (eye_count <= 1 and shift_ratio < -0.15):
            # Right profile detector OR single eye with leftward shift
            head_direction = "right_profile"
            is_frontal = False
            
        elif eye_count == 1:
            # Single eye detected but not clear profile - use position
            if shift_ratio > 0.05:
                head_direction = "left_profile"
            elif shift_ratio < -0.05:
                head_direction = "right_profile"
            else:
                head_direction = "slight_turn"
            is_frontal = False
            
        elif face_type == "frontal":
            # Frontal detected but not both eyes - might be slight turn
            if abs(shift_ratio) > 0.1:
                head_direction = "slight_turn"
            else:
                head_direction = "frontal"
            is_frontal = False
        else:
            head_direction = "slight_turn"
            is_frontal = False

        print(f"🧠 Head Direction: {head_direction} (shift: {shift_ratio:.2f})")

        # For API compatibility: convert to boolean is_profile
        is_profile = head_direction in ["left_profile", "right_profile"]

        return True, is_profile, int(face_area), int(eye_count), is_frontal

    except Exception as e:
        print(f"❌ Error in pose detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, "unknown", 0, 0, False