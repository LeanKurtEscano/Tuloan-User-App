from fastapi import FastAPI, UploadFile, File, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deepface import DeepFace
import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict
import time

router = APIRouter()
id_path = None

TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

# Session storage for liveness verification
liveness_sessions: Dict[str, dict] = {}

class LivenessResetRequest(BaseModel):
    session_id: str


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
    Detect head pose (profile/angled view) by checking if a face is detected
    but with reduced frontal characteristics.
    Returns: (face_detected, is_profile, face_area, eye_count, is_frontal)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print("  ❌ Could not read image")
            return False, False, 0, 0, False
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
  
        frontal_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
    
        profile_faces = profile_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_detected = len(frontal_faces) > 0 or len(profile_faces) > 0
        
        if not face_detected:
            print("  ❌ No face detected")
            return False, False, 0, 0, False
        
        # Check if it's a profile view
        is_profile = len(profile_faces) > 0
        
        # Get face area
        if len(frontal_faces) > 0:
            (x, y, w, h) = max(frontal_faces, key=lambda face: face[2] * face[3])
            face_area = w * h
            print(f"  ✅ Frontal face detected - Area: {face_area}")
        else:
            (x, y, w, h) = max(profile_faces, key=lambda face: face[2] * face[3])
            face_area = w * h
            print(f"  📐 Profile face detected - Area: {face_area}")
        
        # Count visible eyes
        x, y, w, h = int(x), int(y), int(w), int(h)
        roi_y_end = int(y + h * 0.6)
        roi_gray = gray[y:roi_y_end, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
        eye_count = len(eyes)
        
        print(f"  👁️ Eyes detected: {eye_count}")
        
        # Determine if facing front (both eyes visible)
        is_frontal = eye_count >= 2
        
        # Profile detection logic:
        # - If profile cascade detected face OR
        # - If frontal face but only 0-1 eyes visible (indicating turned head)
        is_profile = is_profile or (len(frontal_faces) > 0 and eye_count <= 1)
        
        return True, is_profile, int(face_area), int(eye_count), is_frontal
        
    except Exception as e:
        print(f"  ❌ Error in pose detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, False, 0, 0, False


def verify_against_id(live_image_path: str) -> tuple:
    """
    Verify if the person in the live image matches the ID.
    Returns: (is_match, distance, threshold, error_message)
    """
    global id_path
    
    if id_path is None or not os.path.exists(id_path):
        return False, None, None, "ID not uploaded"
    
    try:
        result = DeepFace.verify(
            img1_path=id_path,
            img2_path=live_image_path,
            model_name="ArcFace",
            enforce_detection=True,
            detector_backend="mtcnn"
        )
        
        verified = result["verified"]
        distance = result["distance"]
        threshold = result["threshold"]
        
        print(f"  🔍 ID Verification: {'✅ MATCH' if verified else '❌ NO MATCH'} - Distance: {distance:.4f}, Threshold: {threshold:.4f}")
        
        return verified, distance, threshold, None
        
    except Exception as e:
        error_msg = str(e).lower()
        print(f"  ❌ ID Verification Error: {str(e)}")
        return False, None, None, str(e)


def get_or_create_session(session_id: str) -> dict:
    """Get or create a liveness verification session"""
    if session_id not in liveness_sessions:
        print(f"🆕 Creating new session: {session_id}")
        liveness_sessions[session_id] = {
            "blink_detected": False,
            "left_pose_detected": False,
            "right_pose_detected": False,
            "previous_blink_state": "unknown",
            "previous_left_state": "frontal",
            "previous_right_state": "frontal",
            "last_blink_time": 0,
            "last_closed_time": 0,
            "last_left_time": 0,
            "last_right_time": 0,
            "created_at": time.time(),
            "frame_count": 0,
            "left_frontal_rejected_count": 0,
            "right_frontal_rejected_count": 0
        }
    return liveness_sessions[session_id]


@router.post("/upload-id")
async def upload_id(file: UploadFile = File(...)):
    global id_path
    
    try:
        if not file.content_type.startswith('image/'):
            return {
                "status": "error",
                "message": "Invalid file type. Please upload an image."
            }
        
        id_path = str(TEMP_DIR / "temp_id.jpg")
        
        with open(id_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"✅ ID image saved to {id_path}")
        
        try:
            DeepFace.extract_faces(id_path, enforce_detection=True)
            print("✅ Face detected in ID image")
            
            return {
                "status": "success",
                "message": "ID uploaded successfully. Face detected!"
            }
        except Exception as face_error:
            if os.path.exists(id_path):
                os.remove(id_path)
            id_path = None
            
            print(f"❌ No face detected in ID: {str(face_error)}")
            return {
                "status": "error",
                "message": "No face detected in ID photo. Please upload a clear photo with your face."
            }
            
    except Exception as e:
        print(f"❌ Error uploading ID: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to upload ID: {str(e)}"
        }


@router.post("/detect-blink")
async def detect_blink(file: UploadFile = File(...), session_id: str = "default"):
    """
    Detect single blink for liveness verification.
    """
    live_path = str(TEMP_DIR / f"temp_blink_{session_id}.jpg")
    
    try:
        with open(live_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        session = get_or_create_session(session_id)
        session["frame_count"] += 1
        
        print(f"\n{'='*60}")
        print(f"📸 BLINK CHECK - Frame #{session['frame_count']} - Session: {session_id}")
        
        face_detected, eyes_open, left_ear, right_ear, num_eyes = detect_blink_opencv(live_path)
        
        current_time = time.time()
        current_state = "open" if eyes_open else "closed"
        previous_state = session["previous_blink_state"]
        
        blink_completed = False
        
        if face_detected and not session["blink_detected"]:
            print(f"📊 Blink State: {previous_state} → {current_state}")
            
            # BLINK DETECTION: closed → open transition
            if previous_state == "closed" and current_state == "open":
                time_closed = current_time - session["last_closed_time"]
                time_since_last = current_time - session["last_blink_time"]
                
                print(f"⏱️ Eyes closed duration: {time_closed:.2f}s")
                
                if time_closed >= 0.05 and time_since_last >= 0.2:
                  
                    try:
                        result = DeepFace.verify(
                            img1_path=id_path,
                            img2_path=live_path,
                            model_name="ArcFace",  
                            detector_backend="mtcnn",
                            enforce_detection=True)
                        
                        
                        verified = result["verified"]
                        
                        distance = result["distance"]
                        threshold = result["threshold"]
                        
                        if verified:
                            session["blink_detected"] = True
                            print(f"✅ ✅ ✅ ID VERIFIED during blink! Person matches ID photo! ✅ ✅ ✅")
                        
                    except Exception as e:
                        print(f"  ❌ Error updating session: {str(e)}")
                    
                      
                        
                    session["last_blink_time"] = current_time
                    blink_completed = True
                    print(f"✅ ✅ ✅ BLINK DETECTED! ✅ ✅ ✅")
                else:
                    print(f"⚠️ Blink rejected: too short or too soon")
            
            if current_state == "closed" and previous_state != "closed":
                session["last_closed_time"] = current_time
                print(f"🔴 EYES CLOSED at frame #{session['frame_count']}")
            
            session["previous_blink_state"] = current_state
        
        print(f"📈 Blink Status: {'✅ Complete' if session['blink_detected'] else '⏳ Waiting'}")
        print(f"{'='*60}\n")
        
        if os.path.exists(live_path):
            os.remove(live_path)
        
        avg_ear = (left_ear + right_ear) / 2.0 if left_ear > 0 or right_ear > 0 else 0.0
        
        return {
            "face_detected": bool(face_detected),
            "eyes_open": bool(eyes_open),
            "blink_detected": bool(session["blink_detected"]),
            "blink_completed": blink_completed,
            "session_id": session_id,
            "current_state": current_state,
            "num_eyes_detected": num_eyes,
            "message": "Blink detected!" if session["blink_detected"] else "Waiting for blink..."
        }
        
    except Exception as e:
        print(f"❌ Error in blink detection: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if os.path.exists(live_path):
            os.remove(live_path)
        
        return {
            "face_detected": False,
            "eyes_open": True,
            "blink_detected": False,
            "error": str(e)
        }


@router.post("/detect-head-turn")
async def detect_head_turn(file: UploadFile = File(...), session_id: str = "default", direction: str = "left"):
    """
    Detect head turn (left or right profile) for liveness verification.
    Direction should be 'left' or 'right'.
    IMPORTANT: User must actually turn their head - frontal face will be rejected.
    After successful turn detection, verifies against ID photo.
    """
    live_path = str(TEMP_DIR / f"temp_pose_{session_id}_{direction}.jpg")
    
    try:
        with open(live_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        session = get_or_create_session(session_id)
        session["frame_count"] += 1
        
        print(f"\n{'='*60}")
        print(f"📸 HEAD TURN CHECK ({direction.upper()}) - Frame #{session['frame_count']} - Session: {session_id}")
        
        face_detected, is_profile, face_area, eye_count, is_frontal = detect_head_pose(live_path)
        
        current_time = time.time()
        pose_completed = False
        id_verified = False
        id_distance = None
        id_threshold = None
        rejection_reason = None
        
        if face_detected:
            state_key = f"previous_{direction}_state"
            detected_key = f"{direction}_pose_detected"
            time_key = f"last_{direction}_time"
            frontal_reject_key = f"{direction}_frontal_rejected_count"
            
            current_state = "profile" if is_profile else "frontal"
            previous_state = session.get(state_key, "frontal")
            
            print(f"📊 {direction.capitalize()} Pose State: {previous_state} → {current_state}")
            print(f"📐 Face Area: {face_area}, Eyes: {eye_count}, Is Frontal: {is_frontal}")
            
            # CRITICAL: Reject if user is facing front (both eyes visible)
            if is_frontal and not session[detected_key]:
                session[frontal_reject_key] = session.get(frontal_reject_key, 0) + 1
                rejection_reason = f"Please turn your head to the {direction}, not facing front"
                print(f"❌ REJECTED: User is facing front (both eyes visible)")
                print(f"⚠️ Frontal rejections for {direction}: {session[frontal_reject_key]}")
            
            # HEAD TURN DETECTION: profile detected and not yet completed
            elif is_profile and not session[detected_key]:
                time_since_last = current_time - session[time_key]
                
                # Verify it's actually a profile (1 or fewer eyes visible)
                if eye_count <= 1 and time_since_last >= 0.3:
                    print(f"✅ ✅ ✅ {direction.upper()} TURN DETECTED! ✅ ✅ ✅")
                    print(f"🔍 Now verifying against ID photo...")
                    
                    # Verify against ID photo
                    is_match, distance, threshold, error_msg = verify_against_id(live_path)
                    
                    if is_match:
                        session[detected_key] = True
                        session[time_key] = current_time
                        pose_completed = True
                        id_verified = True
                        id_distance = distance
                        id_threshold = threshold
                        print(f"✅ ✅ ✅ ID VERIFIED! Person matches ID photo! ✅ ✅ ✅")
                    else:
                        rejection_reason = "Person does not match ID photo"
                        print(f"❌ ID VERIFICATION FAILED: {error_msg if error_msg else 'No match'}")
                        if distance and threshold:
                            id_distance = distance
                            id_threshold = threshold
                else:
                    rejection_reason = f"Turn not confirmed: eyes={eye_count}, time={time_since_last:.2f}s"
                    print(f"⚠️ {direction.capitalize()} turn not confirmed: eyes={eye_count}, time={time_since_last:.2f}s")
            
            session[state_key] = current_state
        else:
            rejection_reason = "No face detected"
            print(f"⚠️ No face detected")
        
        detected_key = f"{direction}_pose_detected"
        print(f"📈 {direction.capitalize()} Turn Status: {'✅ Complete' if session[detected_key] else '⏳ Waiting'}")
        print(f"{'='*60}\n")
        
        if os.path.exists(live_path):
            os.remove(live_path)
        
        # Prepare response message
        if session[detected_key]:
            message = f"{direction.capitalize()} turn verified and ID confirmed!"
        elif rejection_reason:
            message = rejection_reason
        else:
            message = f"Turn your head to the {direction}..."
        
        return {
            "face_detected": bool(face_detected),
            "is_profile": bool(is_profile),
            "is_frontal": bool(is_frontal),
            "pose_detected": bool(session[detected_key]),
            "pose_completed": pose_completed,
            "id_verified": id_verified,
            "id_distance": float(id_distance) if id_distance is not None else None,
            "id_threshold": float(id_threshold) if id_threshold is not None else None,
            "direction": direction,
            "face_area": face_area,
            "eye_count": eye_count,
            "session_id": session_id,
            "rejection_reason": rejection_reason,
            "message": message
        }
        
    except Exception as e:
        print(f"❌ Error in head turn detection: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if os.path.exists(live_path):
            os.remove(live_path)
        
        return {
            "face_detected": False,
            "is_profile": False,
            "is_frontal": False,
            "pose_detected": False,
            "id_verified": False,
            "error": str(e)
        }

@router.post("/compare")
async def compare(file: UploadFile = File(...)):
    global id_path
    
    if id_path is None or not os.path.exists(id_path):
        return {
            "match": False,
            "message": "ID not uploaded. Please upload ID first.",
            "no_id": True
        }
    
    live_path = str(TEMP_DIR / "temp_live.jpg")
    
    try:
        with open(live_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        result = DeepFace.verify(
            img1_path=id_path,
            img2_path=live_path,
            model_name="ArcFace",
            enforce_detection=True,
            detector_backend="mtcnn"
        )
        
        verified = result["verified"]
        distance = result["distance"]
        threshold = result["threshold"]
        
        print(f"{'✅ MATCH' if verified else '❌ NO MATCH'} - Distance: {distance:.4f}, Threshold: {threshold:.4f}")
        
        if os.path.exists(live_path):
            os.remove(live_path)
        
        return {
            "match": bool(verified),
            "distance": float(distance),
            "threshold": float(threshold),
            "model": result["model"],
            "detector": result["detector_backend"],
            "message": "Face verified!" if verified else "Face does not match"
        }
        
    except ValueError as ve:
        error_msg = str(ve).lower()
        
        if os.path.exists(live_path):
            os.remove(live_path)
        
        if "face could not be detected" in error_msg or "no face" in error_msg:
            print("⚠️ No face detected in live frame")
            return {
                "match": False,
                "message": "No face detected. Please look at the camera.",
                "no_face_detected": True
            }
        else:
            print(f"⚠️ Verification error: {str(ve)}")
            return {
                "match": False,
                "message": str(ve),
                "error": True
            }
            
    except Exception as e:
        print(f"❌ Unexpected error during comparison: {str(e)}")
        
        if os.path.exists(live_path):
            os.remove(live_path)
        
        return {
            "match": False,
            "message": f"Verification error: {str(e)}",
            "error": True
        }


@router.post("/reset-liveness-session")
async def reset_liveness_session(request: LivenessResetRequest):
    """Reset a specific liveness verification session"""
    try:
        if request.session_id in liveness_sessions:
            del liveness_sessions[request.session_id]
            print(f"✅ Liveness session {request.session_id} cleared")
        
        return {
            "status": "success",
            "message": "Liveness session reset successfully"
        }
    except Exception as e:
        print(f"❌ Error resetting liveness session: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


@router.post("/reset")
async def reset_id():
    global id_path
    
    try:
        if id_path and os.path.exists(id_path):
            os.remove(id_path)
            print("✅ ID image cleared")
        
        id_path = None
        liveness_sessions.clear()
        
        return {
            "status": "success",
            "message": "ID and all sessions cleared successfully"
        }
    except Exception as e:
        print(f"❌ Error clearing ID: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "id_uploaded": id_path is not None and os.path.exists(id_path) if id_path else False,
        "active_sessions": len(liveness_sessions)
    }


@router.get("/session-status/{session_id}")
async def get_session_status(session_id: str):
    """Get the status of a liveness verification session"""
    if session_id in liveness_sessions:
        session = liveness_sessions[session_id]
        return {
            "exists": True,
            "blink_detected": session["blink_detected"],
            "left_pose_detected": session["left_pose_detected"],
            "right_pose_detected": session["right_pose_detected"],
            "liveness_complete": session["blink_detected"] and session["left_pose_detected"] and session["right_pose_detected"],
            "created_at": session["created_at"],
            "frame_count": session["frame_count"]
        }
    return {
        "exists": False,
        "message": "Session not found"
    }