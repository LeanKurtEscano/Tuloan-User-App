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

# Blink detection state storage (session-based)
blink_sessions: Dict[str, dict] = {}

class BlinkResetRequest(BaseModel):
    session_id: str


def detect_blink_opencv(image_path: str):
    """
    Detect blinks using OpenCV's Haar Cascade for eyes.
    Returns: (face_detected, eyes_open, left_ear, right_ear, num_eyes_detected)
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print("  ❌ Could not read image")
            return False, True, 0.0, 0.0, 0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load Haar Cascade classifiers
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Convert to list and check length explicitly
        faces_list = list(faces) if len(faces) > 0 else []
        
        if len(faces_list) == 0:
            print("  ❌ No face detected")
            return False, True, 0.0, 0.0, 0
        
        # Get the largest face
        (x, y, w, h) = max(faces_list, key=lambda face: face[2] * face[3])
        
        # Ensure coordinates are integers
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        print(f"  ✅ Face detected at ({x},{y}) size {w}x{h}")
        
        # Region of interest for eyes (upper half of face)
        roi_y_end = int(y + h * 0.6)
        roi_gray = gray[y:roi_y_end, x:x+w]
        
        # Try MULTIPLE eye detection strategies
        eyes_list = []
        
        # Strategy 1: Very lenient detection
        eyes1 = eye_cascade.detectMultiScale(
            roi_gray, 
            scaleFactor=1.05,
            minNeighbors=2,      # Even more lenient
            minSize=(10, 10)     # Even smaller
        )
        if len(eyes1) > 0:
            eyes_list = list(eyes1)
            print(f"  👁️ Strategy 1: Found {len(eyes_list)} eyes")
        
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
                print(f"  👁️ Strategy 2: Found {len(eyes_list)} eyes")
        
        # Strategy 3: If still no eyes, apply histogram equalization
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
                print(f"  👁️ Strategy 3 (enhanced): Found {len(eyes_list)} eyes")
        
        num_eyes = len(eyes_list)
        print(f"  👁️ Total eyes detected: {num_eyes}")
        
        # CRITICAL LOGIC CHANGE: Use NUMBER of eyes detected as primary indicator
        # When you blink, eyes disappear from detection!
        
        if num_eyes >= 2:
            # Both eyes visible = OPEN
            eyes_sorted = sorted(eyes_list, key=lambda e: e[0])
            left_eye = eyes_sorted[0]
            right_eye = eyes_sorted[1]
            
            left_ratio = float(left_eye[3]) / float(left_eye[2]) if left_eye[2] > 0 else 0.0
            right_ratio = float(right_eye[3]) / float(right_eye[2]) if right_eye[2] > 0 else 0.0
            
            print(f"  👁️ Both eyes visible - Ratios: L={left_ratio:.3f}, R={right_ratio:.3f}")
            return True, True, left_ratio, right_ratio, num_eyes  # Eyes OPEN
        
        elif num_eyes == 1:
            # Only 1 eye visible = might be blinking or side view
            eye = eyes_list[0]
            ratio = float(eye[3]) / float(eye[2]) if eye[2] > 0 else 0.0
            print(f"  ⚠️ Only 1 eye detected - Ratio: {ratio:.3f} - marking as CLOSED")
            return True, False, ratio, ratio, num_eyes  # Eyes CLOSED
        
        else:
            # No eyes detected but face is there = DEFINITELY CLOSED/BLINKING
            print(f"  🔴 NO EYES DETECTED - Eyes are CLOSED!")
            return True, False, 0.0, 0.0, num_eyes  # Eyes CLOSED
            
    except Exception as e:
        print(f"  ❌ Error in blink detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, True, 0.0, 0.0, 0


def get_or_create_session(session_id: str) -> dict:
    """Get or create a blink detection session"""
    if session_id not in blink_sessions:
        print(f"🆕 Creating new session: {session_id}")
        blink_sessions[session_id] = {
            "blink_count": 0,
            "previous_state": "unknown",
            "last_blink_time": 0,
            "last_closed_time": 0,
            "last_eyes_count": 0,
            "created_at": time.time(),
            "frame_count": 0
        }
    return blink_sessions[session_id]


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
    ULTRA-SIMPLIFIED BLINK DETECTION based on eye count.
    Logic: 2 eyes → 0-1 eyes → 2 eyes = BLINK!
    """
    live_path = str(TEMP_DIR / f"temp_blink_{session_id}.jpg")
    
    try:
        # Save the frame
        with open(live_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Get session data
        session = get_or_create_session(session_id)
        session["frame_count"] += 1
        
        print(f"\n{'='*60}")
        print(f"📸 Frame #{session['frame_count']} - Session: {session_id}")
        
        # Detect eye state
        face_detected, eyes_open, left_ear, right_ear, num_eyes = detect_blink_opencv(live_path)
        
        current_time = time.time()
        current_state = "open" if eyes_open else "closed"
        previous_state = session["previous_state"]
        previous_eyes_count = session["last_eyes_count"]
        
        blink_detected = False
        
        if face_detected:
            print(f"📊 State: {previous_state} → {current_state}")
            print(f"👁️ Eyes count: {previous_eyes_count} → {num_eyes}")
            
            # BLINK DETECTION: closed → open transition
            if previous_state == "closed" and current_state == "open":
                time_closed = current_time - session["last_closed_time"]
                time_since_last_blink = current_time - session["last_blink_time"]
                
                print(f"⏱️ Eyes closed duration: {time_closed:.2f}s")
                print(f"⏱️ Time since last blink: {time_since_last_blink:.2f}s")
                
                # Very lenient timing: just need some closed time
                if time_closed >= 0.05 and time_since_last_blink >= 0.2:
                    session["blink_count"] += 1
                    session["last_blink_time"] = current_time
                    blink_detected = True
                    print(f"✅ ✅ ✅ BLINK #{session['blink_count']} DETECTED! ✅ ✅ ✅")
                else:
                    print(f"⚠️ Blink rejected: too short or too soon")
            
            # Track when eyes close
            if current_state == "closed" and previous_state != "closed":
                session["last_closed_time"] = current_time
                print(f"🔴 EYES CLOSED at frame #{session['frame_count']}")
            
            # Track when eyes open
            if current_state == "open" and previous_state == "closed":
                print(f"🟢 EYES OPENED at frame #{session['frame_count']}")
            
            # Update state
            session["previous_state"] = current_state
            session["last_eyes_count"] = num_eyes
        else:
            print(f"⚠️ No face detected in frame")
        
        print(f"📈 Session Stats: {session['blink_count']} blinks total")
        print(f"{'='*60}\n")
        
        # Cleanup
        if os.path.exists(live_path):
            os.remove(live_path)
        
        # Calculate average EAR for debugging
        avg_ear = (left_ear + right_ear) / 2.0 if left_ear > 0 or right_ear > 0 else 0.0
        
        return {
            "face_detected": bool(face_detected),
            "eyes_open": bool(eyes_open),
            "blink_count": int(session["blink_count"]),
            "blink_detected": blink_detected,
            "left_ear": float(left_ear),
            "right_ear": float(right_ear),
            "avg_ear": float(avg_ear),
            "num_eyes_detected": num_eyes,
            "session_id": session_id,
            "current_state": current_state,
            "previous_state": previous_state,
            "frame_count": session["frame_count"],
            "message": f"Blinks detected: {session['blink_count']}"
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
            "blink_count": 0,
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
        # Save live image
        with open(live_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Perform face verification
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


@router.post("/reset-blink-session")
async def reset_blink_session(request: BlinkResetRequest):
    """Reset a specific blink detection session"""
    try:
        if request.session_id in blink_sessions:
            del blink_sessions[request.session_id]
            print(f"✅ Blink session {request.session_id} cleared")
        
        return {
            "status": "success",
            "message": "Blink session reset successfully"
        }
    except Exception as e:
        print(f"❌ Error resetting blink session: {str(e)}")
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
        
        # Clear all blink sessions
        blink_sessions.clear()
        
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
        "active_sessions": len(blink_sessions)
    }


@router.get("/session-status/{session_id}")
async def get_session_status(session_id: str):
    """Get the status of a blink detection session"""
    if session_id in blink_sessions:
        session = blink_sessions[session_id]
        return {
            "exists": True,
            "blink_count": session["blink_count"],
            "created_at": session["created_at"],
            "last_blink": session["last_blink_time"],
            "frame_count": session["frame_count"]
        }
    return {
        "exists": False,
        "message": "Session not found"
    }