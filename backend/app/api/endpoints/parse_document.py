from fastapi import FastAPI, UploadFile, File, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import os
from pathlib import Path

router = APIRouter()
id_path = None

TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)


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
            "match": verified,
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


@router.post("/reset")
async def reset_id():
    global id_path
    
    try:
        if id_path and os.path.exists(id_path):
            os.remove(id_path)
            print("✅ ID image cleared")
        
        id_path = None
        
        return {
            "status": "success",
            "message": "ID cleared successfully"
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
        "id_uploaded": id_path is not None and os.path.exists(id_path) if id_path else False
    }