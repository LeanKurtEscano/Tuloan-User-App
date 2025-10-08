from ast import Dict
import time
import os
from deepface import DeepFace

# blink_sessions: Dict[str, dict] = {}

class FaceRecognitionService:
    def __init__(self):
        pass

    def verify_against_id(live_image_path: str, id_path: str) -> tuple:
        """
        Verify if the person in the live image matches the ID.
        Returns: (is_match, distance, threshold, error_message)
        """
     

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

            print(
                f"  🔍 ID Verification: {'✅ MATCH' if verified else '❌ NO MATCH'} "
                f"- Distance: {distance:.4f}, Threshold: {threshold:.4f}"
            )

            return verified, distance, threshold, None

        except Exception as e:
            error_msg = str(e).lower()
            print(f"  ❌ ID Verification Error: {str(e)}")
            return False, None, None, str(e)

