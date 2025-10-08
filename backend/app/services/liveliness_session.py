import time
from typing import Dict


class LivenessSessionManager:
    def __init__(self):
        # Stores all active liveness sessions
        self.liveness_sessions: Dict[str, dict] = {}

    def get_or_create_session(self, session_id: str) -> dict:
        """Get or create a liveness verification session."""
        if session_id not in self.liveness_sessions:
            print(f"🆕 Creating new session: {session_id}")
            self.liveness_sessions[session_id] = {
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
        return self.liveness_sessions[session_id]