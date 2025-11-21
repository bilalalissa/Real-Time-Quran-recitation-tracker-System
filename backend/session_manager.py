"""
Session State Manager for Quran Recitation Tracking
Maintains per-user session state for alignment continuity
"""

from typing import Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class SessionState:
    """State for a single recitation session"""
    global_word_pos: int = 0
    last_confidence: float = 0.0
    mode: str = "tracking"  # "tracking" or "search"
    consecutive_low_confidence: int = 0
    webm_header: bytes = None  # pyright: ignore[reportAssignmentType] # WebM container header for chunks


class SessionManager:
    """Manages session states for all connected users"""
    
    def __init__(self, confidence_threshold: float = 0.4, max_low_confidence: int = 3):
        self.sessions: Dict[str, SessionState] = {}
        self.confidence_threshold = confidence_threshold
        self.max_low_confidence = max_low_confidence
    
    def create_session(self, sid: str) -> SessionState:
        """Create a new session"""
        state = SessionState()
        self.sessions[sid] = state
        return state
    
    def get_session(self, sid: str) -> SessionState:
        """Get session state, creating if not exists"""
        if sid not in self.sessions:
            return self.create_session(sid)
        return self.sessions[sid]
    
    def delete_session(self, sid: str):
        """Delete a session"""
        if sid in self.sessions:
            del self.sessions[sid]
    
    def update_from_alignment(self, sid: str, confidence: float, furthest_global_index: int):
        """Update session state based on alignment result"""
        state = self.get_session(sid)
        
        # Update position (only move forward)
        state.global_word_pos = max(state.global_word_pos, furthest_global_index)
        
        # Update confidence
        state.last_confidence = confidence
        
        # Track consecutive low confidence
        if confidence < self.confidence_threshold:
            state.consecutive_low_confidence += 1
        else:
            state.consecutive_low_confidence = 0
        
        # Switch to search mode if confidence is consistently low
        if state.consecutive_low_confidence >= self.max_low_confidence:
            state.mode = "search"
        else:
            state.mode = "tracking"
    
    def reset_session_progress(self, sid: str):
        """Reset progress (e.g., when user changes page)"""
        state = self.get_session(sid)
        state.global_word_pos = 0
        state.last_confidence = 0.0
        state.mode = "tracking"
        state.consecutive_low_confidence = 0
        state.webm_header = None # type: ignore
    
    def get_session_info(self, sid: str) -> Dict[str, Any]:
        """Get session info as dict for debugging/monitoring"""
        state = self.get_session(sid)
        return asdict(state)
    
    def has_session(self, sid: str) -> bool:
        """Check if session exists"""
        return sid in self.sessions

