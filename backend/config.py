"""
Configuration file for tunable parameters
"""

# ==============================================================================
# Alignment Settings
# ==============================================================================

# Similarity Thresholds
WORD_SIMILARITY_THRESHOLD = 0.45  # Minimum threshold to consider a word correct (0.7-0.85 recommended)
SEGMENT_SCORE_THRESHOLD = 0.5     # Minimum threshold to accept a candidate segment (0.4-0.6 recommended)

# Scoring Weights
LEVENSHTEIN_WEIGHT = 0.7  # Weight of Levenshtein distance in segment calculation
LENGTH_PENALTY_WEIGHT = 0.3  # Weight of length difference in segment calculation

# Alignment Costs
DELETE_COST = 0.8   # Cost of extra spoken word
INSERT_COST = 0.8   # Cost of missing Quranic word

# Tracking Window
TRACKING_WINDOW_SIZE = 40      # Number of words in search window (tracking mode)
BACKWARD_MARGIN = 15           # Number of words to go back from current position

# Segment Generation Limits
MIN_SEGMENT_WORDS = 5   # Minimum number of words in segment
MAX_SEGMENT_WORDS = 25  # Maximum number of words in segment
SEGMENT_STRIDE = 3      # Overlap between consecutive segments

# ==============================================================================
# Session Settings
# ==============================================================================

# Confidence & Tracking
CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence before switching to search mode
MAX_LOW_CONFIDENCE_CHUNKS = 3  # Number of low confidence chunks before switching to search

# ==============================================================================
# Audio Settings
# ==============================================================================

# Silence Detection
MIN_AUDIO_ENERGY = 0.01  # Minimum audio energy (lower = silence)
                         # Recommended values:
                         # 0.005 = very sensitive
                         # 0.01  = medium (default)
                         # 0.02  = strict

# ==============================================================================
# Logging Settings
# ==============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_AUDIO_ENERGY = True  # Log audio energy for each chunk
LOG_ALIGNMENT_DETAILS = True  # Log alignment details

# ==============================================================================
# Tuning Notes
# ==============================================================================

"""
If results show:

1. Too many false positives (incorrect words considered correct):
   - Increase WORD_SIMILARITY_THRESHOLD to 0.8
   - Increase SEGMENT_SCORE_THRESHOLD to 0.55

2. Too many false negatives (correct words considered incorrect):
   - Decrease WORD_SIMILARITY_THRESHOLD to 0.7
   - Decrease SEGMENT_SCORE_THRESHOLD to 0.45
   
3. Tracking loses position frequently:
   - Increase TRACKING_WINDOW_SIZE to 80
   - Increase BACKWARD_MARGIN to 20
   - Decrease CONFIDENCE_THRESHOLD to 0.3

4. Silence is processed as audio:
   - Increase MIN_AUDIO_ENERGY to 0.02 or 0.03

5. Real audio is rejected:
   - Decrease MIN_AUDIO_ENERGY to 0.005

6. Processing is slow:
   - Reduce TRACKING_WINDOW_SIZE to 40
   - Reduce MAX_SEGMENT_WORDS to 20
"""

