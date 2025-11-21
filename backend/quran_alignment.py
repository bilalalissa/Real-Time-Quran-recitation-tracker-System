"""
Quran recitation tracker Engine
fuzzy search + Levenshtein-based word alignment
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from Levenshtein import distance as levenshtein_distance
import re
import logging


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class WordEntry:
    """Represents a single word in the Quran"""
    global_index: int
    sura: int
    aya: int
    aya_id: int
    word_index: int  # position within the verse
    text: str  # normalized text


@dataclass
class VerseEntry:
    """Represents a complete verse"""
    aya_id: int
    sura: int
    aya: int
    normalized_text: str
    words: List[WordEntry]


@dataclass
class SegmentCandidate:
    """A candidate segment for alignment"""
    words: List[WordEntry]
    text: str
    start_global_index: int
    end_global_index: int
    score: float = 0.0


@dataclass
class AlignmentMatch:
    """Result of aligning a spoken word to a Quran word"""
    spoken_word: Optional[str]
    quran_word: Optional[WordEntry]
    similarity: float
    alignment_type: str  # "match", "insert", "delete"
    is_correct: bool


@dataclass
class AlignmentResult:
    """Complete alignment result for a chunk"""
    matches: List[AlignmentMatch]
    confidence: float
    furthest_global_index: int
    segment_score: float


# ============================================================================
# Configuration
# ============================================================================

class AlignmentConfig:
    """Configuration parameters for alignment"""
    
    # Segment scoring
    ALPHA = 0.7  # weight for normalized Levenshtein distance
    BETA = 0.3   # weight for length penalty
    SEGMENT_THRESHOLD = 0.45  # minimum score for segment candidates
    
    # Word-level alignment
    WORD_THRESHOLD = 0.7  # minimum similarity for correct word
    DELETE_COST = 0.8     # cost for extra spoken word
    INSERT_COST = 0.8     # cost for missing Quran word
    
    # Tracking mode
    WINDOW_SIZE = 50      # words to search in tracking mode
    BACKWARD_MARGIN = 15  # words to look back from anchor
    
    # Search mode
    CONFIDENCE_THRESHOLD = 0.4  # switch to search mode if below this
    
    # Segment generation
    MIN_SEGMENT_WORDS = 5
    MAX_SEGMENT_WORDS = 25
    SEGMENT_STRIDE = 3  # overlap between consecutive segments


# ============================================================================
# Text Normalization
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize Arabic text (remove diacritics, normalize letters)"""
    # Remove diacritics
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # Remove non-Arabic characters except spaces
    text = re.sub(r'[^\u0621-\u063A\u0641-\u064A\s]', '', text)
    # Normalize letters
    text = text.replace('ٱ', 'ا').replace('ى', 'ي').replace('ة', 'ه')
    return text.strip()


def calculate_similarity(word1: str, word2: str) -> float:
    """Calculate similarity between two words (0 to 1)"""
    if not word1 or not word2:
        return 0.0
    max_len = max(len(word1), len(word2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(word1, word2)
    return 1.0 - (dist / max_len)


# ============================================================================
# Quran Data Builder
# ============================================================================

class QuranDataBuilder:
    """Builds indexed data structures from raw Quran JSON"""
    
    @staticmethod
    def build_indices(quran_data: List[Dict[str, Any]]) -> Tuple[List[WordEntry], List[VerseEntry], Dict[int, VerseEntry]]:
        """
        Build ALL_WORDS, VERSES, and aya_id lookup map
        
        Returns:
            (all_words, verses, aya_id_map)
        """
        all_words: List[WordEntry] = []
        verses: List[VerseEntry] = []
        aya_id_map: Dict[int, VerseEntry] = {}
        
        global_index = 0
        
        for aya_data in quran_data:
            aya_id = aya_data['id']
            sura = aya_data.get('sura_no', aya_data.get('sura', 0))  # Handle both key names
            aya = aya_data.get('aya_no', aya_data.get('aya', 0))
            
            # Normalize full verse text
            aya_text_emlaey = aya_data.get('aya_text_emlaey', '')
            normalized_verse = normalize_text(aya_text_emlaey)
            words_in_verse = normalized_verse.split()
            
            # Create WordEntry for each word
            verse_words: List[WordEntry] = []
            for word_index, word_text in enumerate(words_in_verse):
                word_entry = WordEntry(
                    global_index=global_index,
                    sura=sura,
                    aya=aya,
                    aya_id=aya_id,
                    word_index=word_index,
                    text=word_text
                )
                all_words.append(word_entry)
                verse_words.append(word_entry)
                global_index += 1
            
            # Create VerseEntry
            verse_entry = VerseEntry(
                aya_id=aya_id,
                sura=sura,
                aya=aya,
                normalized_text=normalized_verse,
                words=verse_words
            )
            verses.append(verse_entry)
            aya_id_map[aya_id] = verse_entry
        
        return all_words, verses, aya_id_map


# ============================================================================
# Segment Generation
# ============================================================================

class SegmentGenerator:
    """Generates candidate segments for fuzzy matching"""
    
    def __init__(self, all_words: List[WordEntry], config: AlignmentConfig):
        self.all_words = all_words
        self.config = config
    
    def generate_tracking_candidates(
        self, 
        anchor_pos: int, 
        page_verse_ids: Optional[List[int]] = None,
        aya_id_map: Optional[Dict[int, VerseEntry]] = None
    ) -> List[SegmentCandidate]:
        """Generate segments in tracking mode (local window)
        
        Args:
            anchor_pos: Current global word position
            page_verse_ids: Optional list of verse IDs to constrain search to current page
            aya_id_map: Map of verse IDs to VerseEntry (required if page_verse_ids is provided)
        
        Returns:
            List of segment candidates within the tracking window
        """
        # Calculate page boundaries if page_verse_ids is provided
        if page_verse_ids and aya_id_map:
            page_min_index, page_max_index = self._get_page_boundaries(page_verse_ids, aya_id_map)
            
            # Constrain anchor_pos to page boundaries
            anchor_pos = max(page_min_index, min(anchor_pos, page_max_index))
            
            # Calculate window within page boundaries
            start = max(anchor_pos - self.config.BACKWARD_MARGIN, page_min_index)
            end = min(start + self.config.WINDOW_SIZE, page_max_index + 1)
            
            # Adjust start if window is too small
            if end - start < self.config.MIN_SEGMENT_WORDS:
                start = max(page_min_index, end - self.config.WINDOW_SIZE)
        else:
            # Original behavior: no page constraints
            start = max(anchor_pos - self.config.BACKWARD_MARGIN, 0)
            end = min(start + self.config.WINDOW_SIZE, len(self.all_words))
        
        window_words = self.all_words[start:end]
        
        return self._generate_segments_from_words(window_words)
    
    def _get_page_boundaries(
        self, 
        page_verse_ids: List[int], 
        aya_id_map: Dict[int, VerseEntry]
    ) -> Tuple[int, int]:
        """Calculate min and max global_index for verses in the page
        
        Args:
            page_verse_ids: List of verse IDs in the current page
            aya_id_map: Map of verse IDs to VerseEntry
        
        Returns:
            Tuple of (min_global_index, max_global_index)
        """
        min_index = float('inf')
        max_index = -1
        
        for aya_id in page_verse_ids:
            if aya_id not in aya_id_map:
                continue
            verse = aya_id_map[aya_id]
            if len(verse.words) > 0:
                min_index = min(min_index, verse.words[0].global_index)
                max_index = max(max_index, verse.words[-1].global_index)
        
        # Fallback to entire Quran if no valid verses found
        if min_index == float('inf') or max_index == -1:
            min_index = 0
            max_index = len(self.all_words) - 1
        
        return (int(min_index), int(max_index))
    
    def generate_search_candidates(self, verse_ids: List[int], aya_id_map: Dict[int, VerseEntry]) -> List[SegmentCandidate]:
        """Generate segments in search mode (full verses)"""
        segments: List[SegmentCandidate] = []
        
        for aya_id in verse_ids:
            if aya_id not in aya_id_map:
                continue
            verse = aya_id_map[aya_id]
            
            # Full verse as one segment
            if len(verse.words) > 0:
                segment = SegmentCandidate(
                    words=verse.words,
                    text=" ".join(w.text for w in verse.words),
                    start_global_index=verse.words[0].global_index,
                    end_global_index=verse.words[-1].global_index
                )
                segments.append(segment)
        
        return segments
    
    def _generate_segments_from_words(self, words: List[WordEntry]) -> List[SegmentCandidate]:
        """Generate sliding window segments from a list of words"""
        segments: List[SegmentCandidate] = []
        
        for start_idx in range(0, len(words), self.config.SEGMENT_STRIDE):
            for length in range(self.config.MIN_SEGMENT_WORDS, self.config.MAX_SEGMENT_WORDS + 1):
                end_idx = start_idx + length
                if end_idx > len(words):
                    break
                
                segment_words = words[start_idx:end_idx]
                segment = SegmentCandidate(
                    words=segment_words,
                    text=" ".join(w.text for w in segment_words),
                    start_global_index=segment_words[0].global_index,
                    end_global_index=segment_words[-1].global_index
                )
                segments.append(segment)
        
        return segments


# ============================================================================
# Segment Scoring
# ============================================================================

class SegmentScorer:
    """Scores segment candidates using Levenshtein distance"""
    
    def __init__(self, config: AlignmentConfig):
        self.config = config
    
    def score_segment(self, spoken_text: str, segment: SegmentCandidate) -> float:
        """
        Score a segment candidate against spoken text
        
        Returns score between 0 and 1 (higher is better)
        """
        Q = spoken_text
        V = segment.text
        
        dist = levenshtein_distance(Q, V)
        len_q = len(Q)
        len_v = len(V)
        
        if max(len_q, len_v) == 0:
            return 0.0
        
        norm_dist = dist / max(len_q, len_v)
        length_penalty = abs(len_q - len_v) / max(len_q, len_v)
        
        score = 1.0 - (self.config.ALPHA * norm_dist + self.config.BETA * length_penalty)
        
        return max(0.0, score)
    
    def find_best_segments(self, spoken_text: str, candidates: List[SegmentCandidate], top_n: int = 5) -> List[SegmentCandidate]:
        """Find top N best matching segments"""
        for candidate in candidates:
            candidate.score = self.score_segment(spoken_text, candidate)
        
        # Filter by threshold and sort
        valid_candidates = [c for c in candidates if c.score >= self.config.SEGMENT_THRESHOLD]
        valid_candidates.sort(key=lambda x: x.score, reverse=True)
        
        return valid_candidates[:top_n]


# ============================================================================
# Word-Level Alignment (Dynamic Programming)
# ============================================================================

class WordAligner:
    """Performs word-level alignment using Dynamic Programming"""
    
    def __init__(self, config: AlignmentConfig):
        self.config = config
    
    def align(self, spoken_words: List[str], segment_words: List[WordEntry]) -> List[AlignmentMatch]:
        """
        Align spoken words to Quran segment words using DP
        
        Returns list of alignment matches
        """
        m = len(spoken_words)
        n = len(segment_words)
        
        if m == 0 or n == 0:
            return []
        
        # Initialize DP table
        dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0.0
        
        # Fill first row (insertions from Quran)
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j-1] + self.config.INSERT_COST
        
        # Fill first column (deletions from spoken)
        for i in range(1, m + 1):
            dp[i][0] = dp[i-1][0] + self.config.DELETE_COST
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                spoken_word = spoken_words[i-1]
                quran_word = segment_words[j-1].text
                
                similarity = calculate_similarity(spoken_word, quran_word)
                match_cost = 1.0 - similarity
                
                dp[i][j] = min(
                    dp[i-1][j-1] + match_cost,  # match/substitute
                    dp[i-1][j] + self.config.DELETE_COST,  # delete from spoken
                    dp[i][j-1] + self.config.INSERT_COST   # insert from Quran
                )
        
        # Backtrack to get alignment path
        return self._backtrack(dp, spoken_words, segment_words)
    
    def _backtrack(self, dp: List[List[float]], spoken_words: List[str], segment_words: List[WordEntry]) -> List[AlignmentMatch]:
        """Backtrack through DP table to extract alignment"""
        matches: List[AlignmentMatch] = []
        i = len(spoken_words)
        j = len(segment_words)
        
        while i > 0 or j > 0:
            if i == 0:
                # Remaining Quran words (missing)
                j -= 1
                matches.append(AlignmentMatch(
                    spoken_word=None,
                    quran_word=segment_words[j],
                    similarity=0.0,
                    alignment_type="delete",
                    is_correct=False
                ))
            elif j == 0:
                # Remaining spoken words (extra)
                i -= 1
                matches.append(AlignmentMatch(
                    spoken_word=spoken_words[i],
                    quran_word=None,
                    similarity=0.0,
                    alignment_type="insert",
                    is_correct=False
                ))
            else:
                spoken_word = spoken_words[i-1]
                quran_word = segment_words[j-1]
                similarity = calculate_similarity(spoken_word, quran_word.text)
                match_cost = 1.0 - similarity
                
                # Determine which transition was taken
                if dp[i][j] == dp[i-1][j-1] + match_cost:
                    # Match/substitute
                    matches.append(AlignmentMatch(
                        spoken_word=spoken_word,
                        quran_word=quran_word,
                        similarity=similarity,
                        alignment_type="match",
                        is_correct=similarity >= self.config.WORD_THRESHOLD
                    ))
                    i -= 1
                    j -= 1
                elif dp[i][j] == dp[i-1][j] + self.config.DELETE_COST:
                    # Delete from spoken (extra word)
                    matches.append(AlignmentMatch(
                        spoken_word=spoken_word,
                        quran_word=None,
                        similarity=0.0,
                        alignment_type="insert",
                        is_correct=False
                    ))
                    i -= 1
                else:
                    # Insert from Quran (missing word)
                    matches.append(AlignmentMatch(
                        spoken_word=None,
                        quran_word=quran_word,
                        similarity=0.0,
                        alignment_type="delete",
                        is_correct=False
                    ))
                    j -= 1
        
        matches.reverse()
        return matches


# ============================================================================
# Special Phrase Handler (Basmallah, Istiatha)
# ============================================================================

class SpecialPhraseHandler:
    """Handles detection and alignment of common Quranic phrases"""
    
    def __init__(self):
        self.basmallah = normalize_text("بسم الله الرحمن الرحيم")
        self.istiatha = normalize_text("أعوذ بالله من الشيطان الرجيم")
        self.basmallah_words = self.basmallah.split()
        self.istiatha_words = self.istiatha.split()
    
    def detect_and_strip_special_phrases(self, spoken_words: List[str]) -> Tuple[List[str], List[str]]:
        """
        Detect special phrases at start of spoken words
        
        Returns:
            (remaining_words, detected_phrases)
        """
        detected = []
        remaining = spoken_words.copy()
        
        # Check for Istiatha first (usually comes before Basmallah)
        if self._matches_phrase(remaining, self.istiatha_words):
            detected.append("istiatha")
            remaining = remaining[len(self.istiatha_words):]
        
        # Check for Basmallah
        if self._matches_phrase(remaining, self.basmallah_words):
            detected.append("basmallah")
            remaining = remaining[len(self.basmallah_words):]
        
        return remaining, detected
    
    def _matches_phrase(self, spoken_words: List[str], phrase_words: List[str], threshold: float = 0.7) -> bool:
        """Check if spoken words start with a phrase"""
        if len(spoken_words) < len(phrase_words):
            return False
        
        total_similarity = 0.0
        for i, phrase_word in enumerate(phrase_words):
            similarity = calculate_similarity(spoken_words[i], phrase_word)
            total_similarity += similarity
        
        avg_similarity = total_similarity / len(phrase_words)
        return avg_similarity >= threshold


# ============================================================================
# Main Alignment Engine
# ============================================================================

class QuranAlignmentEngine:
    """Main engine coordinating the alignment process"""
    
    def __init__(self, quran_data: List[Dict[str, Any]], config: Optional[AlignmentConfig] = None):
        self.config = config or AlignmentConfig()
        
        # Build indices
        self.all_words, self.verses, self.aya_id_map = QuranDataBuilder.build_indices(quran_data)
        
        # Initialize components
        self.segment_generator = SegmentGenerator(self.all_words, self.config)
        self.segment_scorer = SegmentScorer(self.config)
        self.word_aligner = WordAligner(self.config)
        self.special_phrase_handler = SpecialPhraseHandler()
    
    def align_transcript(
        self,
        spoken_words: List[str],
        anchor_pos: int,
        mode: str = "tracking",
        page_verse_ids: Optional[List[int]] = None
    ) -> AlignmentResult:
        """
        Main alignment function
        
        Args:
            spoken_words: List of normalized spoken words
            anchor_pos: Current global word position
            mode: "tracking" or "search"
            page_verse_ids: Verse IDs for current page (used in search mode)
        
        Returns:
            AlignmentResult with matches and confidence
        """
        if not spoken_words:
            return AlignmentResult(matches=[], confidence=0.0, furthest_global_index=anchor_pos, segment_score=0.0)
        
        # Handle special phrases (Basmallah, Istiatha)
        remaining_words, detected_phrases = self.special_phrase_handler.detect_and_strip_special_phrases(spoken_words)
        
        # If special phrases were detected, log them (they're considered correct by default)
        # In a full implementation, you might want to emit these as separate results
        if detected_phrases:
            logging.info(f"Detected special phrases: {detected_phrases}")
        
        # Use remaining words for alignment
        alignment_words = remaining_words if remaining_words else spoken_words
        spoken_text = " ".join(alignment_words)
        
        # Generate candidate segments
        if mode == "tracking":
            # Pass page_verse_ids to constrain tracking to current page
            candidates = self.segment_generator.generate_tracking_candidates(
                anchor_pos, 
                page_verse_ids=page_verse_ids,
                aya_id_map=self.aya_id_map
            )
        else:
            candidates = self.segment_generator.generate_search_candidates(
                page_verse_ids or [],
                self.aya_id_map
            )
        
        if not candidates:
            return AlignmentResult(matches=[], confidence=0.0, furthest_global_index=anchor_pos, segment_score=0.0)
        
        # Score and select best segments
        best_segments = self.segment_scorer.find_best_segments(spoken_text, candidates, top_n=3)
        
        if not best_segments:
            return AlignmentResult(matches=[], confidence=0.0, furthest_global_index=anchor_pos, segment_score=0.0)
        
        # Take the best segment and perform word-level alignment
        best_segment = best_segments[0]
        matches = self.word_aligner.align(spoken_words, best_segment.words)
        
        # Calculate confidence
        match_similarities = [m.similarity for m in matches if m.alignment_type == "match"]
        confidence = sum(match_similarities) / len(match_similarities) if match_similarities else 0.0
        
        # Find furthest aligned Quran word
        furthest_index = anchor_pos
        for match in matches:
            if match.quran_word and match.is_correct:
                furthest_index = max(furthest_index, match.quran_word.global_index)
        
        return AlignmentResult(
            matches=matches,
            confidence=confidence,
            furthest_global_index=furthest_index,
            segment_score=best_segment.score
        )

