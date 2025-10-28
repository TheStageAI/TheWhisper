from typing import List, Dict, Any
from editdistance import eval as edit_distance


class LocalAgreement:
    """
    A class that implements a local agreement algorithm for streaming ASR.
    
    This class maintains a history of transcriptions and commits words when they
    appear in multiple consecutive transcriptions, ensuring stability in streaming output.
    
    Attributes:
        history (List[List[Dict]]): A list of recent transcription results.
        history_size (int): Maximum number of transcription results to keep in history.
        committed (List[Dict]): List of words that have been committed to the final output.
        last_committed_time (float): Timestamp of the last committed word.
        majority_threshold (int): Minimum number of occurrences needed to commit a word.
        last_commited (List[Dict]): List of words committed in the most recent update.
    """
    
    def __init__(self, history_size: int = 3, majority_threshold: int = 2) -> None:
        """
        Initialize the LocalAgreement object.
        
        Args:
            history_size (int): Maximum number of transcription results to keep in history.
            majority_threshold (int): Minimum number of occurrences needed to commit a word.
        """
        self.history: List[List[Dict[str, Any]]] = []
        self.history_size: int = history_size
        self.committed: List[Dict[str, Any]] = []
        self.last_committed_time: float = 0.0
        self.majority_threshold: int = majority_threshold
        self.last_commited: List[Dict[str, Any]] = []
        self.not_committed_words: List[Dict[str, Any]] = []
    
    def clear(self) -> None:
        """
        Clear the history and committed words.
        """
        self.history = []
        self.committed = []
        self.last_commited = []
        self.last_committed_time = 0.0
        self.not_committed_words = []
    
    def add_transcription(self, words: List[Dict[str, Any]]) -> None:
        """
        Add a new transcription result to the history and update committed words.
        
        Args:
            words (List[Dict]): List of word dictionaries, each containing 'text', 'start', and 'end' keys.
        """
        if len(self.history) >= self.history_size:
            self.history.pop(0)
        self.history.append(words)
        self._update_committed()

    def _update_committed(self) -> None:
        """
        Update the list of committed words based on the current history.
        
        This method identifies words that appear in multiple transcriptions
        and commits them to the final output if they meet the majority threshold.
        """
        self.last_commited = []

        if len(self.history) < self.history_size:
            return
            
        filtered_history: List[List[Dict[str, Any]]] = []
        for history_element in self.history:
            filtered_history_element: List[Dict[str, Any]] = []
            for word in history_element[::-1]:
                if word['start'] >= self.last_committed_time - 0.2:
                    if len(self.committed) > 0 and self._equal_texts(word, self.committed[-1]) and self._words_time_overlap(word, self.committed[-1]):
                        break
                    else:
                        filtered_history_element.append(word)
                else:
                    break
            filtered_history_element = filtered_history_element[::-1]
            filtered_history.append(filtered_history_element)
        
        commit_new_words: bool = True
        not_committed_words: List[Dict[str, Any]] = []

        for i, word in enumerate(filtered_history[-1]):
            matches: int = 1
            for history_element in filtered_history[:-1]:
                for w in history_element:
                    if self._equal_texts(word, w) and self._words_time_overlap(word, w):
                        matches += 1
            if matches >= self.majority_threshold:
                if (
                    len(self.committed) > 0 and i == 0 
                    and self._equal_texts(word, self.committed[-1]) 
                    and self._words_time_overlap(word, self.committed[-1], offset=0)
                ):
                    pass
                else:
                    if commit_new_words:
                        self.committed.append(word)
                        self.last_commited.append(word)
                        self.last_committed_time = word['end']
            else:
                commit_new_words = False
                not_committed_words.append(word)
        
        self.not_committed_words = not_committed_words

    def _equal_texts(self, w1: Dict[str, Any], w2: Dict[str, Any]) -> bool:
        """
        Check if two words have equal or very similar text.
        
        Uses edit distance to allow for minor differences in transcription.
        
        Args:
            w1 (Dict): First word dictionary with 'text' key.
            w2 (Dict): Second word dictionary with 'text' key.
            
        Returns:
            bool: True if the texts are equal or have edit distance <= 1.
        """
        w1_text: str = w1['text'].lower().strip()
        w2_text: str = w2['text'].lower().strip()
                    
        return edit_distance(w1_text, w2_text) <= 1

    def _words_time_overlap(self, w1: Dict[str, Any], w2: Dict[str, Any], offset: float = 0) -> bool:
        """
        Check if two words overlap in time, with a small tolerance.
        
        Args:
            w1 (Dict): First word dictionary with 'start' and 'end' keys.
            w2 (Dict): Second word dictionary with 'start' and 'end' keys.
            
        Returns:
            bool: True if the words overlap in time (with tolerance).
        """
        return w2['start'] < w1['end'] + offset and w2['end'] > w1['start'] - offset

    def get_final_text(self) -> str:
        """
        Get the final concatenated text of all committed words.
        
        Returns:
            str: The concatenated text of all committed words.
        """
        return ''.join(w['text'] for w in self.committed)
        
    def get_committed_words(self) -> List[Dict[str, Any]]:
        """
        Get all words that have been committed so far.
        
        Returns:
            List[Dict]: List of all committed word dictionaries.
        """
        return self.committed
    
    def get_last_commited_words(self) -> List[Dict[str, Any]]:
        """
        Get words that were committed in the most recent update.
        
        Returns:
            List[Dict]: List of recently committed word dictionaries.
        """
        return self.last_commited
