import nltk
nltk.download('stopwords')
import re

import string

from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))


class CleanText:
    def __init__(self):
        """
        Initialise CleanText class with stopwords and an optional contraction map.
        """
        self.stopwords = set(stopwords.words('english'))
        self.contraction_map = {
            "can't": "can not",
            "won't": "will not",
            "shouldn't": "should not",
            "don't": "do not",
            "isn't": "is not",
            "aren't": "are not",
            "didn't": "did not",
        }

    def clean(self, tokens, include_pos):
        """
        Clean a list of tokens, with or without POS tags.
        :param tokens: List of tokens (with or without POS tags).
        :param include_pos: Boolean indicating whether tokens include POS tags.
        :return: Cleaned list of tokens, in the same format as input.
        """
        cleaned_tokens = []

        for token in tokens:
            if include_pos:
                word, pos_tag = token
            else:
                word = token

            # Lowercase and expand contractions
            word = word.lower()
            if word in self.contraction_map:
                word = self.contraction_map[word]


            if word.strip() == "" or all(char in string.punctuation for char in word):
                continue

            # --- nltk stopwords include very and don't therefore choosing not to remove them ---- # 
            # # Skip stopwords
            # if word in self.stopwords:
            #     continue

            # Append cleaned token
            if include_pos:
                cleaned_tokens.append((word, pos_tag))
            else:
                cleaned_tokens.append(word)

        return cleaned_tokens
