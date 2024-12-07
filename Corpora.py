import os, codecs, sys
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
import re
from preprocessing import CleanText

class MovieReviewCorpus():
    def __init__(self,stemming,pos):
        """
        initialisation of movie review corpus.

        @param stemming: use porter's stemming?
        @type stemming: boolean

        @param pos: use pos tagging?
        @type pos: boolean
        """
        # raw movie reviews
        self.reviews=[]
        # held-out train/test set
        self.train=[]
        self.test=[]
        # folds for cross-validation
        self.folds={}
        # porter stemmer
        self.stemmer=PorterStemmer() if stemming else None
        # part-of-speech tags
        self.pos=pos
        # import movie reviews
        self.get_reviews()

    def get_reviews(self):
        """
        processing of movie reviews.

        1. parse reviews in data/reviews and store in self.reviews.

           the format expected for reviews is: [(string,list), ...] e.g. [("POS",["a","good","movie"]), ("NEG",["a","bad","movie"])].
           in data/reviews there are .tag and .txt files. The .txt files contain the raw reviews and .tag files contain tokenized and pos-tagged reviews.

           to save effort, we recommend you use the .tag files. you can disregard the pos tags to begin with and include them later.
           when storing the pos tags, please use the format for each review: ("POS/NEG", [(token, pos-tag), ...]) e.g. [("POS",[("a","DT"), ("good","JJ"), ...])]

           to use the stemmer the command is: self.stemmer.stem(token)

        2. store training and held-out reviews in self.train/test. files beginning with cv9 go in self.test and others in self.train

        3. store reviews in self.folds. self.folds is a dictionary with the format: self.folds[fold_number] where fold_number is an int 0-9.
           you can get the fold number from the review file name.
        """

        directory = 'data/reviews/'

        parts_list = []

        for item in os.listdir(directory):

            item_path = os.path.join(directory, item)

            if os.path.isdir(item_path):
                
                # here you are at the level of item_path being either /reviews/NEG or /reviews/POS

                label = "POS" if "POS" in item_path else "NEG"

                classified_reviews_dir = item_path

                for filename in os.listdir(classified_reviews_dir):

                    if filename.endswith(".tag"):
                        file_path = os.path.join(classified_reviews_dir, filename)

                        with codecs.open(file_path, 'r', 'utf-8') as file:

                            lines = file.readlines()

                            review_tokens = []


                            for line in lines:

                                line = line.strip()

                                # if " " in line:

                                if not line:
                                    continue

                                
                                token, pos_tag = line.rsplit(maxsplit = 1)
                                    
                                # parts = line.strip().split()

                                # if len(parts) == 2:
                                #     token, pos_tag = parts

                                # stem the token if stemming is enabled

                                if self.stemmer:

                                    token = self.stemmer.stem(token)

                                review_tokens.append((token, pos_tag) if self.pos else token)
                            

                            breakpoint()

                            cleaner = CleanText()

                            review_tokens = cleaner.clean(review_tokens)

                            breakpoint()

                            review = (label, review_tokens)
                            self.reviews.append(review)

                        if filename.startswith('cv9'):
                            self.test.append(review)
                        
                        else:
                            self.train.append(review)
                        
                        fold_num = int(filename[2])

                        if fold_num not in self.folds:
                            self.folds[fold_num] = []

                        self.folds[fold_num].append(review)

        return self.reviews



    
"""
Notes: 
- PorterStemmer 
    - stemmers remove morphological affixes from words, leaving only the word stem (running -> run)
- POS tagging, (Parts-of-Speech) tags - represents the grammatical category of the word its attached to. 

"""

