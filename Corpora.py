import os, codecs, sys
from nltk.stem.porter import PorterStemmer

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

        directory = 'data/reviews/POS'

        parts_list = []

        for filename in os.listdir(directory):

            if filename.endswith(".tag"):
                file_path = os.path.join(directory, filename)

                with codecs.open(file_path, 'r', 'utf-8') as file:
                    lines = file.readlines()

                    label = "POS" if "POS" in filename else "NEG"

                    review_tokens = []

                    for line in lines:
                        parts = line.strip().split()

                        if len(parts) == 2:
                            token, pos_tag = parts

                            # Stem the token if stemming is enabled

                            if self.stemmer:

                                token = self.stemmer.stem(token)
                            
                            review_tokens.append((token, pos_tag) if self.pos else token)
            
                    review = (label, review_tokens)
                    self.reviews.append(review)


        return self.reviews



                    

        return parts_list








"""
Notes: 

- PorterStemmer 
    - stemmers remove morphological affixes from words, leaving only the word stem (running -> run)
- POS tagging, (Parts-of-Speech) tags - represents the grammatical category of the word its attached to. 



"""