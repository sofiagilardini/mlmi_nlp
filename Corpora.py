import os, codecs, sys
from nltk.stem.porter import PorterStemmer
from collections import defaultdict

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
                                    
                                parts = line.strip().split()

                                if len(parts) == 2:
                                    token, pos_tag = parts

                                # stem the token if stemming is enabled

                                if self.stemmer:

                                    token = self.stemmer.stem(token)

                                review_tokens.append((token, pos_tag) if self.pos else token)

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


# --------------------------------------------------------------------------------------------------------------------- #

# from Analysis import Evaluation
# from Analysis import Evaluation

# class SentimentLexicon(Evaluation):
#     def __init__(self):
#         """
#         read in lexicon database and store in self.lexicon
#         """
#         # if multiple entries take last entry by default
#         self.lexicon = self.get_lexicon_dict()

#     def get_lexicon_dict(self):
#         lexicon_dict = {}
#         with open('data/sent_lexicon', 'r') as f:
#             for line in f:
#                 word = line.split()[2].split("=")[1]
#                 polarity = line.split()[5].split("=")[1]
#                 magnitude = line.split()[0].split("=")[1]
#                 lexicon_dict[word] = [magnitude, polarity]
#         return lexicon_dict

#     def classify(self,reviews,threshold,magnitude):
#         """
#         classify movie reviews using self.lexicon.
#         self.lexicon is a dictionary of word: [polarity_info, magnitude_info], e.g. "bad": ["negative","strongsubj"].
#         explore data/sent_lexicon to get a better understanding of the sentiment lexicon.
#         store the predictions in self.predictions as a list of strings where "+" and "-" are correct/incorrect classifications respectively e.g. ["+","-","+",...]

#         @param reviews: movie reviews
#         @type reviews: list of (string, list) tuples corresponding to (label, content)

#         @param threshold: threshold to center decisions on. instead of using 0, there may be a bias in the reviews themselves which could be accounted for.
#                           experiment for good threshold values.
#         @type threshold: integer

#         @type magnitude: use magnitude information from self.lexicon?
#         @param magnitude: boolean
#         """


#         # reset predictions

#         self.predictions = []


#         for review_ in reviews:

#             true_label = review_[0]

#             vibe_checker = 0

#             for word in review_[1]:

#                 # if word in self.lexicon and word not in stopwords.words('english'):

#                 if word in self.lexicon:

#                     nuanced_sentiment = self.lexicon[word][0]

#                     binary_sentiment = self.lexicon[word][1]

                    
#                     # print(f"vibechecker before {word}: {vibe_checker}")

#                     if magnitude:
                        
#                         if nuanced_sentiment == 'strongsubj':
#                             weight = 4
                            

#                         elif nuanced_sentiment == 'weaksubj':
#                             weight = 0.5
                        
#                         # vibe_checker += 1*weight if binary_sentiment == 'positive' else -1*weight

#                         if binary_sentiment == 'positive':
#                             vibe_checker += 1 * weight
                        
#                         elif binary_sentiment == 'negative':
#                             vibe_checker -= 1 * weight
                        

#                     else:
#                         if binary_sentiment == 'positive':
#                             vibe_checker += 1
                        
#                         elif binary_sentiment == 'negative':
#                             vibe_checker -= 1
                
#                     # print(f"vibechecker after {word}: {vibe_checker}")

#             if vibe_checker > threshold:
#                 pred_label = 'POS'

#             else:
#                 pred_label = 'NEG'

#             # print(f'true label: {true_label},  pred label: {pred_label}, vibe counter: {vibe_checker}')
            
#             if true_label == pred_label:
#                 self.predictions.append('+')
            
#             else:
#                 self.predictions.append('-')

#     # classify2 allows for the changing of strong_weight and weak_weight ( I built this )

#     def classify2(self,reviews,threshold,magnitude, strong_weight, weak_weight):
#             """
#             classify movie reviews using self.lexicon.
#             self.lexicon is a dictionary of word: [polarity_info, magnitude_info], e.g. "bad": ["negative","strongsubj"].
#             explore data/sent_lexicon to get a better understanding of the sentiment lexicon.
#             store the predictions in self.predictions as a list of strings where "+" and "-" are correct/incorrect classifications respectively e.g. ["+","-","+",...]

#             @param reviews: movie reviews
#             @type reviews: list of (string, list) tuples corresponding to (label, content)

#             @param threshold: threshold to center decisions on. instead of using 0, there may be a bias in the reviews themselves which could be accounted for.
#                             experiment for good threshold values.
#             @type threshold: integer

#             @type magnitude: use magnitude information from self.lexicon?
#             @param magnitude: boolean
#             """


#             # reset predictions

#             self.predictions = []


#             for review_ in reviews:

#                 true_label = review_[0]

#                 vibe_checker = 0

#                 for word in review_[1]:

#                     # if word in self.lexicon and word not in stopwords.words('english'):

#                     if word in self.lexicon:

#                         nuanced_sentiment = self.lexicon[word][0]

#                         binary_sentiment = self.lexicon[word][1]

                        
#                         # print(f"vibechecker before {word}: {vibe_checker}")

#                         if magnitude:
                            
#                             if nuanced_sentiment == 'strongsubj':
#                                 weight = strong_weight
                                

#                             elif nuanced_sentiment == 'weaksubj':
#                                 weight = weak_weight
                            
#                             # vibe_checker += 1*weight if binary_sentiment == 'positive' else -1*weight

#                             if binary_sentiment == 'positive':
#                                 vibe_checker += 1 * weight
                            
#                             elif binary_sentiment == 'negative':
#                                 vibe_checker -= 1 * weight
                            

#                         else:
#                             if binary_sentiment == 'positive':
#                                 vibe_checker += 1
                            
#                             elif binary_sentiment == 'negative':
#                                 vibe_checker -= 1
                    
#                         # print(f"vibechecker after {word}: {vibe_checker}")

#                 if vibe_checker > threshold:
#                     pred_label = 'POS'

#                 else:
#                     pred_label = 'NEG'

#                 # print(f'true label: {true_label},  pred label: {pred_label}, vibe counter: {vibe_checker}')
                
#                 if true_label == pred_label:
#                     self.predictions.append('+')
                
#                 else:
#                     self.predictions.append('-')

    










# import os, codecs, sys
# from nltk.stem.porter import PorterStemmer

# class MovieReviewCorpus():
#     def __init__(self,stemming,pos):
#         """
#         initialisation of movie review corpus.

#         @param stemming: use porter's stemming?
#         @type stemming: boolean

#         @param pos: use pos tagging?
#         @type pos: boolean
#         """
#         # raw movie reviews
#         self.reviews=[]
#         # held-out train/test set
#         self.train=[]
#         self.test=[]
#         # folds for cross-validation
#         self.folds={}
#         # porter stemmer
#         self.stemmer=PorterStemmer() if stemming else None
#         # part-of-speech tags
#         self.pos=pos
#         # import movie reviews
#         self.get_reviews()

#     def get_reviews(self):
#         """
#         processing of movie reviews.

#         1. parse reviews in data/reviews and store in self.reviews.

#            the format expected for reviews is: [(string,list), ...] e.g. [("POS",["a","good","movie"]), ("NEG",["a","bad","movie"])].
#            in data/reviews there are .tag and .txt files. The .txt files contain the raw reviews and .tag files contain tokenized and pos-tagged reviews.

#            to save effort, we recommend you use the .tag files. you can disregard the pos tags to begin with and include them later.
#            when storing the pos tags, please use the format for each review: ("POS/NEG", [(token, pos-tag), ...]) e.g. [("POS",[("a","DT"), ("good","JJ"), ...])]

#            to use the stemmer the command is: self.stemmer.stem(token)

#         2. store training and held-out reviews in self.train/test. files beginning with cv9 go in self.test and others in self.train

#         3. store reviews in self.folds. self.folds is a dictionary with the format: self.folds[fold_number] where fold_number is an int 0-9.
#            you can get the fold number from the review file name.
#         """

#         directory = 'data/reviews/POS'

#         parts_list = []

#         for filename in os.listdir(directory):

#             if filename.endswith(".tag"):
#                 file_path = os.path.join(directory, filename)

#                 with codecs.open(file_path, 'r', 'utf-8') as file:
#                     lines = file.readlines()

#                     label = "POS" if "POS" in filename else "NEG"

#                     review_tokens = []

#                     for line in lines:
#                         parts = line.strip().split()

#                         if len(parts) == 2:
#                             token, pos_tag = parts

#                             # Stem the token if stemming is enabled

#                             if self.stemmer:

#                                 token = self.stemmer.stem(token)
                            
#                             review_tokens.append((token, pos_tag) if self.pos else token)
            
#                     review = (label, review_tokens)
#                     self.reviews.append(review)


#         return self.reviews



                    

#         return parts_list








# """
# Notes: 

# - PorterStemmer 
#     - stemmers remove morphological affixes from words, leaving only the word stem (running -> run)
# - POS tagging, (Parts-of-Speech) tags - represents the grammatical category of the word its attached to. 



# """