import os
from subprocess import call
from nltk.util import ngrams
from Analysis import Evaluation
import numpy as np
from sklearn import svm
from collections import defaultdict
import time

class NaiveBayesText(Evaluation):
    def __init__(self,smoothing,bigrams,trigrams,discard_closed_class):
        """
        initialisation of NaiveBayesText classifier.

        @param smoothing: use smoothing?
        @type smoothing: booleanp

        @param bigrams: add bigrams?
        @type bigrams: boolean

        @param trigrams: add trigrams?
        @type trigrams: boolean

        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean
        """
        # set of features for classifier
        self.vocabulary=set()
        # prior probability
        self.prior={}
        # conditional probablility
        self.condProb={}
        # use smoothing?
        self.smoothing=smoothing
        # add bigrams?
        self.bigrams=bigrams
        # add trigrams?
        self.trigrams=trigrams
        # restrict unigrams to nouns, adjectives, adverbs and verbs?
        self.discard_closed_class=discard_closed_class
        # stored predictions from test instances
        self.predictions=[]

    def extractVocabulary(self,reviews):
        """
        extract features from training data and store in self.vocabulary.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        for sentiment,review in reviews:
            for token in self.extractReviewTokens(review):
                self.vocabulary.add(token)

    def extractReviewTokens(self,review):
        """
        extract tokens from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of strings
        """
        text=[]
        for token in review:
            # check if pos tags are included in review e.g. ("bad","JJ")
            if len(token)==2 and self.discard_closed_class:
                if token[1][0:2] in ["NN","JJ","RB","VB"]: text.append(token)
            else:
                text.append(token)
        if self.bigrams:
            for bigram in ngrams(review,2): text.append(bigram)
        if self.trigrams:
            for trigram in ngrams(review,3): text.append(trigram)
        return text

    def create_vocab_dict(self):
        vocab_to_id = {}
        for word in self.vocabulary:
            vocab_to_id[word] = len(vocab_to_id)
        return vocab_to_id

    def train(self,reviews):
        """
        train NaiveBayesText classifier.

        1. reset self.vocabulary, self.prior and self.condProb
        2. extract vocabulary (i.e. get features for training)
        3. get prior and conditional probability for each label ("POS","NEG") and store in self.prior and self.condProb
           note: to get conditional concatenate all text from reviews in each class and calculate token frequencies
                 to speed this up simply do one run of the movie reviews and count token frequencies if the token is in the vocabulary,
                 then iterate the vocabulary and calculate conditional probability (i.e. don't read the movie reviews in their entirety
                 each time you need to calculate a probability for each token in the vocabulary)

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        """


        self.vocabulary = set()
        self.prior = {}
        self.condProb = {"POS" : defaultdict(float), 
                         "NEG" : defaultdict(float)}
        
        self.extractVocabulary(reviews)

        class_counts = {"POS" : 0, "NEG" : 0}
        word_counts = {"POS" : defaultdict(int), "NEG" : defaultdict(int)}

        # Count occurences 

        for label, review in reviews:
            class_counts[label] += 1

            for token in self.extractReviewTokens(review):
                if token in self.vocabulary:
                    word_counts[label][token] += 1


        # Calculate prior probabilities

        total_reviews = len(reviews)

        for label in class_counts:
            self.prior[label] = class_counts[label] / total_reviews

        # Calculate conditional probabilities

        alpha = 1 if self.smoothing else 0 

        for label in word_counts:

            total_word_count_in_class = sum(word_counts[label].values())

            for word in self.vocabulary:

                self.condProb[label][word] = (
                    (word_counts[label][word] + alpha) / (total_word_count_in_class + alpha * len(self.vocabulary))
                )



    def test(self,reviews):
        """
        test NaiveBayesText classifier and store predictions in self.predictions.
        self.predictions should contain a "+" if prediction was correct and "-" otherwise.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """

        # self.predictions = []

        i = 0

        for label, review in reviews:


            condProbReview_POS = np.log(self.prior["POS"])
            condProbReview_NEG = np.log(self.prior["NEG"])

            for token in self.extractReviewTokens(review):

                if token in self.vocabulary:
                    condProbReview_POS += np.log(self.condProb["POS"][token])
                    condProbReview_NEG+= np.log(self.condProb["NEG"][token])

            i += 1

            
            if condProbReview_POS > condProbReview_NEG:
                prediction = "POS"
            else:
                prediction = "NEG"
            
            if prediction == label:
                self.predictions.append('+')
            else:
                self.predictions.append('-')


class SVMText(Evaluation):
    def __init__(self,bigrams,trigrams,discard_closed_class):
        """
        initialisation of SVMText object

        @param bigrams: add bigrams?
        @type bigrams: boolean

        @param trigrams: add trigrams?
        @type trigrams: boolean

        @param svmlight_dir: location of smvlight binaries
        @type svmlight_dir: string

        @param svmlight_dir: location of smvlight binaries
        @type svmlight_dir: string

        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean
        """
        self.svm_classifier = svm.SVC()
        self.predictions=[]
        self.vocabulary=set()
        # add in bigrams?
        self.bigrams=bigrams
        # add in trigrams?
        self.trigrams=trigrams
        # restrict to nouns, adjectives, adverbs and verbs?
        self.discard_closed_class=discard_closed_class

    def extractVocabulary(self,reviews):
        self.vocabulary = set()
        for sentiment, review in reviews:
            for token in self.extractReviewTokens(review):
                 self.vocabulary.add(token)

    def extractReviewTokens(self,review):
        """
        extract tokens from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of strings
        """
        text=[]
        for term in review:
            # check if pos tags are included in review e.g. ("bad","JJ")
            if len(term)==2 and self.discard_closed_class:
                if term[1][0:2] in ["NN","JJ","RB","VB"]: text.append(term)
            else:
                text.append(term)
        if self.bigrams:
            for bigram in ngrams(review,2): text.append(term)
        if self.trigrams:
            for trigram in ngrams(review,3): text.append(term)
        return text

    def getFeatures(self,reviews):
        """
        determine features and labels from training reviews.

        1. extract vocabulary (i.e. get features for training)
        2. extract features for each review as well as saving the sentiment
        3. append each feature to self.input_features and each label to self.labels
        (self.input_features will then be a list of list, where the inner list is
        the features)

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """

        self.extractVocabulary(reviews)

        # Each key is a word from self.vocabulary (the set of all unique words) in our vocab list
        # Each value is the index (int)

        vocab_to_id = {word: idx for idx, word in enumerate(self.vocabulary)}

        # Prepare input features and labels
        self.input_features = []
        self.labels = []

        for sentiment, review in reviews:

            # Initalise a BoW vector of the same size as vocabulary

            bow_vector = [0] * len(self.vocabulary)

            # Count occurences of each word in the review

            for token in self.extractReviewTokens(review):
                if token in vocab_to_id:
                    bow_vector[vocab_to_id[token]] += 1

            
            self.input_features.append(bow_vector)

            self.labels.append(1 if sentiment == "POS" else -1)

        # TODO Q6.

    def train(self,reviews):
        """
        train svm. This uses the sklearn SVM module, and further details can be found using
        the sci-kit docs. You can try changing the SVM parameters. 

        @param reviews: training data
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        # function to determine features in training set.
        self.getFeatures(reviews)

        # reset SVM classifier and train SVM model
        self.svm_classifier = svm.SVC()
        self.svm_classifier.fit(self.input_features, self.labels)

    def test(self,reviews):
        """
        test svm

        @param reviews: test data
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """

        vocab_to_id = {word: indx for indx, word in enumerate(self.vocabulary)}

        test_features = []

        for _, review in reviews:

            bow_vector = [0] *len(self.vocabulary)

            # Count occurences of each word in the review

            for token in self.extractReviewTokens(review):
                if token in vocab_to_id:
                    bow_vector[vocab_to_id[token]] += 1
            
            test_features.append(bow_vector)

        

        predictions = self.svm_classifier.predict(test_features)
        true_labels = [1 if sentiment == "POS" else -1 for sentiment, _ in reviews]


        for pred, true in zip(predictions, true_labels):
            self.predictions.append("+" if pred == true else "-")


        # TODO Q6.1



class SVM_Doc2Vec(Evaluation):

    def __init__(self, doc2vec_model):

        """
        
        Initialise the SVM_Doc2Vec object. 

        @param doc2vec_model : Pre-trained Doc2Vec model for generating embeddings
        """

        self.svm_classifier = svm.SVC()
        self.predictions=[]
        self.doc2vec_model = doc2vec_model


    def getFeatures(self, reviews):
        
        """

        Extract features and labels for training/testing.

        1. Infer document vectors using Doc2Vec model/
        2. Extract corresponding sentiment labels
        
        
        """

        self.input_features = []
        self.labels = []


        for sentiment, review in reviews:
            
            doc_vector = self.doc2vec_model.infer_vector(review)

            self.input_features.append(doc_vector)

            self.labels.append(1 if sentiment == "POS" else -1)

    
    def train(self, reviews):
        
        """
        
        Train the SVM model using Doc2Vec embeddings

        @param reviews: Training data (list of (sentiment, review) list

        """

        self.getFeatures(reviews)

        print("Training SVM")
        start_time = time.time()
        self.svm_classifier.fit(self.input_features, self.labels)
        print(f"SVM training complete: {time.time()- start_time} seconds")

    
    def test(self, reviews):

        """

        Test the SVM model using Doc2Vec embeddings

        @param reviews: Test data, list of (sentiment, review)
        
        
        """


        test_features = [self.doc2vec_model.infer_vector(review) for _, review in reviews]

        # Predict labels for the test set 

        predictions = self.svm_classifier.predict(test_features)
        true_labels = [1 if sentiment == "POS" else -1 for sentiment, _ in reviews]

        self.predictions = ["+" if pred == true else "-" for pred, true in zip(predictions, true_labels)]

        correct = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
        accuracy = correct / len(true_labels)
        print(f"Test Accuracy: {accuracy:.3f}")

        return accuracy





        


    















































# import os
# from subprocess import call
# from nltk.util import ngrams
# from Analysis import Evaluation
# import numpy as np
# from sklearn import svm

# class NaiveBayesText(Evaluation):
#     def __init__(self,smoothing,bigrams,trigrams,discard_closed_class):
#         """
#         initialisation of NaiveBayesText classifier.

#         @param smoothing: use smoothing?
#         @type smoothing: booleanp

#         @param bigrams: add bigrams?
#         @type bigrams: boolean

#         @param trigrams: add trigrams?
#         @type trigrams: boolean

#         @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
#         @type discard_closed_class: boolean
#         """
#         # set of features for classifier
#         self.vocabulary=set()
#         # prior probability
#         self.prior={}
#         # conditional probablility
#         self.condProb={}
#         # use smoothing?
#         self.smoothing=smoothing
#         # add bigrams?
#         self.bigrams=bigrams
#         # add trigrams?
#         self.trigrams=trigrams
#         # restrict unigrams to nouns, adjectives, adverbs and verbs?
#         self.discard_closed_class=discard_closed_class
#         # stored predictions from test instances
#         self.predictions=[]

#     def extractVocabulary(self,reviews):
#         """
#         extract features from training data and store in self.vocabulary.

#         @param reviews: movie reviews
#         @type reviews: list of (string, list) tuples corresponding to (label, content)
#         """
#         for sentiment,review in reviews:
#             for token in self.extractReviewTokens(review):
#                 self.vocabulary.add(token)

#     def extractReviewTokens(self,review):
#         """
#         extract tokens from reviews.

#         @param reviews: movie reviews
#         @type reviews: list of (string, list) tuples corresponding to (label, content)

#         @return: list of strings
#         """
#         text=[]
#         for token in review:
#             # check if pos tags are included in review e.g. ("bad","JJ")
#             if len(token)==2 and self.discard_closed_class:
#                 if token[1][0:2] in ["NN","JJ","RB","VB"]: text.append(token)
#             else:
#                 text.append(token)
#         if self.bigrams:
#             for bigram in ngrams(review,2): text.append(bigram)
#         if self.trigrams:
#             for trigram in ngrams(review,3): text.append(trigram)
#         return text

#     def create_vocab_dict(self):
#         vocab_to_id = {}
#         for word in self.vocabulary:
#             vocab_to_id[word] = len(vocab_to_id)
#         return vocab_to_id

#     def train(self,reviews):
#         """
#         train NaiveBayesText classifier.

#         1. reset self.vocabulary, self.prior and self.condProb
#         2. extract vocabulary (i.e. get features for training)
#         3. get prior and conditional probability for each label ("POS","NEG") and store in self.prior and self.condProb
#            note: to get conditional concatenate all text from reviews in each class and calculate token frequencies
#                  to speed this up simply do one run of the movie reviews and count token frequencies if the token is in the vocabulary,
#                  then iterate the vocabulary and calculate conditional probability (i.e. don't read the movie reviews in their entirety
#                  each time you need to calculate a probability for each token in the vocabulary)

#         @param reviews: movie reviews
#         @type reviews: list of (string, list) tuples corresponding to (label, content)
#         """
#         # TODO Q1
#         # TODO Q2 (use switch for smoothing from self.smoothing)

#     def test(self,reviews):
#         """
#         test NaiveBayesText classifier and store predictions in self.predictions.
#         self.predictions should contain a "+" if prediction was correct and "-" otherwise.

#         @param reviews: movie reviews
#         @type reviews: list of (string, list) tuples corresponding to (label, content)
#         """
#         # TODO Q1

# class SVMText(Evaluation):
#     def __init__(self,bigrams,trigrams,discard_closed_class):
#         """
#         initialisation of SVMText object

#         @param bigrams: add bigrams?
#         @type bigrams: boolean

#         @param trigrams: add trigrams?
#         @type trigrams: boolean

#         @param svmlight_dir: location of smvlight binaries
#         @type svmlight_dir: string

#         @param svmlight_dir: location of smvlight binaries
#         @type svmlight_dir: string

#         @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
#         @type discard_closed_class: boolean
#         """
#         self.svm_classifier = svm.SVC()
#         self.predictions=[]
#         self.vocabulary=set()
#         # add in bigrams?
#         self.bigrams=bigrams
#         # add in trigrams?
#         self.trigrams=trigrams
#         # restrict to nouns, adjectives, adverbs and verbs?
#         self.discard_closed_class=discard_closed_class

#     def extractVocabulary(self,reviews):
#         self.vocabulary = set()
#         for sentiment, review in reviews:
#             for token in self.extractReviewTokens(review):
#                  self.vocabulary.add(token)

#     def extractReviewTokens(self,review):
#         """
#         extract tokens from reviews.

#         @param reviews: movie reviews
#         @type reviews: list of (string, list) tuples corresponding to (label, content)

#         @return: list of strings
#         """
#         text=[]
#         for term in review:
#             # check if pos tags are included in review e.g. ("bad","JJ")
#             if len(term)==2 and self.discard_closed_class:
#                 if term[1][0:2] in ["NN","JJ","RB","VB"]: text.append(term)
#             else:
#                 text.append(term)
#         if self.bigrams:
#             for bigram in ngrams(review,2): text.append(term)
#         if self.trigrams:
#             for trigram in ngrams(review,3): text.append(term)
#         return text

#     def getFeatures(self,reviews):
#         """
#         determine features and labels from training reviews.

#         1. extract vocabulary (i.e. get features for training)
#         2. extract features for each review as well as saving the sentiment
#         3. append each feature to self.input_features and each label to self.labels
#         (self.input_features will then be a list of list, where the inner list is
#         the features)

#         @param reviews: movie reviews
#         @type reviews: list of (string, list) tuples corresponding to (label, content)
#         """

#         self.input_features = []
#         self.labels = []

#         # TODO Q6.

#     def train(self,reviews):
#         """
#         train svm. This uses the sklearn SVM module, and further details can be found using
#         the sci-kit docs. You can try changing the SVM parameters. 

#         @param reviews: training data
#         @type reviews: list of (string, list) tuples corresponding to (label, content)
#         """
#         # function to determine features in training set.
#         self.getFeatures(reviews)

#         # reset SVM classifier and train SVM model
#         self.svm_classifier = svm.SVC()
#         self.svm_classifier.fit(self.input_features, self.labels)

#     def test(self,reviews):
#         """
#         test svm

#         @param reviews: test data
#         @type reviews: list of (string, list) tuples corresponding to (label, content)
#         """

#         # TODO Q6.1
