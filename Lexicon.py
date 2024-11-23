
from Analysis import Evaluation
from Analysis import Evaluation

class SentimentLexicon(Evaluation):
    def __init__(self):
        """
        read in lexicon database and store in self.lexicon
        """
        # if multiple entries take last entry by default
        self.lexicon = self.get_lexicon_dict()

    def get_lexicon_dict(self):
        lexicon_dict = {}
        with open('data/sent_lexicon', 'r') as f:
            for line in f:
                word = line.split()[2].split("=")[1]
                polarity = line.split()[5].split("=")[1]
                magnitude = line.split()[0].split("=")[1]
                lexicon_dict[word] = [magnitude, polarity]
        return lexicon_dict

    def classify(self,reviews,threshold,magnitude):
        """
        classify movie reviews using self.lexicon.
        self.lexicon is a dictionary of word: [polarity_info, magnitude_info], e.g. "bad": ["negative","strongsubj"].
        explore data/sent_lexicon to get a better understanding of the sentiment lexicon.
        store the predictions in self.predictions as a list of strings where "+" and "-" are correct/incorrect classifications respectively e.g. ["+","-","+",...]

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @param threshold: threshold to center decisions on. instead of using 0, there may be a bias in the reviews themselves which could be accounted for.
                          experiment for good threshold values.
        @type threshold: integer

        @type magnitude: use magnitude information from self.lexicon?
        @param magnitude: boolean
        """


        # reset predictions

        self.predictions = []


        for review_ in reviews:

            true_label = review_[0]

            vibe_checker = 0

            for word in review_[1]:

                # if word in self.lexicon and word not in stopwords.words('english'):

                if word in self.lexicon:

                    nuanced_sentiment = self.lexicon[word][0]

                    binary_sentiment = self.lexicon[word][1]

                    
                    # print(f"vibechecker before {word}: {vibe_checker}")

                    if magnitude:
                        
                        if nuanced_sentiment == 'strongsubj':
                            weight = 4
                            

                        elif nuanced_sentiment == 'weaksubj':
                            weight = 0.5
                        
                        # vibe_checker += 1*weight if binary_sentiment == 'positive' else -1*weight

                        if binary_sentiment == 'positive':
                            vibe_checker += 1 * weight
                        
                        elif binary_sentiment == 'negative':
                            vibe_checker -= 1 * weight
                        

                    else:
                        if binary_sentiment == 'positive':
                            vibe_checker += 1
                        
                        elif binary_sentiment == 'negative':
                            vibe_checker -= 1
                
                    # print(f"vibechecker after {word}: {vibe_checker}")

            if vibe_checker > threshold:
                pred_label = 'POS'

            else:
                pred_label = 'NEG'

            # print(f'true label: {true_label},  pred label: {pred_label}, vibe counter: {vibe_checker}')
            
            if true_label == pred_label:
                self.predictions.append('+')
            
            else:
                self.predictions.append('-')

    # classify2 allows for the changing of strong_weight and weak_weight ( I built this )

    def classify2(self,reviews,threshold,magnitude, strong_weight, weak_weight):
            """
            classify movie reviews using self.lexicon.
            self.lexicon is a dictionary of word: [polarity_info, magnitude_info], e.g. "bad": ["negative","strongsubj"].
            explore data/sent_lexicon to get a better understanding of the sentiment lexicon.
            store the predictions in self.predictions as a list of strings where "+" and "-" are correct/incorrect classifications respectively e.g. ["+","-","+",...]

            @param reviews: movie reviews
            @type reviews: list of (string, list) tuples corresponding to (label, content)

            @param threshold: threshold to center decisions on. instead of using 0, there may be a bias in the reviews themselves which could be accounted for.
                            experiment for good threshold values.
            @type threshold: integer

            @type magnitude: use magnitude information from self.lexicon?
            @param magnitude: boolean
            """


            # reset predictions

            self.predictions = []


            for review_ in reviews:

                true_label = review_[0]

                vibe_checker = 0

                for word in review_[1]:

                    # if word in self.lexicon and word not in stopwords.words('english'):

                    if word in self.lexicon:

                        nuanced_sentiment = self.lexicon[word][0]

                        binary_sentiment = self.lexicon[word][1]

                        
                        # print(f"vibechecker before {word}: {vibe_checker}")

                        if magnitude:
                            
                            if nuanced_sentiment == 'strongsubj':
                                weight = strong_weight
                                

                            elif nuanced_sentiment == 'weaksubj':
                                weight = weak_weight
                            
                            # vibe_checker += 1*weight if binary_sentiment == 'positive' else -1*weight

                            if binary_sentiment == 'positive':
                                vibe_checker += 1 * weight
                            
                            elif binary_sentiment == 'negative':
                                vibe_checker -= 1 * weight
                            

                        else:
                            if binary_sentiment == 'positive':
                                vibe_checker += 1
                            
                            elif binary_sentiment == 'negative':
                                vibe_checker -= 1
                    
                        # print(f"vibechecker after {word}: {vibe_checker}")

                if vibe_checker > threshold:
                    pred_label = 'POS'

                else:
                    pred_label = 'NEG'

                # print(f'true label: {true_label},  pred label: {pred_label}, vibe counter: {vibe_checker}')
                
                if true_label == pred_label:
                    self.predictions.append('+')
                
                else:
                    self.predictions.append('-')

    













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
#         # reset predictions
#         self.predictions=[]
#         # TODO Q0
