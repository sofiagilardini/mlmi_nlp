import nltk
nltk.download('stopwords')
import re

from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))


class CleanText():

    def __init__(self):

        """
        Initialise CleanText class 
        """

        self.stop_words= stop_words

    
    def clean(self, text):

        text = [word.lower() for word in text]

        text = [word for word in text if re.fullmatch(r'\w+', word)]

        # # remove special characters and html tags

        # text = re.sub(r'<.*?>', '', text)
        # text = re.sub(r'[^\w\s]', '', text)
        # text = re.sub(r'[\t\n]', ' ', text)

        # if type(text) == list:
        #     words = text.split()

        filtered_words = [word for word in text if word not in self.stop_words]

        return filtered_words

