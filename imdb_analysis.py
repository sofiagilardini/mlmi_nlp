from Corpora import MovieReviewCorpus
from Lexicon import SentimentLexicon
from Statistics import SignTest
from Classifiers import NaiveBayesText, SVMText
from Extensions import SVMDoc2Vec
from InfoStore import figurePlotting, resultsWrite

import time
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC



class IMDBLoader:
    def __init__(self, data_dir):
        """
        Initialize the IMDBLoader with the dataset directory.

        :param data_dir: Path to the IMDB dataset
        """
        self.data_dir = data_dir

    def load_reviews(self):
        """
        Load labeled IMDB reviews as TaggedDocument objects.

        :return: List of TaggedDocument objects (labeled data only)
        """
        tagged_docs = []
        for split in ['train', 'test']:  # Loop over train and test splits
            split_path = os.path.join(self.data_dir, split)

            # Process labeled data (pos and neg)
            for label in ['pos', 'neg']:
                label_path = os.path.join(split_path, label)
                if os.path.exists(label_path):
                    for filename in os.listdir(label_path):
                        if filename.endswith('.txt'):
                            file_path = os.path.join(label_path, filename)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                review = f.read().strip()
                                doc_id = f"{split}_{label}_{filename}"
                                tagged_docs.append(TaggedDocument(words=review.split(), tags=[doc_id]))
        return tagged_docs



class Doc2VecTrainer:
    def __init__(self, vector_size, window, min_count, epochs, workers, dm=1):
        """
        Initialize the Doc2VecTrainer with hyperparameters.

        :param vector_size: Size of the embedding vector
        :param window: Window size for context
        :param min_count: Minimum frequency of words
        :param epochs: Number of training epochs
        :param workers: Number of worker threads
        :param dm: Training algorithm (1 for Distributed Memory, 0 for DBOW)
        """
        self.model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count,
                             workers=workers, dm=dm, epochs=epochs)

    def train(self, tagged_documents, verbose = True):
        """
        Train the Doc2Vec model on the provided TaggedDocument objects.

        :param tagged_documents: List of TaggedDocument objects
        """
        if verbose:
            print("Building vocabulary...")
        start_time = time.time()
        
        self.model.build_vocab(tagged_documents)
        if verbose:
            print(f"Vocabulary built in {time.time() - start_time:.2f} seconds.")

        if verbose:
            print("Training Doc2Vec model...")
        start_time = time.time()
        
        self.model.train(tagged_documents, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        if verbose:
            print(f"Doc2Vec training completed in {time.time() - start_time:.2f} seconds!")


    def infer_vector(self, words):
        """
        Infer a vector for a new document.

        :param words: List of words in the document
        :return: Inferred vector
        """
        return self.model.infer_vector(words)





class SVMClassifier:
    def __init__(self, kernel='linear', C=1.0):
        """
        Initialize the SVM classifier with hyperparameters.

        :param kernel: Kernel type for SVM
        :param C: Regularization parameter
        """
        self.model = SVC(kernel=kernel, C=C)

    def train(self, train_vectors, train_labels, verbose = True):
        """
        Train the SVM model.

        :param train_vectors: List of feature vectors
        :param train_labels: List of corresponding labels
        """

        if verbose:
            print("SVMClassifier training...")

        start_time = time.time()
        self.model.fit(train_vectors, train_labels)

        if verbose:
            print(f"SVM training completed in {time.time() - start_time:.2f} seconds")

    def predict(self, test_vectors):
        """
        Predict labels for the test set.

        :param test_vectors: List of feature vectors
        :return: Predicted labels
        """
        return self.model.predict(test_vectors)

    def evaluate(self, test_vectors, test_labels):
        """
        Evaluate the SVM model and calculate accuracy.

        :param test_vectors: List of feature vectors
        :param test_labels: List of ground-truth labels
        :return: Accuracy score
        """
        predictions = self.predict(test_vectors)
        accuracy = accuracy_score(test_labels, predictions)
        print(f"SVM Accuracy: {accuracy:.3f}")
        return accuracy
