from Corpora import MovieReviewCorpus
from Lexicon import SentimentLexicon
from Statistics import SignTest
from Classifiers import NaiveBayesText, SVMText
from Extensions import SVMDoc2Vec
from InfoStore import figurePlotting, resultsWrite

from imdb_analysis import IMDBLoader, Doc2VecTrainer, SVMClassifier


import multiprocessing

print(f"CPU available: {multiprocessing.cpu_count()}")


corpus=MovieReviewCorpus(stemming=False,pos=False)

# Step 1: Load labeled IMDB data for Doc2Vec training
imdb_loader = IMDBLoader('data/aclImdb')
imdb_tagged_docs = imdb_loader.load_reviews()

# Step 2: Train Doc2Vec on IMDB labeled data
doc2vec_trainer = Doc2VecTrainer(vector_size=100, window=5, epochs=200)
doc2vec_trainer.train(imdb_tagged_docs)

# Step 3: Prepare embeddings for IMDB training data
imdb_vectors = [doc2vec_trainer.infer_vector(doc.words) for doc in imdb_tagged_docs]
imdb_labels = [1 if "pos" in doc.tags[0] else -1 for doc in imdb_tagged_docs]

# Step 4: Infer embeddings for the `corpus` dataset
train_vectors = [doc2vec_trainer.infer_vector(review.split()) for _, review in corpus.train]
train_labels = [1 if sentiment == "POS" else -1 for sentiment, _ in corpus.train]

test_vectors = [doc2vec_trainer.infer_vector(review.split()) for _, review in corpus.test]
test_labels = [1 if sentiment == "POS" else -1 for sentiment, _ in corpus.test]

# Step 5: Train SVM on IMDB data and test on corpus
svm_classifier = SVMClassifier(kernel='linear', C=1.0)
svm_classifier.train(imdb_vectors, imdb_labels)  # Train on IMDB embeddings

# Test on `corpus`
accuracy = svm_classifier.evaluate(test_vectors, test_labels)
print(f"Final Accuracy on corpus: {accuracy:.3f}")
