from Corpora import MovieReviewCorpus
from Lexicon import SentimentLexicon
from Statistics import SignTest
from Classifiers import NaiveBayesText, SVMText, SVM_Doc2Vec
from Extensions import SVMDoc2Vec
from InfoStore import figurePlotting, resultsWrite

from imdb_analysis import IMDBLoader, Doc2VecTrainer, SVMClassifier
from itertools import product
import os
import time

from gensim.models import Doc2Vec

import multiprocessing

results = resultsWrite("IMDB_Results.txt")
results.refreshResults()

trainingResults = resultsWrite("Doc2VecTrain.txt")


print(f"CPU available: {multiprocessing.cpu_count()}")

workers = multiprocessing.cpu_count() - 2


corpus=MovieReviewCorpus(stemming=False,pos=False)

# Load labeled IMDB data for Doc2Vec training
imdb_loader = IMDBLoader('data/aclImdb')
imdb_tagged_docs = imdb_loader.load_reviews()


# ---------------- # 


def train_and_save_model(tagged_documents, save_dir, dm, vector_size, window, min_count, epochs):
    """
    Train a Doc2Vec model and save it to a file.
    Skip training if the model file already exists.

    :param tagged_documents: List of TaggedDocument objects for training
    :param save_dir: Directory to save the models
    :param dm: 0 or 1, training algorithm (DBOW or DM)
    :param vector_size: Size of the document embeddings
    :param window: Window size for context words
    :param min_count: Minimum word frequency to include in the vocabulary
    :param epochs: Number of training epochs
    """
    # Construct the model filename based on parameters
    model_filename = f"doc2vec_dm{dm}_vec{vector_size}_win{window}_min{min_count}_epochs{epochs}.model"
    model_path = os.path.join(save_dir, model_filename)

    # Check if the model file already exists
    if os.path.exists(model_path):
        print(f"Model {model_filename} already exists. Skipping training.")
        return Doc2Vec.load(model_path)  # Load and return the existing model

    # Train the Doc2Vec model
    trainingResults.savePrint_noQ(f"Training model: {model_filename}...")
    start_time = time.time()

    model = Doc2Vec(
        tagged_documents,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        dm=dm,
        epochs=epochs,
    )

    trainingResults.savePrint_noQ(f"Finished training model: {model_filename}: took {time.time() - start_time} seconds")

    # Save the model to disk
    model.save(model_path)
    trainingResults.savePrint_noQ(f"Model saved: {model_path}")
    trainingResults.savePrint_noQ('\n')

    return model


def load_model(save_dir, dm, vector_size, window, min_count, epochs):
    """
    Load a previously saved Doc2Vec model.

    :param save_dir: Directory where models are saved
    :param dm: Training algorithm
    :param vector_size: Dimensionality of embeddings
    :param window: Context window size
    :param min_count: Minimum word frequency
    :return: Loaded Doc2Vec model
    """

    model_id = f"doc2vec_dm{dm}_vec{vector_size}_win{window}_min{min_count}_epochs{epochs}"

    model_filename = f"{model_id}.model"
    model_path = os.path.join(save_dir, model_filename)

    if os.path.exists(model_path):
        print(f"Loading model: {model_filename}")
        return model_id, Doc2Vec.load(model_path)
    else:
        breakpoint()
        raise FileNotFoundError(f"Model {model_filename} does not exist.")
    

parameter_grid = {

    'dm' : [0, 1], 
    'vector_size' : [50, 100],
    'window' : [5, 10],
    'min_count' : [2, 4], 
    'epochs' : [3, 5]
}

model_dir = 'Doc2Vec_Models'

os.makedirs(model_dir, exist_ok=True)

param_combinations = list(product(parameter_grid["dm"],
                                  parameter_grid["vector_size"],
                                  parameter_grid["window"],
                                  parameter_grid["min_count"],
                                  parameter_grid["epochs"]))

for dm, vector_size, window, min_count, epochs in param_combinations:
    model = train_and_save_model(
        tagged_documents=imdb_tagged_docs,  
        save_dir=model_dir,
        dm=dm,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs)
    


for dm, vector_size, window, min_count, epochs in param_combinations:


    model_id, doc2vec_model = load_model(
        save_dir=model_dir, 
        dm = dm, 
        vector_size=vector_size, 
        window=window, 
        min_count = min_count, 
        epochs = epochs
    )


    Q_no = f'Q 8: ModelID: {model_id}'

    results.savePrint_noQ(f'----------- ** {Q_no} ** ----------------')



    SVM_d2v = SVM_Doc2Vec(doc2vec_model=doc2vec_model)
    SVM_d2v.crossValidate(corpus, Q_no)

    results.savePrint_noQ("\n")
    results.savePrint_noQ("Average of performances across folds:")
    results.savePrint_noQ(f"Accuracy across folds: {SVM_d2v.getAccuracy():.3f}")
    results.savePrint_noQ(f"Std. Dev across folds: {SVM_d2v.getStdDeviation():.3f}")
    results.space()
    results.space()




