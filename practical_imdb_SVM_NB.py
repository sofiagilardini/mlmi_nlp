from Corpora import MovieReviewCorpus
from Statistics import SignTest
from Classifiers import NaiveBayesText, SVM_Doc2Vec
from imdb_analysis import IMDBLoader
from itertools import product
import os
import time
from gensim.models import Doc2Vec
import multiprocessing
import csv

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

dir = 'doc2vec_results'
if os.path.exists(dir) == False:
    os.makedirs(dir)

# Initialize results storage
results = resultsWrite(f"{dir}/IMDB_Results.txt")
results.refreshResults()

trainingResults = resultsWrite(f"{dir}/Doc2VecTrain.txt")
trainingResults.refreshResults()

print(f"CPU available: {multiprocessing.cpu_count()}")
workers = multiprocessing.cpu_count() - 2

# Define the corpus
corpus = MovieReviewCorpus(stemming=False, pos=False)

# Define CSV file for results export
csv_file = "f{dir}/Doc2Vec_test_results.csv"

# Baseline: Naive Bayes with MovieReviewCorpus (stemming=False, pos=False)
test_ID = "Baseline, 000"
NB_baseline = NaiveBayesText(smoothing=True, bigrams=False, trigrams=False, discard_closed_class=False)
NB_baseline.crossValidate_nb_svm(corpus, test_ID=test_ID, results_path="Ignore.txt")
baseline_preds = NB_baseline.predictions
baseline_avg_acc = NB_baseline.getAccuracy()
baseline_std = NB_baseline.getStdDeviation()

# Load labeled IMDB data for Doc2Vec training
imdb_loader = IMDBLoader('data/aclImdb')
imdb_tagged_docs = imdb_loader.load_reviews()

# Parameter grid for Doc2Vec training
parameter_grid = {
    'dm': [0, 1],
    'vector_size': [50, 100],
    'window': [5, 10],
    'min_count': [1, 2, 4],
    'epochs': [25, 50, 75]
}


# Directory to save Doc2Vec models
model_dir = 'Doc2Vec_Models'
os.makedirs(model_dir, exist_ok=True)

# Generate parameter combinations
param_combinations = list(product(
    parameter_grid["dm"],
    parameter_grid["vector_size"],
    parameter_grid["window"],
    parameter_grid["min_count"],
    parameter_grid["epochs"]
))

# Function to train and save a Doc2Vec model
def train_and_save_model(tagged_documents, save_dir, dm, vector_size, window, min_count, epochs):
    model_filename = f"doc2vec_dm{dm}_vec{vector_size}_win{window}_min{min_count}_epochs{epochs}.model"
    model_path = os.path.join(save_dir, model_filename)

    if os.path.exists(model_path):
        print(f"Model {model_filename} already exists. Skipping training.")
        return Doc2Vec.load(model_path)

    print(f"Training model: {model_filename}...")
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

    print(f"Finished training model: {model_filename} (took {(time.time() - start_time):.2f} seconds)")
    model.save(model_path)
    print(f"Model saved: {model_path}\n")
    return model

# Function to load a trained Doc2Vec model
def load_model(save_dir, dm, vector_size, window, min_count, epochs):
    model_id = f"doc2vec_dm{dm}_vec{vector_size}_win{window}_min{min_count}_epochs{epochs}"
    model_filename = f"{model_id}.model"
    model_path = os.path.join(save_dir, model_filename)

    if os.path.exists(model_path):
        print(f"Loading model: {model_filename}")
        return model_id, Doc2Vec.load(model_path)
    else:
        raise FileNotFoundError(f"Model {model_filename} does not exist.")

# Open CSV file for writing
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Doc2Vec_ModelID", "SVM_Avg_Accuracy", "SVM_Std", "P-Value"
    ])
    writer.writerow(["Baseline", f"{baseline_avg_acc:.2f}", f"{baseline_std:.2f}", "baseline"])

    # Train and evaluate Doc2Vec models
    for dm, vector_size, window, min_count, epochs in param_combinations:
        model_id, doc2vec_model = load_model(
            save_dir=model_dir,
            dm=dm,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs
        )

        results.savePrint_noQ(f'----------- ** ModelID: {model_id} ** ----------------')

        # Evaluate SVM with Doc2Vec embeddings
        SVM_d2v = SVM_Doc2Vec(doc2vec_model=doc2vec_model)
        SVM_d2v.crossValidate_Doc2Vec(corpus, modelID=model_id)
        svm_avg_acc = SVM_d2v.getAccuracy()
        svm_std = SVM_d2v.getStdDeviation()

        # Calculate p-value using SignTest
        signTest = SignTest()
        p_value = f"{signTest.getSignificance(SVM_d2v.predictions, baseline_preds):.3f}"

        # Write results to CSV
        writer.writerow([
            model_id, f"{svm_avg_acc:.2f}", f"{svm_std:.2f}", p_value
        ])

print(f"Results have been exported to {csv_file}.")





