from Corpora import MovieReviewCorpus
from Lexicon import SentimentLexicon
from Statistics import SignTest
from Classifiers import NaiveBayesText, SVMText
from Extensions import SVMDoc2Vec
from InfoStore import figurePlotting, resultsWrite
from preprocessing import CleanText

from itertools import product

import csv
from itertools import product



# Define all possible corpora based on stemming and pos-tagging
corpora = {
    (0, 0): MovieReviewCorpus(stemming=False, pos=False),
    (0, 1): MovieReviewCorpus(stemming=False, pos=True),
    (1, 0): MovieReviewCorpus(stemming=True, pos=False),
    (1, 1): MovieReviewCorpus(stemming=True, pos=True),
}





import csv
from itertools import product

# Define result path and refresh results
resultspath = "SVM_CV_results.txt"
results = resultsWrite(resultspath)
results.refreshResults()

# Define CSV file for results export
csv_file = "SVM_test_results.csv"

# Sign test for significance testing
signTest = SignTest()

# Generate all combinations of binary options for S (stemming), P (pos-tagging), and DCC (discard closed class)
combinations = list(product([0, 1], repeat=3))  # [(0, 0, 0), (0, 0, 1), ...]

# Initialise predictions dictionary to store results
all_predictions = {}

# Open CSV file for writing
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write CSV header
    writer.writerow([
        "TestID", "Stemming", "POS-Tagging", "Discard Closed Class", "Avg Accuracy", "Std Deviation", "P-Value"
    ])

    # Loop through each combination of Stemming (S), POS-Tagging (P), and DCC
    for S, P, DCC in combinations:
        # Retrieve the corresponding corpus
        corpus = corpora[(S, P)]  # Stemmed = S, POS = P

        # Initialise SVM with the given settings
        SVM = SVMText(bigrams=False, trigrams=False, discard_closed_class=bool(DCC))

        # Generate test ID based on S, P, DCC
        test_ID = f"TestID: {S}{P}{DCC}"

        # Cross-validate and store predictions
        SVM.crossValidate_nb_svm(corpus, test_ID=test_ID, results_path=resultspath)
        all_predictions[test_ID] = SVM.predictions

        # Get accuracy and standard deviation
        avg_accuracy = SVM.getAccuracy()
        std_deviation = SVM.getStdDeviation()

        # Calculate p-value (or mark as baseline for TestID: 000)
        if test_ID == "TestID: 000":
            p_value = "baseline"
        else:
            baseline_preds = all_predictions["TestID: 000"]
            p_value = f"{signTest.getSignificance(all_predictions[test_ID], baseline_preds):.3f}"

        # Write results to CSV
        writer.writerow([
            test_ID, bool(S), bool(P), bool(DCC), f"{avg_accuracy:.2f}",
            f"{std_deviation:.2f}", p_value
        ])

# Notify user of CSV export completion
print(f"Results have been exported to {csv_file}.")








import csv
from itertools import product

# Define result path and refresh results
resultspath = "NB_CV_results.txt"
results = resultsWrite(resultspath)
results.refreshResults()

# Define CSV file for results export
csv_file = "NB_test_results.csv"


# Sign test for significance testing
signTest = SignTest()

# Generate all combinations of binary options for B (bigrams), T (trigrams), and S (stemming)
combinations = list(product([0, 1], repeat=3))  # [(0, 0, 0), (0, 0, 1), ...]

# Initialise predictions dictionary to store results
all_predictions = {}

# Open CSV file for writing
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write CSV header
    writer.writerow([
        "TestID", "N-Gram", "Stemming", "Number of Features", "Avg Accuracy", "Std Deviation", "P-Value"
    ])

    # Loop through each combination of Bigrams (B), Trigrams (T), and Stemming (S)
    for B, T, S in combinations:
        # Retrieve the corresponding corpus
        corpus = corpora[(S, 0)]  # Stemmed = S, pos=False (update if POS is used)

        # Initialise Naive Bayes with the given settings
        NB = NaiveBayesText(smoothing=True, bigrams=bool(B), trigrams=bool(T), discard_closed_class=False)

        # Generate test ID based on B, T, S
        test_ID = f"TestID: {B}{T}{S}"

        # Cross-validate and store predictions
        NB.crossValidate_nb_svm(corpus, test_ID=test_ID, results_path=resultspath)
        all_predictions[test_ID] = NB.predictions

        # Calculate feature count
        num_features = len(NB.vocabulary)

        # Get accuracy and standard deviation
        avg_accuracy = NB.getAccuracy()
        std_deviation = NB.getStdDeviation()

        # Determine n-gram type
        ngram_type = (
            "uni" if B == 0 and T == 0 else
            "bi" if B == 1 and T == 0 else
            "tri" if B == 0 and T == 1 else
            "bi+tri"
        )

        # Calculate p-value (or mark as baseline for TestID: 000)
        if test_ID == "TestID: 000":
            p_value = "baseline"
        else:
            baseline_preds = all_predictions["TestID: 000"]
            p_value = f"{signTest.getSignificance(all_predictions[test_ID], baseline_preds):.3f}"

        # Write results to CSV
        writer.writerow([
            test_ID, ngram_type, bool(S), num_features, f"{avg_accuracy:.2f}",
            f"{std_deviation:.2f}", p_value
        ])

# Notify user of CSV export completion
print(f"Results have been exported to {csv_file}.")


























# resultspath = "NB_CV_results.txt"
# results = resultsWrite(resultspath)
# results.refreshResults()

# plotting = figurePlotting()


# # Define result path and refresh results
# resultspath = "NB_CV_results.txt"
# results = resultsWrite(resultspath)
# results.refreshResults()

# plotting = figurePlotting()

# # Define all possible corpora based on stemming and pos-tagging
# corpora = {
#     (0, 0): MovieReviewCorpus(stemming=False, pos=False),
#     (0, 1): MovieReviewCorpus(stemming=False, pos=True),
#     (1, 0): MovieReviewCorpus(stemming=True, pos=False),
#     (1, 1): MovieReviewCorpus(stemming=True, pos=True),
# }

# # Sign test for significance testing
# signTest = SignTest()

# # Generate all combinations of binary options for B, T, S
# combinations = list(product([0, 1], repeat=3))  # [(0, 0, 0), (0, 0, 1), ...]

# # Initialise predictions dictionary to store results
# all_predictions = {}

# # Loop through each combination of Bigrams (B), Trigrams (T), and Stemmed (S)
# for B, T, S in combinations:
#     # Retrieve the corresponding corpus
#     corpus = corpora[(S, 0)]  # Stemmed = S, pos=False (update if POS is used)

#     # Initialise Naive Bayes with the given settings
#     NB = NaiveBayesText(smoothing=True, bigrams=bool(B), trigrams=bool(T), discard_closed_class=False)

#     # Generate test ID based on B, T, S
#     test_ID = f"TestID: {B}{T}{S}"

#     # Cross-validate and store predictions
#     NB.crossValidate_nb_svm(corpus, test_ID=test_ID, results_path=resultspath)
#     all_predictions[test_ID] = NB.predictions

#     # If it's not the baseline (000), calculate significance against baseline
#     if test_ID != "TestID: 000":
#         baseline_preds = all_predictions["TestID: 000"]
#         p_value = signTest.getSignificance(all_predictions[test_ID], baseline_preds)
#         significance = "significant" if p_value < 0.05 else "not significant"

#         # Print and save the results
#         print_st = f"-> P value {test_ID} wrt to 000: {p_value:.3f} ({significance}) <-"
#         print(print_st)
#         results.savePrint_noQ(print_st)




# # Define result path and refresh results
# resultspath = "SVM_CV_results.txt"
# results = resultsWrite(resultspath)
# results.refreshResults()

# plotting = figurePlotting()


# # Sign test for significance testing
# signTest = SignTest()

# # Generate all combinations of binary options for S (stemming), P (pos-tagging), and DCC (discard closed class)
# combinations = list(product([0, 1], repeat=3))  # [(0, 0, 0), (0, 0, 1), ...]

# # Initialise predictions dictionary to store results
# all_predictions = {}

# # Loop through each combination of Stemming (S), POS (P), and Discard Closed Class (DCC)
# for S, P, DCC in combinations:
#     # Retrieve the corresponding corpus
#     corpus = corpora[(S, P)]  # Stemmed = S, POS = P

#     # Initialise SVM with the given settings
#     SVM = SVMText(bigrams=False, trigrams=False, discard_closed_class=bool(DCC))

#     # Generate test ID based on S, P, DCC
#     test_ID = f"TestID: {S}{P}{DCC}"

#     # Cross-validate and store predictions
#     SVM.crossValidate_nb_svm(corpus, test_ID=test_ID, results_path=resultspath)
#     all_predictions[test_ID] = SVM.predictions

#     # If it's not the baseline (000), calculate significance against baseline
#     if test_ID != "TestID: 000":
#         baseline_preds = all_predictions["TestID: 000"]
#         p_value = signTest.getSignificance(all_predictions[test_ID], baseline_preds)
#         significance = "significant" if p_value < 0.05 else "not significant"

#         # Print and save the results
#         print_st = f"-> P value {test_ID} wrt to 000: {p_value:.3f} ({significance}) <-"
#         print(print_st)
#         results.savePrint_noQ(print_st)
