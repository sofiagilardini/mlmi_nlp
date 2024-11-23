from Corpora import MovieReviewCorpus
from Lexicon import SentimentLexicon
from Statistics import SignTest
from Classifiers import NaiveBayesText, SVMText
from Extensions import SVMDoc2Vec
from InfoStore import figurePlotting, resultsWrite



results = resultsWrite("Results.txt")
results.refreshResults()

plotting = figurePlotting()



# retrieve corpus
corpus=MovieReviewCorpus(stemming=False,pos=False)

# use sign test for all significance testing
signTest=SignTest()

print_st = "----------- ** classifying reviews using sentiment lexicon ** ----------"

print(print_st)
results.savePrint_noQ(print_st)

del print_st

# read in lexicon
lexicon=SentimentLexicon()



# on average there are more positive than negative words per review (~7.13 more positive than negative per review)
# to take this bias into account will use threshold (roughly the bias itself) to make it harder to classify as positive
threshold=8

# question 0.1
lexicon.classify(corpus.reviews,threshold,magnitude=False)
token_preds=lexicon.predictions

Q_no = "Q 0.1 part_a"
print_st = f"token-only results: {lexicon.getAccuracy():.2f}"
print(print_st)
results.savePrint(Q_no, print_st)

del print_st, Q_no



Q_no="Q 0.1 part_b"

lexicon.classify(corpus.reviews,threshold,magnitude=True)
magnitude_preds=lexicon.predictions

print_st = f"magnitude results:{lexicon.getAccuracy():.2f}"
print(print_st)
results.savePrint(Q_no, print_st)

del print_st, Q_no



# question 0.2
Q_no = "Q 0.2"
p_value=signTest.getSignificance(token_preds,magnitude_preds)
significance = "significant" if p_value < 0.05 else "not significant"

print_st = f"-> Magnitude lexicon results are {significance} with respect to token-only"
print(print_st)
results.savePrint(Q_no, print_st)

del print_st, Q_no




# ------ plot heatmap ------- #

plotting.plotHeatmap(threshold=threshold)

# ------- end of heatmap --------- # 

# question 1.0

Q_no = "Q 1.0"

print_st = "--------- ** classifying reviews using Naive Bayes on held-out test set ** -----------"
print(print_st)
results.savePrint_noQ(print_st)

del print_st


NB=NaiveBayesText(smoothing=False,bigrams=False,trigrams=False,discard_closed_class=False)
NB.train(corpus.train)
NB.test(corpus.test)
# store predictions from classifier
non_smoothed_preds=NB.predictions

print_st = f"Accuracy without smoothing: {NB.getAccuracy():.2f}"
print(print_st)
results.savePrint(Q_no, print_st)

del print_st, Q_no



# question 2.0
# use smoothing

Q_no = "Q 2.0"

NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=False,discard_closed_class=False)
NB.train(corpus.train)
NB.test(corpus.test)
smoothed_preds=NB.predictions
# saving this for use later
num_non_stemmed_features=len(NB.vocabulary)

print_st = f"Accuracy using smoothing: {NB.getAccuracy():.2f}"
print(print_st)
results.savePrint(Q_no, print_st)

del print_st, Q_no




# question 2.1
# see if smoothing significantly improves results

Q_no = "Q 2.1"

p_value=signTest.getSignificance(non_smoothed_preds,smoothed_preds)
significance = "significant" if p_value < 0.05 else "not significant"

print_st = f"-> Results using smoothing are {significance} with respect to no smoothing"
print(print_st)
results.savePrint(Q_no, print_st)

del print_st, Q_no


# breakpoint()

# question 3.0
print("--- classifying reviews using 10-fold cross-evaluation ---")
# using previous instantiated object
NB.crossValidate(corpus)
# using cross-eval for smoothed predictions from now on
smoothed_preds=NB.predictions
# print(f"Accuracy: {NB.getAccuracy():.3f}")
# print(f"Std. Dev: {NB.getStdDeviation()}")

breakpoint()

# question 4.0
print("--- stemming corpus ---")
# retrieve corpus with tokenized text and stemming (using porter)
stemmed_corpus=MovieReviewCorpus(stemming=True,pos=False)
print("--- cross-validating NB using stemming ---")
NB.crossValidate(stemmed_corpus)
stemmed_preds=NB.predictions
print(f"Accuracy: {NB.getAccuracy():.3f}")
print(f"Std. Dev: {NB.getStdDeviation():.3f}")

# TODO Q4.1
# see if stemming significantly improves results on smoothed NB

# TODO Q4.2
print("--- determining the number of features before/after stemming ---")

# question Q5.0
# cross-validate model using smoothing and bigrams
print("--- cross-validating naive bayes using smoothing and bigrams ---")
NB=NaiveBayesText(smoothing=True,bigrams=True,trigrams=False,discard_closed_class=False)
NB.crossValidate(corpus)
smoothed_and_bigram_preds=NB.predictions
print(f"Accuracy: {NB.getAccuracy():.2f}") 
print(f"Std. Dev: {NB.getStdDeviation():.2f}")


# see if bigrams significantly improves results on smoothed NB only
p_value=signTest.getSignificance(smoothed_preds,smoothed_and_bigram_preds)
signifance = "significant" if p_value < 0.05 else "not significant"
print(f"results using smoothing and bigrams are {signifance} with respect to smoothing only")


# TODO Q5.1

# TODO Q6 and 6.1
print("--- classifying reviews using SVM 10-fold cross-eval ---")

# TODO Q7
print("--- adding in POS information to corpus ---")
print("--- training svm on word+pos features ----")
print("--- training svm discarding closed-class words ---")

# question 8.0
print("--- using document embeddings ---")
