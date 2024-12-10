from Corpora import MovieReviewCorpus
from Lexicon import SentimentLexicon
from Statistics import SignTest
from Classifiers import NaiveBayesText, SVMText
from Extensions import SVMDoc2Vec
from InfoStore import figurePlotting, resultsWrite
from preprocessing import CleanText



resultsdoc = "Results.txt"
results = resultsWrite(resultsdoc)
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


results.savePrint_noQ("\n \n")


# ------ plot heatmap ------- #

plotting.plotHeatmap(threshold=threshold)

# ------- end of heatmap --------- # 

# question 1.0

Q_no = "Q 1.0"

print_st = "--------- ** classifying reviews using Naive Bayes on held-out test set ** -----------"
print(print_st)
results.savePrint_noQ(print_st)

del print_st

results.savePrint_noQ("\n")


NB=NaiveBayesText(smoothing=False,bigrams=False,trigrams=False,discard_closed_class=False)
NB.train(corpus.train)
NB.test(corpus.test)
# store predictions from classifier
non_smoothed_preds=NB.predictions

print_st = f"Accuracy without smoothing: {NB.getAccuracy():.2f}"
print(print_st)
results.savePrint(Q_no, print_st)

del print_st, Q_no


results.savePrint_noQ("\n \n")


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

results.savePrint_noQ("\n")



# question 2.1
# see if smoothing significantly improves results

Q_no = "Q 2.1"

p_value=signTest.getSignificance(non_smoothed_preds,smoothed_preds)
significance = "significant" if p_value < 0.05 else "not significant"

print_st = f"-> Results using smoothing are {significance} with respect to no smoothing"
print(print_st)
results.savePrint(Q_no, print_st)

del print_st, Q_no

results.savePrint_noQ("\n \n")


# breakpoint()

# question 3.0

results.savePrint_noQ(f"--------- ** NB: Cross-Validation ** ----------------")


Q_no = "Q 3.0"
Q_id = "Q3_NB_CV"


print("--- classifying reviews using 10-fold cross-evaluation ---")
# using previous instantiated object
NB.crossValidate(corpus, Q_no = "Q 3.0", Q_id=Q_id)
# using cross-eval for smoothed predictions from now on
smoothed_preds=NB.predictions 


del Q_no, Q_id

# breakpoint()

results.savePrint_noQ("\n \n")

results.savePrint_noQ(f"-------------- ** NB: Cross-Validation with Stemming ** ----------------")


# question 4.0
Q_no = "Q 4.0"
Q_id = "Q4_NB_CV_withStem"

print("--- stemming corpus ---")
# retrieve corpus with tokenized text and stemming (using porter)
stemmed_corpus=MovieReviewCorpus(stemming=True,pos=False)
print("--- cross-validating NB using stemming ---")
NB.crossValidate(stemmed_corpus, Q_no = "Q 4.0", Q_id = Q_id)
stemmed_preds=NB.predictions


del Q_no, Q_id

results.savePrint_noQ("\n")


# TODO Q4.1
# see if stemming significantly improves results on smoothed NB

Q_no = "Q 4.1"

p_value=signTest.getSignificance(stemmed_preds,smoothed_preds)
significance = "significant" if p_value < 0.05 else "not significant"

print_st = f"-> Results using stemming are {significance} with respect to no stemming <-"
print(print_st)
results.savePrint(Q_no, print_st)

del Q_no

results.savePrint_noQ("\n")
results.savePrint_noQ("\n")



# TODO Q4.2

results.savePrint_noQ("Q 4.2")
results.savePrint_noQ("--------------- ** determining the number of features before/after stemming ** ---------")



num_stemmed_features = len(NB.vocabulary)
print(f'-> Number of features before stemming: {num_non_stemmed_features} \n Number of features after stemming: {num_stemmed_features}')
results.savePrint_noQ(f'-> Number of features before stemming: {num_non_stemmed_features} \n Number of features after stemming: {num_stemmed_features}')

results.savePrint_noQ("\n \n")

# question Q5.0
# cross-validate model using smoothing and bigrams

results.savePrint_noQ(f"--------------- ** NB: Cross-Validation with Smoothing and Bigrams ** ----------------")


Q_no = "Q 5.0 part_a (Smoothing + Bigrams)"
Q_id = "Q5.0_NB_Smooth_Bigram"
print("--- cross-validating naive bayes using Smoothing + Bigrams ---")
NB=NaiveBayesText(smoothing=True,bigrams=True,trigrams=False,discard_closed_class=False)
NB.crossValidate(corpus, "Q 5.0", Q_id=Q_id)
smoothed_and_bigram_preds=NB.predictions


results.savePrint_noQ("\n \n")

del Q_no, Q_id

results.savePrint_noQ("\n")



# see if bigrams significantly improves results on smoothed NB only

Q_no = "Q 5.0 part_b (Smoothing + Bigrams)"
results.savePrint_noQ("\n")
results.savePrint_noQ(Q_no)

p_value=signTest.getSignificance(smoothed_preds,smoothed_and_bigram_preds)
signifance = "significant" if p_value < 0.05 else "not significant"

print_st = f" -> Results using smoothing and bigrams are {signifance} with respect to smoothing only"
results.savePrint_noQ(print_st)

print(print_st)

del print_st, Q_no


# TODO Q5.1

Q_no = "Q 5.1 part_a (Smoothing + Bigram)" 


# How many features does the BoW model have to take into account now?
smoothed_and_bigram_features=len(NB.vocabulary)

results.savePrint_noQ("\n")
results.savePrint_noQ(Q_no)
results.savePrint_noQ(f"-> Number of features Stemmed+Bigram: {smoothed_and_bigram_features}")


# How does this compare to the number of features at Q3?
results.savePrint_noQ(f"-> Number of features from Q3: {num_non_stemmed_features}")

del Q_no



# --------------- smoothing and trigrams ------------- #

results.savePrint_noQ("\n \n")


results.savePrint_noQ(f"--------------- ** NB: Cross-Validation with Smoothing and Trigrams ** ----------------")


Q_no = "Q 5.0 part_a (Smoothing + Trigrams)"
Q_id = "Q5.0_NB_Smooth_Trigram"

print("--- Cross-validating naive bayes using Smoothing + Trigrams ---")
NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=True,discard_closed_class=False)
NB.crossValidate(corpus, "Q 5.0", Q_id=Q_id)
smoothed_and_bigram_preds=NB.predictions


results.savePrint_noQ("\n \n")

del Q_no, Q_id




# see if bigrams significantly improves results on smoothed NB only

Q_no = "Q 5.0 part_b (Smoothing + Trigrams)"
results.savePrint_noQ("\n")
results.savePrint_noQ(Q_no)

p_value=signTest.getSignificance(smoothed_preds,smoothed_and_bigram_preds)
signifance = "significant" if p_value < 0.05 else "not significant"

print_st = f" -> Results using Smoothing + Trigrams are {signifance} with respect to smoothing only"
results.savePrint_noQ(print_st)

print(print_st)

del print_st, Q_no


# TODO Q5.1

Q_no = "Q 5.1 part_a (Smoothing + Trigrams)"


# How many features does the BoW model have to take into account now?
smoothed_and_bigram_features=len(NB.vocabulary)

results.savePrint_noQ("\n")
results.savePrint_noQ(Q_no)
results.savePrint_noQ(f"-> Number of features Smoothing+Trigram: {smoothed_and_bigram_features}")


# How does this compare to the number of features at Q3?
results.savePrint_noQ(f"-> Number of features from Q3: {num_non_stemmed_features}")

del Q_no


results.savePrint_noQ("\n \n")



# TODO Q6 and 6.1
print("-------------- ** Classifying reviews using SVM 10-fold cross-eval [no POS] **-------------")


Q_no = "Q 6.0"
Q_id = "Q6.0_SVM_NoBigram_NoTrigram_NoPos"

print_st = "--------- ** Cross-validating SVM using NoBigram + NoTrigram [no Pos] ** ---------"

results.savePrint_noQ(print_st)
SVM=SVMText(bigrams=False,trigrams=False,discard_closed_class=False)
SVM.crossValidate(corpus, Q_no, Q_id=Q_id)
smoothed_and_bigram_preds=SVM.predictions


results.savePrint_noQ("\n \n")

del Q_no, print_st, Q_id

results.savePrint_noQ("\n")


# TODO Q7
print("----------- ** Adding in POS information to corpus **------------")

corpus_withPos =MovieReviewCorpus(stemming=False,pos=True)


Q_no = "Q 7.0"
Q_id = "Q7.0_SVM_NoBigram_NoTrigram_WithPos"

print_st = "--- Cross-validating SVM using NoBigram + NoTrigram [with POS] ---"
results.savePrint_noQ(print_st)

SVM=SVMText(bigrams=False,trigrams=False,discard_closed_class=False)
SVM.crossValidate(corpus_withPos, Q_no, Q_id)
smoothed_and_bigram_preds=SVM.predictions


results.savePrint_noQ("\n \n")

del Q_no, print_st, Q_id



print("----------- ** Training svm on word+pos features **-------------") # I'm confused isn't this just the same?
print("----------- ** Training svm discarding closed-class words **------------")

Q_no = "Q 7.1"
Q_id = "Q7.1_SVM_NoBigram_NoTrigram_WithPos_DiscardTrue"

print_st = "--- Cross-validating SVM using NoBigram + NoTrigram [with POS][discard closed-class]---"
results.savePrint_noQ(print_st)

SVM=SVMText(bigrams=False,trigrams=False,discard_closed_class=True)
SVM.crossValidate(corpus_withPos, Q_no, Q_id=Q_id)
smoothed_and_bigram_preds=SVM.predictions


results.savePrint_noQ("\n \n")

del Q_no, print_st, Q_id



# question 8.0
print("----------- ** Using document embeddings **------------")

# Train on big data, (imdb) but then test the accuracy on our dataset (the same 10 folds we used for the other models)
# So we can compare performance

