import math,sys
import numpy as np

class Evaluation():
    """
    general evaluation class implemented by classifiers
    """
    def crossValidate(self,corpus, Q_no):

        from InfoStore import figurePlotting, resultsWrite

        """
        function to perform 10-fold cross-validation for a classifier.
        each classifier will be inheriting from the evaluation class so you will have access
        to the classifier's train and test functions.

        1. read reviews from corpus.folds and store 9 folds in train_files and 1 in test_files
        2. pass data to self.train and self.test e.g., self.train(train_files)
        3. repeat for another 9 runs making sure to test on a different fold each time

        @param corpus: corpus of movie reviews
        @type corpus: MovieReviewCorpus object
        """
        # reset predictions
        self.predictions=[]

        print("HERE")

        splits = {}


        # for list_review_fold in corpus.folds:
        #     for i in range(len(list_review_fold)):
        #         print("here")
        #         splits[i] = list_review_fold[i]


        results = resultsWrite()


        index_list = np.arange(len(corpus.folds))

        CV_dict = {}

        results_list = []

        for counter, index in enumerate(index_list):
            test_index = index
            train_files = []
            test_files = []
            
            training_index_list = np.delete(index_list, index)
            print(f'training: {training_index_list}, test: {test_index}')

            # train_files = corpus.folds[training_index_list]
            # test_files = corpus.folds[test_index]

            for fold_indx in training_index_list:
                train_files.extend(corpus.folds[fold_indx])

            test_files = corpus.folds[test_index]

            self.train(train_files)
            self.test(test_files)


            print_st = f"Test fold: {test_index}; \n Accuracy: {self.getAccuracy():3f} \n Std. Dev: {self.getStdDeviation()}"
            print(print_st)

            if counter == 0:
                results.savePrint_noQ(Q_no)
                results.savePrint_noQ(print_st)
            else:
                results.savePrint_noQ(print_st)
            
            results_list.append(self.getAccuracy())

        avg_cv_acc = np.mean(results_list)
        std_cv_acc = np.std(results_list)


        results.savePrint_noQ('----------------------------------')
        results.savePrint_noQ("Average of performances across fold:")
        results.savePrint_noQ(f"Mean performance: {avg_cv_acc}")
        results.savePrint_noQ(f"Std between fold performances: {std_cv_acc}")
        # results.savePrint_noQ(f"Variance between fold performances: {var_cv_acc}")

        print("Avg Acc: ", avg_cv_acc)
        print("Var: ", std_cv_acc)


        # TODO Q3

    def getStdDeviation(self):
        """
        get standard deviation across folds in cross-validation.
        """
        # get the avg accuracy and initialize square deviations
        avgAccuracy,square_deviations=self.getAccuracy(),0
        # find the number of instances in each fold
        fold_size=len(self.predictions)//10
        # calculate the sum of the square deviations from mean
        for fold in range(0,len(self.predictions),fold_size):
            square_deviations+=(self.predictions[fold:fold+fold_size].count("+")/float(fold_size) - avgAccuracy)**2
        # std deviation is the square root of the variance (mean of square deviations)
        return math.sqrt(square_deviations/10.0)

    def getAccuracy(self):
        """
        get accuracy of classifier.

        @return: float containing percentage correct
        """
        # note: data set is balanced so just taking number of correctly classified over total
        # "+" = correctly classified and "-" = error
        return self.predictions.count("+")/float(len(self.predictions))
