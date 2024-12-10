import math,sys
import numpy as np
import csv
import re
import os

class Evaluation():
    """
    general evaluation class implemented by classifiers
    """
    def crossValidate(self,corpus, Q_no, Q_id):

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
        # # reset predictions
        self.predictions=[]


        if "8" not in Q_no:
            results = resultsWrite()
        else:
            results = resultsWrite("IMDB_Results.txt")


        index_list = np.arange(len(corpus.folds))


        if not os.path.exists('./CV_results'):
            os.makedirs("./CV_results")

        csv_file = f'CV_results/{Q_id}_results.csv'


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

            start_index = len(test_files)*test_index


            fold_pred = self.predictions[start_index:]


            if len(fold_pred) > 0:
                fold_acc = fold_pred.count("+") / len(fold_pred)


            print_st = f"Test fold: {test_index}; \n Accuracy: {fold_acc}"


            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([test_index, fold_acc])


            if counter == 0:
                results.savePrint_noQ(Q_no)
                results.savePrint_noQ(print_st)
            else:
                results.savePrint_noQ(print_st)

            del fold_pred, fold_acc
            

        # get Acc and Std across all folds:

        all_folds_acc = f"{self.getAccuracy():3f}"
        all_folds_std = f"{self.getStdDeviation():3f}"

        results.savePrint_noQ("\nAcross all folds:")
        results.savePrint_noQ(f"Acc: {all_folds_acc}")
        results.savePrint_noQ(f"Std: {all_folds_std}")


        # TODO Q3



    def crossValidate_Doc2Vec(self, corpus, modelID):
        """
        Perform 10-fold cross-validation for a Doc2Vec model, export results to both
        a text file and a CSV file.

        @param corpus: Corpus of movie reviews
        @type corpus: MovieReviewCorpus object
        @param modelID: Unique identifier for the model (based on hyperparameters)
        @type modelID: string
        """
        from InfoStore import figurePlotting, resultsWrite

        # Reset predictions
        self.predictions = []

        # Initialize results file for text output
        results = resultsWrite("IMDB_Results.txt")

        # Initialize CSV file for results
        csv_file = f"IMDB_CV_results.csv"
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            # Write the header if not already written
            if not os.path.exists(csv_file):
                writer.writerow(["ModelID", "TestFold", "Accuracy", "DM", "VectorSize", "Window", "MinCount", "Epochs"])

        # Create index list for folds
        index_list = np.arange(len(corpus.folds))
        CV_dict = {}
        results_list = []

        # Extract hyperparameters from modelID
        params = self.parse_model_id(modelID)
        dm = params["dm"]
        vector_size = params["vector_size"]
        window = params["window"]
        min_count = params["min_count"]
        epochs = params["epochs"]


        for counter, index in enumerate(index_list):
            test_index = index
            train_files = []
            test_files = []

            # Create train/test splits
            training_index_list = np.delete(index_list, index)
            print(f'training: {training_index_list}, test: {test_index}')
            
            for fold_indx in training_index_list:
                train_files.extend(corpus.folds[fold_indx])
            test_files = corpus.folds[test_index]

            # Train and test
            self.train(train_files)
            self.test(test_files)


            start_index = len(test_files)*test_index


            fold_pred = self.predictions[start_index:]

            print(f"Length self.predictions: {len(self.predictions)}")
            print(f"Start index: {start_index}")
            print(f"Length fold_pred: {len(fold_pred)}")


            if len(fold_pred) > 0:
                fold_acc = fold_pred.count("+") / len(fold_pred)

            print_st = f"Test fold: {test_index}; \n Accuracy: {fold_acc}"


            print(print_st)

            # Write to text file
            results.savePrint_noQ(print_st)

            # Write to CSV file
            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([modelID, test_index, fold_acc, dm, vector_size, window, min_count, epochs])


        # Calculate overall statistics
        all_folds_acc = f"{self.getAccuracy():3f}"
        all_folds_std = f"{self.getStdDeviation():3f}"


        # Log overall statistics to text file
        results.savePrint_noQ('----------------------------------')
        results.savePrint_noQ("Average of performances across folds:")
        results.savePrint_noQ(f"Average: {all_folds_acc}")
        results.savePrint_noQ(f"Std: {all_folds_std}")
        results.savePrint_noQ("\n")
        results.savePrint_noQ("\n")

        # Log overall statistics to CSV file

        csv_file_summary = "IMDB_GridSearch_Summary.CSV"

        with open(csv_file_summary, mode="a", newline="") as file:
            writer = csv.writer(file)
            print("DM", dm)
            writer.writerow([modelID, all_folds_acc, all_folds_std, dm, vector_size, window, min_count, epochs])

        print(f"Results exported to IMDB_Results.txt and {csv_file}.")


    def crossValidate_nb_svm(self,corpus, test_ID, results_path):

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
        # # reset predictions
        self.predictions=[]


        
        results = resultsWrite(results_path)

        index_list = np.arange(len(corpus.folds))


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

            start_index = len(test_files)*test_index


            fold_pred = self.predictions[start_index:]


            if len(fold_pred) > 0:
                fold_acc = fold_pred.count("+") / len(fold_pred)


            print_st = f"Test fold: {test_index}; \n Accuracy: {fold_acc}"

            if counter == 0:
                results.savePrint_noQ('\n')
                results.savePrint_noQ(f"-----------------{test_ID}----------------------")
                results.savePrint_noQ(print_st)
            else:
                results.savePrint_noQ(print_st)

            del fold_pred, fold_acc
            

        # get Acc and Std across all folds:

        all_folds_acc = f"{self.getAccuracy():3f}"
        all_folds_std = f"{self.getStdDeviation():3f}"

        results.savePrint_noQ("\nAcross all folds:")
        results.savePrint_noQ(f"Acc: {all_folds_acc}")
        results.savePrint_noQ(f"Std: {all_folds_std}")
        results.savePrint_noQ('\n')


    


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



    def parse_model_id(self, modelID):
        """
        Parse the modelID string to extract hyperparameters.

        @param modelID: String in the format "doc2vec_dm{dm}_vec{vector_size}_win{window}_min{min_count}_epochs{epochs}"
        @return: Dictionary of hyperparameters
        """
        match = re.match(
            r"doc2vec_dm(?P<dm>\d+)_vec(?P<vector_size>\d+)_win(?P<window>\d+)_min(?P<min_count>\d+)_epochs(?P<epochs>\d+)",
            modelID,
        )
        if not match:
            raise ValueError(f"Invalid modelID format: {modelID}")
        
        # Extract hyperparameters as integers
        return {
            "dm": int(match.group("dm")),
            "vector_size": int(match.group("vector_size")),
            "window": int(match.group("window")),
            "min_count": int(match.group("min_count")),
            "epochs": int(match.group("epochs")),
        }
