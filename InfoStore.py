import matplotlib.pyplot as plt
import seaborn as sns
from Lexicon import SentimentLexicon
from Corpora import MovieReviewCorpus
import numpy as np

params = {
    "font.family": "serif",
}


import matplotlib as mpl
mpl.rcParams.update(params)

class figurePlotting():

    def plotHeatmap(self, threshold):

        corpus=MovieReviewCorpus(stemming=False,pos=False)


        lexicon = SentimentLexicon()

        strong_weight_list = [1, 2, 3, 4, 5, 7]
        weak_weight_list = [1, 0.75, 0.5, 0.25, 0.1, 0.05]

        dim1 = len(strong_weight_list)
        dim2 = len(weak_weight_list)

        accuracy_array = np.zeros((dim1, dim2))
        print(accuracy_array.shape)

        for i_s, strong in enumerate(strong_weight_list):
            for i_w, weak in enumerate(weak_weight_list):

                lexicon.classify2(corpus.reviews, threshold, magnitude=True, strong_weight=strong, weak_weight=weak)
                token_preds=lexicon.predictions
                accuracy = lexicon.getAccuracy()
                print(f"token-only results: {lexicon.getAccuracy():.2f}")

                accuracy_array[i_s, i_w] = accuracy

        sns.heatmap(accuracy_array, annot = True)

        plot_dir = './Figures'

        plt.savefig(f'{plot_dir}/Q0_heatmap.png')