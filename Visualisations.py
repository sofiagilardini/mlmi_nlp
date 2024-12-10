from sklearn.manifold import TSNE
# from umap import UMAP
import plotly.express as px
import numpy as np
from gensim.models import Doc2Vec
from imdb_analysis import IMDBLoader, Doc2VecTrainer, SVMClassifier
import multiprocessing
from Corpora import MovieReviewCorpus
import matplotlib.pyplot as plt


print(f"CPU available: {multiprocessing.cpu_count()}")
workers = multiprocessing.cpu_count() - 2


# Load labeled IMDB data for Doc2Vec training
imdb_loader = IMDBLoader('data/aclImdb')
imdb_tagged_docs = imdb_loader.load_reviews()

corpus = MovieReviewCorpus(stemming=False, pos=False)


# vector_size = 50
# window = 5
# min_count = 2
# dm = 9
# epochs = 50

# best_doc2vec_model = Doc2Vec(
#         imdb_tagged_docs,
#         vector_size=vector_size,
#         window=window,
#         min_count=min_count,
#         workers=workers,
#         dm=dm,
#         epochs=epochs)


model_path = '/home/sofia/MLMI/mlmi_nlp/Doc2Vec_Models/doc2vec_dm0_vec50_win5_min1_epochs25.model'
best_doc2vec_model = Doc2Vec.load(model_path)

# Dimensionality reduction and plotting functions
def plot_embeddings(embeddings, labels, title, dim=2, method='t-SNE', interactive=False):

    if dim == 2:
        reducer = TSNE(n_components=2) #if method == 't-SNE'
        reduced_embeddings = reducer.fit_transform(embeddings)
        fig = px.scatter(
            x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1],
            color=labels, title=title,
            labels={'x': 'Dim 1', 'y': 'Dim 2'}
        )
    elif dim == 3:
        reducer = TSNE(n_components=3) #if method == 't-SNE'
        reduced_embeddings = reducer.fit_transform(embeddings)
        fig = px.scatter_3d(
            x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], z=reduced_embeddings[:, 2],
            color=labels, title=title,
            labels={'x': 'Dim 1', 'y': 'Dim 2', 'z': 'Dim 3'}
        )

    if interactive:
        fig.show()
    else:
        fig.write_html(f"{title}_{method}_{dim}D.html")
        print(f"Saved interactive plot: {title}_{method}_{dim}D.html")



# Generate embeddings for training data
doc_vectors = [best_doc2vec_model.infer_vector([token for token in review]) for label, review in corpus.train]
labels = ["POS" if label == "POS" else "NEG" for label, _ in corpus.train]


# # 2D t-SNE
# plot_embeddings(doc_vectors, labels, "t-SNE 2D Visualization", dim=2, method='t-SNE', interactive=True)

# # 3D t-SNE
# plot_embeddings(doc_vectors, labels, "t-SNE 3D Visualization", dim=3, method='t-SNE', interactive=True)

# # 2D UMAP
# plot_embeddings(doc_vectors, labels, "UMAP 2D Visualization", dim=2, method='UMAP', interactive=True)

# # 3D UMAP
# plot_embeddings(doc_vectors, labels, "UMAP 3D Visualization", dim=3, method='UMAP', interactive=True)





import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
import seaborn


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# Example documents (5 positive, 5 negative)
# Updated example documents (5 positive, 5 negative)
documents = [
    # Positive documents
    "This movie was absolutely amazing with a great cast and inspiring story. Every scene was beautifully crafted, and the soundtrack complemented the emotional journey perfectly. I couldn’t help but feel deeply connected to the characters and their struggles.",
    "A truly fantastic experience with stellar performances and breathtaking visuals. The dialogue was sharp and witty, and the pacing kept me hooked from start to finish. It’s the kind of film that stays with you long after it ends.",
    "An uplifting and heartwarming tale with remarkable characters and a memorable soundtrack. The chemistry between the leads was outstanding, and the cinematography captured the beauty of the setting in every frame.",
    "The film was a masterpiece with incredible acting and stunning cinematography. The story was both compelling and thought-provoking, tackling difficult themes with grace and sensitivity. A must-watch for anyone who loves great cinema.",
    "A highly entertaining and delightful film that kept me engaged throughout. The humor was spot on, and the emotional moments were heartfelt and genuine. It’s a movie that leaves you feeling good and inspired.",

    # Negative documents
    "The film was a total disaster. Poor acting and terrible pacing ruined any potential it had. The characters were one-dimensional, and the story felt like it was cobbled together without much thought.",
    "A completely boring experience with a weak plot and flat characters. Nothing about this movie worked, and it was painful to sit through the entire runtime.",
    "What a waste of time with no direction and awful dialogue. The movie tried to be something it clearly wasn’t, and the result was an incoherent mess.",
    "The movie was forgettable, slow, and lacked any compelling moments. It felt like a chore to watch, and by the end, I regretted spending my time on it.",
    "Poor writing and mediocre performances made this film hard to sit through. The story lacked originality, and the entire production felt lazy and uninspired."
]


# Critical words found in documents
critical_words = ["amazing", "fantastic", "heartwarming", "disaster", "boring", "waste", "poor", "stunning"]

# Assume `best_doc2vec_model` is your pre-trained Doc2Vec model
# Infer embeddings for documents
doc_embeddings = [best_doc2vec_model.infer_vector(doc.split()) for doc in documents]

# Infer embeddings for critical words
word_embeddings = [best_doc2vec_model.wv[word] for word in critical_words]

# -----------------------------
# DOCUMENT-TO-DOCUMENT SIMILARITY
# -----------------------------
# Compute cosine similarity matrix for documents
doc_sim_matrix = cosine_similarity(doc_embeddings)

# Save document-to-document similarity to CSV
with open("document_similarity_matrix.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([""] + [f"Doc {i+1}" for i in range(len(documents))])  # Header row
    for i, row in enumerate(doc_sim_matrix):
        writer.writerow([f"Doc {i+1}"] + list(row))
print("Document-to-Document Similarity Matrix saved to 'document_similarity_matrix.csv'.")

# Heatmap for document-to-document similarity
sns.heatmap(
    doc_sim_matrix, annot=True, cmap="coolwarm",
    xticklabels=[f"Doc {i+1}" for i in range(len(documents))],
    yticklabels=[f"Doc {i+1}" for i in range(len(documents))]
)
plt.title("Document-to-Document Cosine Similarity")
plt.xlabel("Document")
plt.ylabel("Document")
plt.savefig("document_similarity_heatmap.png")
plt.show()

# -----------------------------
# DOCUMENT-TO-CRITICAL-WORD SIMILARITY
# -----------------------------
# Compute cosine similarity between each document and all critical words
doc_word_sim = {}
for i, doc in enumerate(documents):
    doc_word_sim[f"Doc {i+1}"] = [
        (word, cosine_similarity([doc_embeddings[i]], [best_doc2vec_model.wv[word]])[0][0])
        for word in critical_words
    ]

# Save document-to-critical-word similarity to CSV
with open("document_to_word_similarity.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Document", "Word", "Similarity"])
    for doc, word_sims in doc_word_sim.items():
        for word, sim in word_sims:
            writer.writerow([doc, word, sim])
print("Document-to-Critical-Word Similarity saved to 'document_to_word_similarity.csv'.")

# Create a similarity matrix for heatmap visualization
doc_word_matrix = np.array([
    [cosine_similarity([doc_embeddings[i]], [best_doc2vec_model.wv[word]])[0][0] for word in critical_words]
    for i in range(len(documents))
])

# Heatmap for document-to-critical-word similarity
sns.heatmap(
    doc_word_matrix, annot=True, cmap="coolwarm",
    xticklabels=critical_words,
    yticklabels=[f"Doc {i+1}" for i in range(len(documents))]
)
plt.title("Document-to-Critical-Word Cosine Similarity")
plt.xlabel("Critical Words")
plt.ylabel("Documents")
plt.savefig("document_to_word_similarity_heatmap.png")
plt.show()
