# MNIST Digit Classification with PCA and KNN

A handwritten-digit classifier for a modified MNIST dataset, built with PCA for
feature extraction and a K-Nearest-Neighbours classifier, both implemented by
hand using only numpy and the Python standard library (no scikit-learn). The
system is trained on clean 28x28 digit images and evaluated on two harder test
sets: images with added Gaussian noise, and images with 15x15 masked regions
simulating occlusion.

## Results

| Test set    | Accuracy |
|-------------|----------|
| Noisy test  | 93.3%    |
| Masked test | 72.5%    |

## Approach

Feature extraction (PCA). PCA is implemented from scratch in apply_pca. During
training it centres the images by subtracting the mean, builds the covariance
matrix, and computes its eigenvalues and eigenvectors. The eigenvectors are
sorted by descending eigenvalue (variance), and the top 100 are kept as the
principal components. The training mean and components are stored in the model.
At test time the same mean is used to centre the test images, which are then
projected onto the components to give a 100-dimensional representation. PCA was
chosen because it is unsupervised and linear: it needs no labels and preserves
the spread of the data, which keeps the feature stage cheap compared to a
supervised method like LDA.

Classifier (KNN). KNN is implemented from scratch. Training just stores the
reduced training features and labels. Prediction computes the cosine similarity
between a test point and every training point, takes the five nearest, and
predicts the label by weighted voting, using each neighbour's cosine similarity
as its weight so closer neighbours count for more. KNN pairs well with PCA here:
it has no training phase, and the dimensionality reduction keeps the
distance computations manageable.

## Design decisions

The similarity metric and voting scheme were chosen by comparing four
combinations on both test sets:

| Metric    | Voting   | Noisy | Masked |
|-----------|----------|-------|--------|
| Euclidean | majority | 94.1% | 70.3%  |
| Euclidean | weighted | 93.9% | 71.0%  |
| Cosine    | majority | 92.9% | 72.1%  |
| Cosine    | weighted | 93.3% | 72.5%  |

Euclidean distance did slightly better on noisy images, but cosine similarity
was clearly more robust on masked images, and weighted voting beat majority
voting on both metrics. Cosine similarity with weighted voting was chosen for
the final system because it gives the best balance across the two conditions
rather than winning one at the expense of the other. The number of principal
components (100) and the value of k (5) were both selected by iterative
experimentation for the best balanced accuracy across the two test sets.

## Files

- system.py - my implementation: the PCA feature extraction and the KNN
  classifier.

The training and evaluation harness (train.py, evaluate.py), the utility
functions (utils.py), and the dataset were provided as course scaffolding and
are not included here. This repo contains only the code I wrote, which plugs
into that harness.

## Tech

Python, numpy, and the standard library.
