from utils import *
import numpy as np

def apply_pca(images, n_components=100, mode='train'):
    if mode == 'train':
        # Center the training images
        mean_image = np.mean(images, axis=0)
        centered_images = images - mean_image

        # Compute covariance matrix
        covariance_matrix = np.cov(centered_images, rowvar=False)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues by desending order and save the indices
        sorted_indices = np.argsort(eigenvalues)[::-1]
        # Sort eigenvectors by the indices
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select top n_components
        top_eigenvectors = eigenvectors[:, :n_components]

        # Save mean of the training images and principal components
        pca_model = {
            'mean_image': mean_image,
            'components': top_eigenvectors
        }
        save_model(pca_model, 'pca_model.pkl')

    elif mode == 'test':
        # Load mean of the training images and principal components 
        model = load_model('pca_model.pkl')
        mean_image = model['mean_image']
        top_eigenvectors = model['components']
        centered_images = images - mean_image
    
    return np.dot(centered_images, top_eigenvectors)

def image_to_reduced_feature(images, split='test'):
    if split == 'train': #When training
        return apply_pca(images, mode='train')
    else: #When testing 
        return apply_pca(images, mode='test')
 
class KNN:
    def __init__(self, k=5):
        self.k = k
        self.features_train = None
        self.labels_train = None

    def fit(self, features, labels):
        self.features_train = features
        self.labels_train = labels

    def predict(self, features_test):
        predictions = []
        
        for feature in features_test:
            # Compute cosine similarities
            similarities = np.dot(self.features_train, feature) / (
                np.linalg.norm(self.features_train, axis=1) * np.linalg.norm(feature) + 1e-10
            )
            
            # Get top k indices
            k_indices = np.argsort(similarities)[-self.k:]
            # Get top k nearest labels based on the indices
            k_nearest_labels = self.labels_train[k_indices]
            
            # Weighted voting
            weights = similarities[k_indices]
            # Saves the tuple of label and weight of a neighbor in a dictionary
            weighted_votes = {}
            for label, weight in zip(k_nearest_labels, weights):
                weighted_votes[label] = weighted_votes.get(label, 0) + weight
            
            # Make prediction based on the most weight
            prediction = max(weighted_votes.items(), key=lambda x: x[1])[0]
            predictions.append(prediction)
        
        return np.array(predictions)

def training_model(train_features, train_labels, k=5):
    model = KNN(k=k)
    model.fit(train_features, train_labels)
    return model