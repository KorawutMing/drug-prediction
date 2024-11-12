from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

def cosine_similarity_matrix(embeddings):
    """
    Computes the cosine similarity matrix for a set of embeddings.
    
    Parameters:
    embeddings (np.ndarray): A 2D array where each row is an embedding vector.
    
    Returns:
    np.ndarray: A 2D array representing the cosine similarity matrix.
    """
    if embeddings.ndim != 2:
        raise ValueError("Input embeddings must be a 2D array.")
    return cosine_similarity(embeddings)

def plot_similarity_matrix(sim_matrix, cmap='gray'):
    """
    Plots the similarity matrix.
    
    Parameters:
    sim_matrix (np.ndarray): A 2D similarity matrix to plot.
    cmap (str): Colormap for the plot.
    """
    plt.imshow(sim_matrix, cmap=cmap)
    plt.colorbar(label="Cosine Similarity")
    plt.title("Cosine Similarity Matrix")
    plt.xlabel("Embeddings Index")
    plt.ylabel("Embeddings Index")
    plt.show()