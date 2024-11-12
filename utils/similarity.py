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

def plot_similarity_matrix(sim_matrix, title, cmap='gray'):
    """
    Plots the similarity matrix.
    
    Parameters:
    sim_matrix (np.ndarray): A 2D similarity matrix to plot.
    cmap (str): Colormap for the plot.
    """
    plt.imshow(sim_matrix, cmap=cmap)
    plt.colorbar(label=f"{title}")
    plt.title(f"{title} Matrix")
    plt.xlabel("Embeddings Index")
    plt.ylabel("Embeddings Index")
    plt.show()

def sparse_similarity(sparse1, sparse2):
    total = 0
    for key1, value1 in sparse1.items():
        for key2, value2 in sparse2.items():
            if key1 == key2:
                total += value1 * value2
    return total

def sparse_similarity_matrix(embeddings):
    """
    Computes the cosine similarity matrix for a set of embeddings using sparse vectors.
    
    Parameters:
    embeddings (List[Dict[int, float]]): A list of sparse vectors.
    
    Returns:
    np.ndarray: A 2D array representing the cosine similarity matrix.
    """
    sim_matrix = np.zeros((len(embeddings), len(embeddings)))
    for i, emb1 in enumerate(embeddings):
        for j, emb2 in enumerate(embeddings):
            sim_matrix[i, j] = sparse_similarity(emb1, emb2)
    # normalize the matrix
    for i in range(len(embeddings)):
        sim_matrix[i] = sim_matrix[i] / np.linalg.norm(sim_matrix[i])
    return sim_matrix