
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityCalculator:

    @staticmethod
    def calculate_similarity(embeddings):

        similarity_matrix = cosine_similarity(embeddings)

        return similarity_matrix

    @staticmethod
    def calculate_query_similarity(query_embedding, corpus_embeddings):
        """Return cosine scores between one query vector and corpus vectors."""
        query = np.asarray(query_embedding).reshape(1, -1)
        corpus = np.asarray(corpus_embeddings)
        return cosine_similarity(query, corpus).ravel()


if __name__ == "__main__":

    from .embeddings import IssueEmbedder

    embedder = IssueEmbedder()

    issues = [
        "Login button is not working",
        "Unable to log in using the login button",
        "Website background colour is incorrect"
    ]

    vectors = embedder.generate_embeddings(issues)

    calculator = SimilarityCalculator()

    similarity_matrix = calculator.calculate_similarity(vectors)

    print("\nCOSINE SIMILARITY MATRIX\n")

    print(similarity_matrix)

    print("\nISSUE COMPARISONS\n")

    for i in range(len(issues)):
        for j in range(i + 1, len(issues)):

            score = similarity_matrix[i][j]

            print(f"Issue {i+1} vs Issue {j+1}: {score:.4f}")
