
from sentence_transformers import SentenceTransformer
import numpy as np


class IssueEmbedder:

    def __init__(self, model_name="all-MiniLM-L6-v2"):

        print("Loading Sentence Transformer model...")

        self.model = SentenceTransformer(model_name)

        print("Model loaded successfully!")

    def generate_embeddings(self, texts):

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings


if __name__ == "__main__":

    embedder = IssueEmbedder()

    issues = [
        "Login button is not working",
        "Unable to log in using the login button",
        "Website background colour is incorrect"
    ]

    vectors = embedder.generate_embeddings(issues)

    print("\nEmbedding Shape:", vectors.shape)

    print("\nFirst Issue Vector:")
    print(vectors[0])
