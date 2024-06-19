import torch
from sentence_transformers import SentenceTransformer, util

# Check if MPS is available
if not torch.backends.mps.is_available():
    raise ValueError("MPS device not available.")
device = torch.device("mps")

# Load the model and move it to the MPS device
model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Example user input and vault embeddings
user_input = "example input"
vault_embeddings = torch.rand(10, 384).to(device)  # Example tensor shape

# Encode user input
input_embedding = model.encode([user_input])
input_embedding = torch.tensor(input_embedding).to(device)

# Move the encoded input to MPS device
input_embedding = input_embedding.to(device)

# Test cosine similarity using sentence-transformers util
try:
    cos_scores = util.cos_sim(input_embedding, vault_embeddings)[0]
    print("Cosine similarity calculation succeeded on MPS.")
    print(cos_scores)
except Exception as e:
    print("Cosine similarity calculation failed on MPS.")
    print(e)
