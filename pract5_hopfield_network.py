import numpy as np

# --- Helper function to print a flat array as a 5x8 matrix ---
def print_matrix(vector, title):
    print(f"{title}:")
    # Reshape the 40-item flat array into a 5 rows by 8 columns grid
    grid = vector.reshape((5, 8))
    for row in grid:
        # Print +1 as '#' (filled) and -1 as '.' (empty)
        print("  " + " ".join(['#' if val == 1 else '.' for val in row]))
    print()

# 1. Setup
N = 40  # Number of neurons (size of each vector)
num_patterns = 4

# Setting a seed so you get the exact same results every time you run it
np.random.seed(42)

# Generate 4 random bipolar vectors (arrays filled with -1s and 1s)
# These are the "memories" we want to store.
patterns = np.random.choice([-1, 1], size=(num_patterns, N))

# 2. Training / Storing Memories (Hebbian Learning)
# Initialize an N x N weight matrix with zeros
W = np.zeros((N, N))

# Add the outer product of each pattern to the weight matrix
for p in patterns:
    W += np.outer(p, p)

# Hopfield networks cannot have self-connections, so set the diagonal to 0
np.fill_diagonal(W, 0)
W /= N # Normalize the weights

# 3. Create a Test Case (Corrupt a memory)
original_pattern = patterns[0].copy()
test_pattern = original_pattern.copy()

# Introduce "noise" by flipping 8 random bits (from 1 to -1, or vice versa)
flip_indices = np.random.choice(N, 8, replace=False)
test_pattern[flip_indices] *= -1

# 4. Retrieval (Recalling the memory)
retrieved_pattern = test_pattern.copy()

# Update the network until it settles (synchronous update)
# Usually takes only a few iterations to converge
for step in range(5):
    # The core update rule: State = sign(Weights matrix * Current State)
    retrieved_pattern = np.sign(W @ retrieved_pattern)
    
    # np.sign returns 0 if the value is exactly 0. We map those back to 1.
    retrieved_pattern[retrieved_pattern == 0] = 1

# 5. The Results (Now beautifully formatted as matrices!)
print_matrix(original_pattern, "Original Pattern")
print_matrix(test_pattern, "Corrupted Pattern (Noise added)")
print_matrix(retrieved_pattern, "Retrieved Pattern (Memory recalled)")

# Check if the network successfully cleaned up the noise
success = np.array_equal(original_pattern, retrieved_pattern)
print(f"Successfully recovered original pattern? {success}")