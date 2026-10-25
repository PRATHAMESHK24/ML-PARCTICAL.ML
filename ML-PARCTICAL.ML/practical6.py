
import numpy as np
import pandas as pd

# =========================================================
# PART A: CREATE A USER-ITEM RATING MATRIX
# =========================================================

ratings_data = {
    "User": [
        "Alice", "Alice", "Alice", "Alice",
        "Bob", "Bob", "Bob", "Bob",
        "Charlie", "Charlie", "Charlie", "Charlie",
        "David", "David", "David", "David",
        "Emma", "Emma", "Emma", "Emma"
    ],
    "Item": [
        "Titanic", "Avatar", "Inception", "Joker",
        "Titanic", "Avatar", "Avengers", "Joker",
        "Titanic", "Inception", "Joker", "Interstellar",
        "Avatar", "Inception", "Avengers", "Interstellar",
        "Titanic", "Avatar", "Avengers", "Interstellar"
    ],
    "Rating": [
        5, 4, 5, 2,
        4, 5, 4, 2,
        5, 4, 3, 5,
        3, 5, 5, 4,
        4, 4, 5, 5
    ]
}

ratings_df = pd.DataFrame(ratings_data)

# Rows = users, columns = items
rating_matrix = ratings_df.pivot(
    index="User",
    columns="Item",
    values="Rating"
)

print("USER-ITEM RATING MATRIX")
print(rating_matrix)

# =========================================================
# PART B: COSINE SIMILARITY WITH MISSING RATINGS
# =========================================================

def masked_cosine_similarity(A, B):
    """
    Calculate cosine similarity for each pair of rows.
    Only positions rated by both rows are used.
    Missing ratings are represented by NaN.
    """
    n = A.shape[0]
    m = B.shape[0]
    similarities = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            mask = np.isfinite(A[i]) & np.isfinite(B[j])

            if not np.any(mask):
                continue

            a = A[i, mask]
            b = B[j, mask]

            denominator = np.linalg.norm(a) * np.linalg.norm(b)

            if denominator > 0:
                similarities[i, j] = np.dot(a, b) / denominator

    return similarities


# =========================================================
# PART C: USER-BASED COLLABORATIVE FILTERING
# =========================================================

def user_based_recommendations(matrix, target_user, top_n=3):
    users = matrix.index.tolist()
    items = matrix.columns.tolist()

    if target_user not in users:
        raise ValueError("User not found in rating matrix.")

    values = matrix.to_numpy(dtype=float)
    user_idx = users.index(target_user)

    # Similarities between the target user and every user
    similarities = masked_cosine_similarity(
        values[[user_idx]], values
    )[0]

    print(f"\nSimilarities for {target_user}:")
    for user, sim in zip(users, similarities):
        if user != target_user:
            print(f"{user}: {sim:.4f}")

    predictions = {}

    for item_idx, item in enumerate(items):
        # Do not recommend already-rated items
        if np.isfinite(values[user_idx, item_idx]):
            continue

        numerator = 0.0
        denominator = 0.0

        for other_idx in range(len(users)):
            if other_idx == user_idx:
                continue

            rating = values[other_idx, item_idx]
            similarity = similarities[other_idx]

            if np.isfinite(rating) and similarity > 0:
                numerator += similarity * rating
                denominator += similarity

        if denominator > 0:
            predictions[item] = numerator / denominator

    return sorted(
        predictions.items(),
        key=lambda pair: pair[1],
        reverse=True
    )[:top_n]


# =========================================================
# PART D: ITEM-BASED COLLABORATIVE FILTERING
# =========================================================

def item_based_recommendations(matrix, target_user, top_n=3):
    users = matrix.index.tolist()
    items = matrix.columns.tolist()

    if target_user not in users:
        raise ValueError("User not found in rating matrix.")

    values = matrix.to_numpy(dtype=float)
    user_idx = users.index(target_user)

    # Items are rows; users are columns
    item_vectors = values.T

    item_similarities = masked_cosine_similarity(
        item_vectors, item_vectors
    )

    predictions = {}

    for item_idx, item in enumerate(items):
        if np.isfinite(values[user_idx, item_idx]):
            continue

        numerator = 0.0
        denominator = 0.0

        # Use only items the target user has already rated
        for rated_idx in range(len(items)):
            rating = values[user_idx, rated_idx]

            if not np.isfinite(rating):
                continue

            similarity = item_similarities[item_idx, rated_idx]

            if similarity > 0:
                numerator += similarity * rating
                denominator += similarity

        if denominator > 0:
            predictions[item] = numerator / denominator

    return sorted(
        predictions.items(),
        key=lambda pair: pair[1],
        reverse=True
    )[:top_n]


# =========================================================
# PART E: GENERATE TOP-N RECOMMENDATIONS
# =========================================================

target_user = "Alice"
top_n = 3

user_recs = user_based_recommendations(
    rating_matrix, target_user, top_n
)

item_recs = item_based_recommendations(
    rating_matrix, target_user, top_n
)

print(f"\nTOP-{top_n} USER-BASED RECOMMENDATIONS FOR {target_user}")
if user_recs:
    for rank, (item, score) in enumerate(user_recs, start=1):
        print(f"{rank}. {item}: predicted rating = {score:.3f}")
else:
    print("No recommendations available.")

print(f"\nTOP-{top_n} ITEM-BASED RECOMMENDATIONS FOR {target_user}")
if item_recs:
    for rank, (item, score) in enumerate(item_recs, start=1):
        print(f"{rank}. {item}: predicted rating = {score:.3f}")
else:
    print("No recommendations available.")

print("\nExperiment completed successfully.")

