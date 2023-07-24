import math
from sklearn.metrics import ndcg_score

def calculate_dcg(relevance_scores, k):
    dcg = 0.0
    for i in range(min(k, len(relevance_scores))):
        rel_i = relevance_scores[i]
        dcg += rel_i / math.log2((i+1) + 1)
    return dcg


def calculate_idcg(relevance_scores, k):
    sorted_relevance_scores = sorted(relevance_scores, reverse=True)
    return calculate_dcg(sorted_relevance_scores, k)


def calculate_ndcg(ground_truth, ranked_list, k):
    dcg = calculate_dcg(ranked_list, k)
    print(f"DCG: {dcg}")
    idcg = calculate_idcg(ground_truth, k)
    print(f"IDCG: {idcg}")

    if idcg == 0.0:
        return 0.0
    else:
        ndcg = dcg / idcg
        return ndcg


# Example usage:
ground_truth = [3, 2, 4, 5, 1]
ranked_list = [3, 1, 2, 4, 5, 6]
k = 6

ground_truth_len = len(ground_truth)
predicted_len = len(ranked_list)

if predicted_len != ground_truth_len:
    if predicted_len < ground_truth_len:
        # truncate the ground truth
        ground_truth = ground_truth[:predicted_len]
    else:
        # truncate the ranked list
        ranked_list = ranked_list[:ground_truth_len]


ndcg_at_k = calculate_ndcg(ground_truth, ranked_list, k)
print(f"nDCG@{k} is: {ndcg_at_k:.4f}")


print(ndcg_score([ground_truth], [ranked_list], k=6))
