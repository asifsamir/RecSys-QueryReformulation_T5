import math

import numpy as np


class PerformanceEvaluate:
    def calculate_average_precision(self, gt, results):
        hits = 0
        sum_precision = 0.0

        for i, doc_id in enumerate(results, 1):
            if doc_id in gt:
                hits += 1
                precision = hits / i
                sum_precision += precision

        if hits == 0:
            return 0.0

        return sum_precision / hits

    def MAP(self, ground_truths, search_results):
        total_average_precision = 0.0

        for gt, results in zip(ground_truths, search_results):
            average_precision = self.calculate_average_precision(gt, results)
            total_average_precision += average_precision

        return total_average_precision / len(ground_truths)

    def calculate_reciprocal_rank(self, gt, results):
        for i, doc_id in enumerate(results, 1):
            if doc_id in gt:
                return 1.0 / i
        return 0.0

    def MRR(self, ground_truths, search_results):
        total_reciprocal_rank = 0.0

        for gt, results in zip(ground_truths, search_results):
            reciprocal_rank = self.calculate_reciprocal_rank(gt, results)
            total_reciprocal_rank += reciprocal_rank

        return total_reciprocal_rank / len(ground_truths)

    def calculate_recall_at_k(self, gt, results, k):
        relevant_items = len(gt)
        relevant_found = sum(1 for doc_id in results[:k] if doc_id in gt)
        return relevant_found / relevant_items

    def avg_Recall_at_K(self, ground_truths, search_results, k):
        total_recall_at_k = 0.0

        for gt, results in zip(ground_truths, search_results):
            recall_at_k = self.calculate_recall_at_k(gt, results, k)
            total_recall_at_k += recall_at_k

        return total_recall_at_k / len(ground_truths)

    def calculate_hit_at_k(self, gt, results, k):
        for doc_id in results[:k]:
            if doc_id in gt:
                return 1
        return 0

    def avg_HIT_at_K(self, ground_truths, search_results, k):
        total_hit_at_k = 0.0

        for gt, results in zip(ground_truths, search_results):
            hit_at_k = self.calculate_hit_at_k(gt, results, k)
            total_hit_at_k += hit_at_k

        return total_hit_at_k / len(ground_truths)

    def calculate_precision_at_k(self, gt, results, k):
        relevant_found = sum(1 for doc_id in results[:k] if doc_id in gt)
        return relevant_found / k

    def avg_Precision_at_K(self, ground_truths, search_results, k):
        total_precision_at_k = 0.0

        for gt, results in zip(ground_truths, search_results):
            precision_at_k = self.calculate_precision_at_k(gt, results, k)
            total_precision_at_k += precision_at_k

        return total_precision_at_k / len(ground_truths)

    def calculate_dcg(self, relevance_scores, k):
        dcg = 0.0
        for i in range(min(k, len(relevance_scores))):
            dcg += (2 ** relevance_scores[i] - 1) / np.log2(i + 2)
        return dcg

    def avg_NDCG_at_K(self, ground_truth, search_results, k):
        ndcg_values = []
        for gt, sr in zip(ground_truth, search_results):
            valid_sr = [item for item in sr if item in gt]
            try:
                idcg = self.calculate_dcg(sorted(gt, reverse=True), k)
                dcg = self.calculate_dcg([gt[item - 1] for item in valid_sr], k)
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_values.append(ndcg)
            except IndexError:
                print(f"Invalid index in search results: {sr}")
        return ndcg_values


# Example usage
ground_truths = [
    [1, 3, 5],
    [4],
    [2, 3, 5],
    [1, 4, 5],
    [1, 2, 3, 4, 5],
]

search_results =[
    [2, 1, 5, 3, 4],
    [1, 2, 3, 4, 5],
    [1, 3, 2, 4, 5],
    [2, 1, 4, 5, 3],
    [3, 1, 2, 4, 5],
]


K = 5
evaluator = PerformanceEvaluate()
map_score = evaluator.MAP(ground_truths, search_results)
mrr_score = evaluator.MRR(ground_truths, search_results)
recall_at_k = evaluator.avg_Recall_at_K(ground_truths, search_results, K)
hit_at_k = evaluator.avg_HIT_at_K(ground_truths, search_results, K)
precision_at_k = evaluator.avg_Precision_at_K(ground_truths, search_results, K)
# ndcg_at_k = evaluator.avg_NDCG_at_K(ground_truths, search_results, K)

print(f"MAP: {map_score:.4f}")
print(f"MRR: {mrr_score:.4f}")
print(f"Recall@{K}: {recall_at_k:.4f}")
print(f"Hit@{K}: {hit_at_k:.4f}")
print(f"Precision@{K}: {precision_at_k:.4f}")
# print(f"NDCG@{K}: {ndcg_at_k:.4f}")
