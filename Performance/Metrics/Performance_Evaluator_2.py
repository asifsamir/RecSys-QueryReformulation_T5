import math
import numpy as np


class AveragePrecision:
    def calculate(self, gt, results):
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


class MAP:
    def __init__(self):
        self.avg_precision = AveragePrecision()

    def calculate(self, ground_truths, search_results):
        total_average_precision = 0.0

        for gt, results in zip(ground_truths, search_results):
            average_precision = self.avg_precision.calculate(gt, results)
            total_average_precision += average_precision

        return total_average_precision / len(ground_truths)


class ReciprocalRank:
    def calculate(self, gt, results):
        for i, doc_id in enumerate(results, 1):
            if doc_id in gt:
                return 1.0 / i
        return 0.0


class MRR:
    def __init__(self):
        self.reciprocal_rank = ReciprocalRank()

    def calculate(self, ground_truths, search_results):
        total_reciprocal_rank = 0.0

        for gt, results in zip(ground_truths, search_results):
            reciprocal_rank = self.reciprocal_rank.calculate(gt, results)
            total_reciprocal_rank += reciprocal_rank

        return total_reciprocal_rank / len(ground_truths)


class RecallAtK:
    def calculate(self, gt, results, k):
        relevant_items = len(gt)
        relevant_found = sum(1 for doc_id in results[:k] if doc_id in gt)
        return relevant_found / relevant_items


class AverageRecall_At_K:
    def __init__(self):
        self.recall_at_k = RecallAtK()

    def calculate(self, ground_truths, search_results, k):
        total_recall_at_k = 0.0

        for gt, results in zip(ground_truths, search_results):
            recall_at_k = self.recall_at_k.calculate(gt, results, k)
            total_recall_at_k += recall_at_k

        return total_recall_at_k / len(ground_truths)


class HitAtK:
    def calculate(self, gt, results, k):
        for doc_id in results[:k]:
            if doc_id in gt:
                return 1
        return 0


class AverageHit_At_K:
    def __init__(self):
        self.hit_at_k = HitAtK()

    def calculate(self, ground_truths, search_results, k):
        total_hit_at_k = 0.0

        for gt, results in zip(ground_truths, search_results):
            hit_at_k = self.hit_at_k.calculate(gt, results, k)
            total_hit_at_k += hit_at_k

        return total_hit_at_k / len(ground_truths)


class PrecisionAtK:
    def calculate(self, gt, results, k):
        relevant_found = sum(1 for doc_id in results[:k] if doc_id in gt)
        return relevant_found / k


class AveragePrecision_At_K:
    def __init__(self):
        self.precision_at_k = PrecisionAtK()

    def calculate(self, ground_truths, search_results, k):
        total_precision_at_k = 0.0

        for gt, results in zip(ground_truths, search_results):
            precision_at_k = self.precision_at_k.calculate(gt, results, k)
            total_precision_at_k += precision_at_k

        return total_precision_at_k / len(ground_truths)


class DiscountedCumulativeGain:
    def calculate(self, relevance_scores, k):
        dcg = 0.0
        for i in range(min(k, len(relevance_scores))):
            dcg += (2 ** relevance_scores[i] - 1) / np.log2(i + 2)
        return dcg


class AverageNDCG_At_K:
    def __init__(self, k):
        self.dcg = DiscountedCumulativeGain()
        self.k = k

    def calculate(self, ground_truth, search_results):
        ndcg_values = []
        for gt, sr in zip(ground_truth, search_results):
            valid_sr = [item for item in sr if item in gt]
            try:
                idcg = self.dcg.calculate(sorted(gt, reverse=True), self.k)
                dcg = self.dcg.calculate([gt[item - 1] for item in valid_sr], self.k)
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_values.append(ndcg)
            except IndexError:
                print(f"Invalid index in search results: {sr}")
        return ndcg_values


# Example usage
ground_truths = [
    ["doc1", "doc3", "doc5"],
    ["doc4"],
    ["doc2", "doc3", "doc5"],
    ["doc1", "doc4", "doc5"],
    ["doc1", "doc2", "doc3", "doc4", "doc5"],
]

search_results = [
    ["doc2", "doc1", "doc5", "doc3", "doc4"],
    ["doc1", "doc2", "doc3", "doc4", "doc5"],
    ["doc1", "doc3", "doc2", "doc4", "doc5"],
    ["doc2", "doc1", "doc4", "doc5", "doc3"],
    ["doc3", "doc1", "doc2", "doc4", "doc5"],
]

K = 5

map_metric = MAP()
mrr_metric = MRR()
recall_at_k = AverageRecall_At_K()
hit_at_k = AverageHit_At_K()
precision_at_k = AveragePrecision_At_K()
ndcg_metric = AverageNDCG_At_K(K)

map_score = map_metric.calculate(ground_truths, search_results)
mrr_score = mrr_metric.calculate(ground_truths, search_results)
recall_score = recall_at_k.calculate(ground_truths, search_results, K)
hit_score = hit_at_k.calculate(ground_truths, search_results, K)
precision_score = precision_at_k.calculate(ground_truths, search_results, K)
# ndcg_score = ndcg_metric.calculate(ground_truths, search_results)

print(f"MAP: {map_score:.4f}")
print(f"MRR: {mrr_score:.4f}")
print(f"Recall@{K}: {recall_score:.4f}")
print(f"Hit@{K}: {hit_score:.4f}")
print(f"Precision@{K}: {precision_score:.4f}")
# print(f"NDCG@{K}: {np.mean(ndcg_score):.4f}")
