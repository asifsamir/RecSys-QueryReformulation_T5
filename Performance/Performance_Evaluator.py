from Performance.Metrics.Evaluation_Metrics_2 import AverageHit_At_K, \
    AverageRecall_At_K, MRR, MAP


class Performance_Evaluator:

    def __init__(self):
        self.map_metric = MAP()
        self.mrr_metric = MRR()
        self.recall_at_k = AverageRecall_At_K()
        self.hit_at_k = AverageHit_At_K()


    def evaluate(self, ground_truths, search_results, K):
        map_score = self.map_metric.calculate(ground_truths, search_results)
        mrr_score = self.mrr_metric.calculate(ground_truths, search_results)
        recall_score = self.recall_at_k.calculate(ground_truths, search_results, K)
        hit_score = self.hit_at_k.calculate(ground_truths, search_results, K)


        # return them as a dictionary
        return {
            "map": map_score,
            "mrr": mrr_score,
            f"recall@{K}": recall_score,
            f"hit@{K}": hit_score
        }



        # print(f"MAP: {map_score:.4f}")
        # print(f"MRR: {mrr_score:.4f}")
        # print(f"Recall@{K}: {recall_score:.4f}")
        # print(f"Hit@{K}: {hit_score:.4f}")
        # print(f"Precision@{K}: {precision_score:.4f}")
        # # print(f"NDCG@{K}: {np.mean(ndcg_score):.4f}")