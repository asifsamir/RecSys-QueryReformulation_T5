class MajorityVotingScorer:
    def __init__(self, List_Collection):
        self.lists = List_Collection

    def score_items(self, do_sort=True, only_keys_array=False):
        scores = {}
        for item in self.get_unique_items():
            count = sum(1 for lst in self.lists if item in lst)
            scores[item] = count

        if do_sort:
            # sort the dictionary by value in descending order
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            scores = dict(sorted_scores)

        if only_keys_array:
            scores = list(scores.keys())

        return scores

    def get_unique_items(self):
        unique_items = set()
        for lst in self.lists:
            unique_items.update(lst)
        return unique_items


if __name__ == '__main__':
    # Example usage:
    list1 = ['apple', 'banana', 'orange']
    list2 = ['apple', 'mango', 'grapes']
    list3 = ['orange', 'grapes', 'pear']
    list4 = ['apple', 'banana', 'mango']
    list5 = ['apple', 'grapes', 'mango']

    lists = [list1, list2, list3, list4, list5]

    scorer = MajorityVotingScorer(lists)
    result = scorer.score_items()
    print(result)
    result = scorer.score_items(only_keys_array=True)

    print(result)
