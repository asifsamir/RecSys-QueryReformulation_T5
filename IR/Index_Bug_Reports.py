import json
from tqdm import tqdm
from IR.Indexer.IndexWriter import IndexWriter

# create index writer object
indexer = IndexWriter()


# read a list from json file
with open('../Data/Augmented/Cleaned_newLine_Data.json', 'r') as f:
    bug_reports = json.load(f)

# iterate through the json list using tqdm
for bug_report in tqdm(bug_reports):
    # get the bug_id, bug_title, bug_description, repo, ground_truths from the json list
    bug_id = bug_report['bug_id']
    bug_title = bug_report['bug_title']
    bug_description = bug_report['bug_description']
    repo = bug_report['repo']
    ground_truths = bug_report['ground_truth']

    # print(bug_id, bug_title, bug_description, repo, ground_truths)
    # break



    # index the document
    response = indexer.index_data(bug_id, bug_title, bug_description, repo, ground_truths)

    # check the response if it failed
    if response['result'] != 'created':
        print(f"Failed to index document with ID: {bug_id}")
        print(response)
        print('--' * 15)





