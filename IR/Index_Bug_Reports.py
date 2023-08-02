import json
from tqdm import tqdm
from IR.Indexer.IndexWriter import IndexWriter
import os
# create index writer object
indexer = IndexWriter()


key_files = '../Data/GroundTruth/File_Mapping'
corpus_dir = 'F:\Data\Blizzzard\BLIZZARD-Replication-Package-ESEC-FSE2018-master\Corpus'
# first list the ckeys files in the directory
files = os.listdir(key_files)

# indexing will be each corpus basis

# iterate over the list of files
for file in tqdm(files, desc="Repository Progress:", position=0, leave=False):
    #replace .ckeys from the end of the file
    repo = file.replace('.ckeys', '')
    key_val_dict = {}

    # load the mapping file
    with open(os.path.join(key_files, file), 'r') as f:
        # read the mapping file line by line
        for line in f:
            # split the line on first occurrence of ':' into two parts
            key, value = line.split(':', 1)

            # add the key value pair to the dictionary
            key_val_dict[key] = value.strip()

    # now list the files in the corpus directory of the repo
    repo_files = os.listdir(os.path.join(corpus_dir, repo))

    # iterate over the list of files
    for repo_file in tqdm(repo_files, desc="File Progress:", leave=False, position=1):
        file_id = repo_file.replace('.java', '')
        # get the file name from the mapping dictionary
        original_file_name_uri = key_val_dict[file_id]

        # now read the file
        with open(os.path.join(corpus_dir, repo, repo_file), 'r') as source_code_file:
            source_code = source_code_file.read()

            # index the document
            response = indexer.index_data(source_code, original_file_name_uri)

            # check the response if it failed
            if response['result'] != 'created':
                print(f"Failed to index document with ID: {file_id}")




# # read a list from json file
# with open('../Data/Augmented/Cleaned_newLine_Data.json', 'r') as f:
#     bug_reports = json.load(f)
#
# # iterate through the json list using tqdm
# for bug_report in tqdm(bug_reports):
#     # get the bug_id, bug_title, bug_description, repo, ground_truths from the json list
#     bug_id = bug_report['bug_id']
#     bug_title = bug_report['bug_title']
#     bug_description = bug_report['bug_description']
#     repo = bug_report['repo']
#     ground_truths = bug_report['ground_truth']
#
#     # print(bug_id, bug_title, bug_description, repo, ground_truths)
#     # break
#
#
#
#     # index the document
#     response = indexer.index_data(bug_id, bug_title, bug_description, repo, ground_truths)
#
#     # check the response if it failed
#     if response['result'] != 'created':
#         print(f"Failed to index document with ID: {bug_id}")
#         print(response)
#         print('--' * 15)





