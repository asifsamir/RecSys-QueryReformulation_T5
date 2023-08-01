import os

# list files in a directory
corpus_dir = 'F:\Data\Blizzzard\BLIZZARD-Replication-Package-ESEC-FSE2018-master\Corpus'
key_dir = 'F:\Data\Blizzzard\BLIZZARD-Replication-Package-ESEC-FSE2018-master\Lucene-Index2File-Mapping'
ground_truth_dir = '../../Data/GroundTruth/File_Mapping'
files = os.listdir(key_dir)

# iterate over the list of files
for file in files:
    mapping = ''
    #replace .ckeys from the end of the file

    repo = file.replace('.ckeys', '')

    # open the file and read line by line
    with open(os.path.join(key_dir, file), 'r') as f:
        for line in f:
            # split the line on first occurrence of ':' into two parts
            key, value = line.split(':', 1)
            # replace 'C:\My MSc\ThesisWorks\Crowdsource_Knowledge_Base\M4CPBugs\experiment\ssystems\ecf' from the value
            value = value.replace('C:\My MSc\ThesisWorks\Crowdsource_Knowledge_Base\M4CPBugs\experiment\ssystems\\' + repo + '\\', '')


            # now replace '\' with '/' in the value
            value = value.replace('\\', '/')

            # now join the key and value with ':'
            clean_line = ':'.join([key, value])

            # append the new line to the list
            mapping += clean_line

    # save mapping string in ground_truth directory
    with open(os.path.join(ground_truth_dir, file), 'w') as file:
        file.write(mapping)



