import os
import re

str_new_doc = 'http://www.aksw.org/gerbil/NifWebService/request_'
str_doc_length = 'Document length'
str_filter = 'Filtered to'
dirpath = 'log/evaluation'

"""

This script reads the log files in `dirpath` and calculates following metrics:

* Average document length
* Average number of candidate entities per mention without a filter
* Average number of candidate entities per mention with spaCy-based filter
* Average number of candidate entities per mention with BERT-based filter
* These averages a) once for each dataset and b) their mean for all datasets

"""

# Max. number of candidate entities per mention (must equal the setting `candidates_limit` in the experiments, cf. gerbil.py)
limit = 500

all_docs_spacy_filtered_sum = []
all_docs_bert_filtered_sum = []
all_docs_unfiltered_sum = []

for filename in os.listdir(dirpath):
    #filename = '20210714-090614-el-gerbil.dbpediaspotlight.bert.important.log'
    if not filename.endswith('log'):
        continue

    if 'none' in filename: # skip experiments without filter
        continue

    is_without_filter = 'none' in filename # for the experiments without filter (in they should not be skipped)
    is_with_spacy = 'spacy' in filename
    is_with_bert = 'bert' in filename

    print(f'\n{filename}')
    #re.findall('\d+-\d+-el-gerbil.(\w+)\.\(w+)\.important\.log')

    current_doc_id = None
    results = {}

    # Extract values from log

    f = open(os.path.join(dirpath, filename))

    for line in f:
        # Is new document?
        if str_new_doc in line:
            # If this is not the very first document, add results of the last document
            if current_doc_id is not None:
                results[current_doc_id] = {'doc_id' : current_doc_id, 'doc_length' : doc_length, 'filter_stats' : filter_stats}

            current_doc_id = re.findall('http\:\/\/www\.aksw\.org\/gerbil\/NifWebService\/request_(\d+)', line)[0]
            #print(f'=== Document {current_doc_id} ===')
            # Reset mention ID and filter stats for new document
            mention_id = 0
            filter_stats = {}

        # Is document length
        if str_doc_length in line:
            doc_length = int(re.findall('Document length: (\d+)', line)[0])
            #print(f'Document length: {doc_length}')

        # Is filter?
        if str_filter in line:
            filtered_candidates = int(re.findall('Filtered to (\d+)', line)[0])
            unfiltered_candidates = int(re.findall('Filtered to \d+/(\d+)', line)[0])
            # Apply limit
            if filtered_candidates > limit:
                filtered_candidates = limit
            if unfiltered_candidates > limit:
                unfiltered_candidates = limit

            #print(f'{filtered_candidates}/{unfiltered_candidates}')
            filter_stats[mention_id] = [filtered_candidates, unfiltered_candidates]
            mention_id += 1

    f.close()

    # Calculate metrics

    doc_count = len(results.keys())
    doc_length_sum = 0
    unfiltered_sum = 0
    filtered_sum = 0
    unfiltered_avg = 0
    filtered_avg = 0
    factions_sum = 0

    # For each document
    for _, doc in results.items():
        doc_length_sum += doc['doc_length']

        doc_filtered_sum = 0
        doc_unfiltered_sum = 0
        doc_faction_sum = 0

        mention_count = len(doc['filter_stats'].keys())

        # For each mention of the document sum filtered and unfiltered (for micro) and sum factions (for macro)
        for _, mention in doc['filter_stats'].items():
            filtered = mention[0]
            unfiltered = mention[1]

            if unfiltered == 0:
                faction = 0 # TODO nothing or all was filtered?
            else:
                faction = filtered/unfiltered

            doc_filtered_sum += filtered
            doc_unfiltered_sum += unfiltered
            doc_faction_sum += faction

        if mention_count == 0:
            doc_filtered_avg = 0 # no mentions -> no candidates to be filtered
            doc_unfiltered_avg = 0 # no mentions -> no candidates to be filtered
        else:
            doc_filtered_avg = doc_filtered_sum / mention_count # average filtered candidates per mention
            doc_unfiltered_avg = doc_unfiltered_sum / mention_count # average unfiltered candidates per mention

        # add the filtered and unfiltered counts of the document (for micro) and the faction for the whole document (for macro)
        filtered_sum += doc_filtered_sum
        unfiltered_sum += doc_unfiltered_sum
        filtered_avg += doc_filtered_avg
        unfiltered_avg += doc_unfiltered_avg

        if mention_count == 0:
            factions_sum += 0 # TODO nothing or all was filtered?
        else:
            factions_sum += doc_faction_sum/mention_count

    doc_length_avg = doc_length_sum/doc_count

    if not is_without_filter: # for the experiments without filter
        faction_micro = filtered_sum/unfiltered_sum
        #faction_macro = factions_sum/doc_count

    avg_filtered_micro = filtered_avg/doc_count
    avg_unfiltered_micro = unfiltered_avg/doc_count

    print(f'Average document length: {doc_length_avg}')

    if not is_without_filter: # for the experiments without filter
        #print(f'Filtered/unfiltered (macro): {faction_macro}')
        print(f'Filtered/unfiltered (micro): {faction_micro}')
        print(f'Average unfiltered (micro): {avg_unfiltered_micro}')
        print(f'Average filtered (micro): {avg_filtered_micro}')

    # Macro metrics for all documents
    all_docs_unfiltered_sum.append(avg_unfiltered_micro)

    if is_with_spacy:
        all_docs_spacy_filtered_sum.append(avg_filtered_micro)
    if is_with_bert:
        all_docs_bert_filtered_sum.append(avg_filtered_micro)

# Metrics for all documents
print('')
# unfiltered
avg_unfiltered = sum(all_docs_unfiltered_sum) / len(all_docs_unfiltered_sum)
print(f'Average unfiltered: {avg_unfiltered}')
# spaCy
avg_filtered_spacy = sum(all_docs_spacy_filtered_sum) / len(all_docs_spacy_filtered_sum)
print(f'Average filtered spaCy: {avg_filtered_spacy}')
# BERT
avg_filtered_bert = sum(all_docs_bert_filtered_sum) / len(all_docs_bert_filtered_sum)
print(f'Average filtered BERT: {avg_filtered_bert}')
