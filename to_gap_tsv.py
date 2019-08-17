import sys
import json
import util

def find_pronoun_cluster(prediction, pronoun_subtoken_span, cluster_key='predicted_clusters'):
    for cluster in prediction[cluster_key]:
        if pronoun_subtoken_span in cluster:
            return cluster
    return []

def read_json(json_file):
    data = {}
    with open(json_file) as f:
        for line in f:
            line = json.loads(line)
            data[line['doc_key']] = line
    return data

def is_aligned(span1, span2):
    if span1[0] >= span2[0] and span1[1] <= span2[1]:
        return True
    if span2[0] >= span1[0] and span2[1] <= span1[1]:
        return True
    return False

def is_substring_aligned(span1, sents, name):
    span_text = ' '.join(sents[span1[0]:span1[1] + 1])
    if span_text in name or name in span_text:
        return True
    return False

def read_tsv_file(tsv_file):
    tsv = {}
    with open(tsv_file) as f:
        for line in f:
            cols = line.split('\t')
            tsv[cols[0]] = cols
    return tsv

def convert(json_file, tsv_file):
    data = read_json(json_file)
    tsv = read_tsv_file(tsv_file) if tsv_file is not None else None
    predictions = ['\t'.join(['ID', 'A-coref', 'B-coref'])]
    for key, datum in data.items():
        prediction = data[key]
        sents = util.flatten(prediction['sentences'])
        if tsv is not None:
            print(list(enumerate(tsv[key])))
            a_offset, b_offset, pronoun_offset = tuple(map(int, tsv[key][5].split(':'))), tuple(map(int, tsv[key][8].split(':'))), tuple(map(int, tsv[key][3].split(':')))
            assert ' '.join(sents[a_offset[0]:a_offset[1]]) == tsv[key][4], (sents[a_offset[0]:a_offset[1]], tsv[key][4])
            assert ' '.join(sents[b_offset[0]:b_offset[1]]) == tsv[key][7], (sents[b_offset[0]:b_offset[1]], tsv[key][7])
            assert ' '.join(sents[pronoun_offset[0]:pronoun_offset[1]]) == tsv[key][2], (sents[pronoun_offset[0]:pronoun_offset[1]], tsv[key][2])
        # continue
        pronoun_cluster = find_pronoun_cluster(prediction, prediction['pronoun_subtoken_span'])
        a_coref, b_coref = 'FALSE', 'FALSE'
        a_text, b_text = (tsv[key][4], tsv[key][7]) if tsv is not None else (None, None)
        for span in pronoun_cluster:
            a_aligned = is_aligned(span, prediction['a_subtoken_span']) if tsv is None else is_substring_aligned(span, sents, a_text)
            b_aligned = is_aligned(span, prediction['b_subtoken_span']) if tsv is None else is_substring_aligned(span, sents, b_text)

            if a_aligned:
                a_coref = 'TRUE'
            if b_aligned:
                b_coref = 'TRUE'
        predictions += ['\t'.join([key, a_coref, b_coref])]
    # write file
    with open(json_file.replace('jsonlines', 'tsv'), 'w') as f:
        f.write('\n'.join(predictions))


if __name__ == '__main__':
    json_file = sys.argv[1]
    tsv_file = sys.argv[2] if len(sys.argv) == 3 else None
    convert(json_file, tsv_file)
