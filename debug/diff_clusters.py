import json
import sys
import numpy as np

def diff(input_file, output_file):
    output = []
    with open(input_file) as f:
        for line in f:
            datum = json.loads(line)
            pred = sorted([[(s, e) for s,e in cluster] for cluster in datum['predicted_clusters']], key=lambda x: x[0])
            gold_clusters = sorted([[(s, e) for s,e in cluster] for cluster in datum['clusters']], key=lambda x: x[0])
            pred_annotations = []
            output.append(datum)
            covered = [False for i in range(len(gold_clusters))]
            for cluster in pred:
                overlap_fn = lambda k: len(set(cluster).intersection(set(k)))
                scores = [len(set(cluster).intersection(set(k))) / len(cluster) for k in gold_clusters]
                best_match = np.argmax(scores)
                pred_annotations.append([])
                for s, e in cluster:
                    present = 0 if (s, e) in gold_clusters[best_match] else 1
                    pred_annotations[-1].append([s, e, present])
                if scores[best_match] > 0:
                    covered[best_match] = True
                    for s, e in set(gold_clusters[best_match]).difference(cluster):
                        pred_annotations[-1].append([s, e, 2])
            for i in range(len(gold_clusters)):
                if not covered[i]:
                    pred_annotations.append([])
                    for s, e in gold_clusters[i]:
                        pred_annotations[-1].append([s, e, 2])
            datum['pred_annotations'] = pred_annotations
            datum['predicted_clusters'] = pred
            datum['clusters'] = gold_clusters
    with open(output_file, 'w') as f:
        for datum in output:
            f.write(json.dumps(datum) + '\n')

if __name__ == '__main__':
    diff(sys.argv[1], sys.argv[2])
