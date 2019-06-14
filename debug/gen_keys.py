import json
import sys
from util import flatten
from collections import defaultdict

def gen_keys(fname, out):
    key_dict = defaultdict(list)
    cluster_len = defaultdict(list)
    buckets = {1: '1', 2: '2', 3: '3-4', 4:'3-4', 5: '5-6', 6: '5-6', 7: '7-9', 8: '7-9', 9: '7-9', 10: '10+'}
    with open(fname) as f:
        for line in f:
            datum = json.loads(line)
            bucket = min(10, len(datum['sentences']))
            key_dict[buckets[bucket]].append(datum['doc_key'])
            for cluster in datum['clusters']:
                cluster = sorted((s,e) for s,e in cluster)
                cluster_len[buckets[bucket]].append(cluster[-1][0] - cluster[0][0])
                # pairs_lens = []
                # for i in range(len(cluster)):
                    # for j in range(i + 1, len(cluster)):
                        # pairs_lens.append(cluster[j][0] - cluster[i][0])
                # cluster_len[buckets[bucket]].append(sum(pairs_lens) / len(pairs_lens))

    print(sum(flatten([x for x in cluster_len.values()])) / len(flatten((x for x in cluster_len.values()))))
    for k, v in key_dict.items():
        print(k, sum(cluster_len[k]) / len(cluster_len[k]))
        with open(out.replace('txt', k + '.txt'), 'w') as f:
            for key in v:
                f.write(key + '\n')

gen_keys(sys.argv[1], sys.argv[2])
