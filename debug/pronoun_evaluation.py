import json
from collections import defaultdict
import sys
from util import flatten

def get_pronoun_mention_pairs(clusters, pronouns):
    pronoun_mention_pairs = []
    unaccounted = set()
    for pronoun in pronouns:
        has_cluster = False
        for cluster in clusters:
            has_pronoun = any([pronoun == s and s == e for (s,e) in cluster])
            if has_pronoun:
                assert has_cluster  is False
                has_cluster = True
                for s, e in cluster:
                    if not (pronoun == s and s == e):
                        pronoun_mention_pairs.append((pronoun, (s,e)))
        if not has_cluster:
            unaccounted.add(pronoun)
            pronoun_mention_pairs.append((pronoun, None))
    return set(pronoun_mention_pairs), set(unaccounted)

def get_mention_pairs(clusters, pronouns):
    pronoun_mention_pairs = []
    unaccounted = set()
    for pronoun in pronouns:
        has_cluster = False
        ps, pe = pronoun
        for cluster in clusters:
            has_pronoun = any([ps == s and pe == e for s, e in cluster])
            if has_pronoun:
                assert has_cluster  is False
                has_cluster = True
                for s, e in cluster:
                    if ps == s and pe == e:
                        pronoun_mention_pairs.append(((ps, pe), (s,e)))
        if not has_cluster:
            unaccounted.add((ps, pe))
            pronoun_mention_pairs.append(((ps, pe), None))
    return set(pronoun_mention_pairs), set(unaccounted)

def evaluate(fname):
    p, r, f1 = [], [], []
    pronoun_text = defaultdict(int)
    num_gold_pairs, num_pred_pairs = 0, 0
    total_gold_singletons, total_pred_singletons, total_singleton_intersection = 0, 0, 0
    with open(fname) as f:
        for line in f:
            datum = json.loads(line)
            tokens = flatten(datum['sentences'])
            #pronouns = flatten(datum['clusters'])
            pair_fn = get_mention_pairs
            # for pidx in pronouns:
                # pronoun_text[(tokens[pidx].lower())] += 1
            gold_pronoun_mention_pairs, gold_singletons = pair_fn(datum['clusters'], flatten(datum['clusters']))
            pred_pronoun_mention_pairs, pred_singletons = pair_fn(datum['predicted_clusters'], flatten(datum['predicted_clusters']))
            total_gold_singletons += len(gold_singletons)
            total_pred_singletons += len(pred_singletons)
            total_singleton_intersection += len(gold_singletons.intersection(pred_singletons))
            intersection = gold_pronoun_mention_pairs.intersection(pred_pronoun_mention_pairs)
            num_gold_pairs += len(gold_pronoun_mention_pairs)
            num_pred_pairs += len(pred_pronoun_mention_pairs)
            this_recall = len(intersection) / len(gold_pronoun_mention_pairs) if  len(gold_pronoun_mention_pairs) > 0 else 1.0
            this_prec = len(intersection) / len(pred_pronoun_mention_pairs) if  len(pred_pronoun_mention_pairs) > 0 else 1.0
            this_f1 = 2 * this_recall * this_prec / (this_recall + this_prec) if this_recall + this_prec > 0 else 0
            p += [this_prec]
            r += [this_recall]
            f1 += [this_f1]
    print('gold_singletons: {}, pred_singletons: {} intersection: {}'.format(total_gold_singletons, total_pred_singletons, total_singleton_intersection))
    print('num_gold: {}, num_pred: {}, P: {}, R: {} F1: {}'.format(num_gold_pairs, num_pred_pairs, sum(p) / len(p), sum(r) / len(r), sum(f1) / len(f1)))
    #print(sum(pronoun_text.values()), sorted(list(pronoun_text.items()), key=lambda k : k[1]))

if __name__ == '__main__':
    evaluate(sys.argv[1])
