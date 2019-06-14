import util
import json

def read_file(fn):
    js_dict = {}
    js_list  = []
    with open(fn) as f:
        for line in f:
            js = json.loads(line)
            js_dict[js['doc_key']] = js
            js_list += [js]
            # import ipdb
            # ipdb.set_trace()
    # print('read', len(js_dict), js_dict.keys())
    return js_list

def compare(bert_json, org_json, key='nw/xinhua/00/chtb_0060_0'):
    bert_json = read_file(bert_json)
    org_json = read_file(org_json)
    bert_text = [item for sublist in bert_json[key]['sentences'] for item in sublist]
    org_text = [item for sublist in org_json[key]['sentences'] for item in sublist]
    print(list(enumerate(zip(bert_json[key]['subtoken_map'], bert_text))))
    for cl in  bert_json[key]['clusters']:
        strings = []
        for ((bs, be)) in cl:
            os, oe =  bert_json[key]['subtoken_map'][bs], bert_json[key]['subtoken_map'][be]
            strings.append((bert_text[bs: be+1], bs, be, os, oe, org_text[os:oe+1]))
        print(strings)
    
    print('---')
    for cl in  org_json[key]['clusters']:
        strings = []
        for ((bs, be)) in cl:
            strings.append((org_text[bs: be+1], bs, be))
        print(strings)

def compare_json(json1, json2):
    json1 = read_file(json1)
    json2 = read_file(json2)
    for i, (l1, l2) in enumerate(zip(json1, json2)):
        assert l1['doc_key'] == l2['doc_key']
        if tuple(util.flatten(l1['sentences'])) != tuple(util.flatten(l2['sentences'])):
            print(i, l1['doc_key'], list(enumerate(util.flatten(l1['sentences']))), list(enumerate(util.flatten(l2['sentences']))))
            for j, (w1, w2) in enumerate(zip(util.flatten(l1['sentences']), util.flatten(l2['sentences']))):
                if w1 != w2:
                    print(j, w1, w2)
            break

compare_json('dev.english.jsonlines', 'data/seg_len_expts/dev.english.230.jsonlines')
