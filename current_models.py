CURRENT_MODELS = {
    'google_base_cased': ('bert-base-cased', '/checkpoint/danqi/coref_eval/bert_models/cased_L-12_H-768_A-12/bert_model.ckpt'),
    'base_small_batch_random': ('bert-base-cased', '/checkpoint/omerlevy/fix_models/base_random/checkpoint_best.pt'),
    'base_small_batch_no_nsp_random': ('bert-base-cased', '/checkpoint/omerlevy/fix_models/base_no_nsp_random/checkpoint_best.pt'),
    'base_small_batch_no_nsp_pair': ('bert-base-cased', '/checkpoint/omerlevy/fix_models/base_pair_external/checkpoint_best.pt'),
    'base_small_batch_geo': ('bert-base-cased', '/checkpoint/omerlevy/fix_models/base_geo/checkpoint_best.pt'),
    'base_small_batch_ner': ('bert-base-cased', '/checkpoint/omerlevy/fix_models/base_ner/checkpoint_best.pt'),
    'base_small_batch_np': ('bert-base-cased', '/checkpoint/omerlevy/fix_models/base_np/checkpoint_best.pt'),
    'base_small_batch_word': ('bert-base-cased', '/checkpoint/omerlevy/fix_models/base_word/checkpoint_best.pt'),
    'base_big_batch_random': ('bert-base-cased', '/private/home/omerlevy/cookie/fast_models/base_random/checkpoint_best.pt'),
    'base_big_batch_no_nsp_random': ('bert-base-cased', '/checkpoint/danqi/fast_models/base_no_nsp_random/checkpoint_best.pt'),
    'base_big_batch_no_nsp_random_200k': ('bert-base-cased', '/checkpoint/danqi/fast_models/base_no_nsp_random/checkpoint_50_200000.pt'),
    'base_big_batch_no_nsp_pair': ('bert-base-cased', '/private/home/omerlevy/cookie/fast_models/base_pair_external/checkpoint_best.pt'),
    'base_big_batch_no_nsp_pair_200k': ('bert-base-cased', '/checkpoint/omerlevy/fast_models/base_pair_external/checkpoint_50_200000.pt'),
    'base_big_batch_geo_200k': ('bert-base-cased', '/checkpoint/danqi/fast_models/base_geo/checkpoint_50_200000.pt'),
    'base_big_batch_ner_200k': ('bert-base-cased', '/checkpoint/danqi/fast_models/base_ner/checkpoint_50_200000.pt'),
    'base_big_batch_np_200k': ('bert-base-cased', '/checkpoint/omerlevy/fast_models/base_np/checkpoint_50_200000.pt'),
    'base_big_batch_word_200k': ('bert-base-cased', '/private/home/omerlevy/cookie/fast_models/base_word/checkpoint_50_200000.pt'),
    'google_large_cased': ('bert-large-cased', '/checkpoint/danqi/coref_eval/bert_models/cased_L-24_H-1024_A-16/bert_model.ckpt'),
    'small_batch_random': ('bert-large-cased', '/checkpoint/omerlevy/span_bert_models/cased/random/checkpoint_best.pt'),
    'small_batch_no_nsp_random': ('bert-large-cased', '/checkpoint/omerlevy/mandar_data/pretraining_models/no_nsp_random/checkpoint_best.pt'),
    'small_batch_no_nsp_pair': ('bert-large-cased', '/checkpoint/omerlevy/mandar_data/pretraining_models/no_nsp_pair/checkpoint_best.pt'),
    'big_batch_random': ('bert-large-cased', '/checkpoint/omerlevy/fast_models/random_2/checkpoint_best.pt'),
    'big_batch_no_nsp_random': ('bert-large-cased', '/checkpoint/omerlevy/fast_models/no_nsp_random_2/checkpoint_best.pt'),
    'big_batch_no_nsp_pair': ('bert-large-cased', '/checkpoint/danqi/fast_models/pair_external_3/checkpoint_best.pt')
}

MODEL_CAT_TO_GOOGLE_DIR = {'bert-base-cased': 'cased_L-12_H-768_A-12', 'bert-base-uncased': 'uncased_L-12_H-768_A-12', 'bert-large-cased': 'cased_L-24_H-1024_A-16', 'bert-large-uncased': 'uncased_L-24_H-1024_A-16', 'bert-base-uncased-1024': 'uncased_L-12_H-768_A-12'}

