CURRENT_MODELS = {
    'google_base_cased': ('bert-base-cased', '/checkpoint/danqi/coref_eval/bert_models/cased_L-12_H-768_A-12/bert_model.ckpt'),
    'base_random': ('bert-base-cased', '/private/home/omerlevy/cookie/fast_models/base_random/checkpoint_best.pt'),
    'base_no_nsp_random': ('bert-base-cased', '/checkpoint/danqi/fast_models/base_no_nsp_random/checkpoint_best.pt'),
    'base_pair_internal': ('bert-base-cased', '/private/home/omerlevy/cookie/fast_models/base_pair_internal/checkpoint_best.pt'),
    'base_pair_external': ('bert-base-cased', '/private/home/omerlevy/cookie/fast_models/base_pair_external/checkpoint_best.pt'),
    'google_large_cased': ('bert-large-cased', '/checkpoint/danqi/coref_eval/bert_models/cased_L-24_H-1024_A-16/bert_model.ckpt'),
    'random': ('bert-large-cased', '/private/home/omerlevy/cookie/fast_models/random/checkpoint_best.pt'),
    'no_nsp_random': ('bert-large-cased', '/private/home/omerlevy/cookie/fast_models/no_nsp_random/checkpoint_best.pt'),
    'pair_external': ('bert-large-cased', '/private/home/omerlevy/cookie/fast_models/pair_external/checkpoint_best.pt'),
    'small_batch_random': ('bert-large-cased', '/checkpoint/omerlevy/span_bert_models/cased/random/checkpoint_best.pt'),
    'small_batch_no_nsp_random': ('bert-large-cased', '/checkpoint/omerlevy/mandar_data/pretraining_models/no_nsp_random/checkpoint_best.pt'),
    'small_batch_no_nsp_pair': ('bert-large-cased', '/checkpoint/omerlevy/mandar_data/pretraining_models/no_nsp_pair/checkpoint_best.pt')
}

MODEL_CAT_TO_GOOGLE_DIR = {'bert-base-cased': 'cased_L-12_H-768_A-12', 'bert-base-uncased': 'uncased_L-12_H-768_A-12', 'bert-large-cased': 'cased_L-24_H-1024_A-16', 'bert-large-uncased': 'uncased_L-24_H-1024_A-16', 'bert-base-uncased-1024': 'uncased_L-12_H-768_A-12'}
