CURRENT_MODELS = {
    'google_base_cased': ('bert-base-cased', '/checkpoint/mandarj/coref_data/bert_models/cased_L-12_H-768_A-12/bert_model.ckpt'),
    'google_base_uncased': ('bert-base-uncased', '/checkpoint/mandarj/coref_data/bert_models/uncased_L-12_H-768_A-12/bert_model.ckpt'),
    'google_large_cased': ('bert-large-cased', '/checkpoint/mandarj/coref_data/bert_models/cased_L-24_H-1024_A-16/bert_model.ckpt'),
    'google_large_uncased': ('bert-large-uncased', '/checkpoint/mandarj/coref_data/bert_models/uncased_L-24_H-1024_A-16/bert_model.ckpt'),
    'base_pair_internal': ('bert-base-cased', '/private/home/omerlevy/cookie/fast_models/base_pair_internal/checkpoint_best.pt'),
    'base_pair_external': ('bert-base-cased', '/private/home/omerlevy/cookie/fast_models/base_pair_external/checkpoint_best.pt'),
    'base_random': ('bert-base-cased', '/private/home/omerlevy/cookie/fast_models/base_random/checkpoint_best.pt')
}

MODEL_CAT_TO_GOOGLE_DIR = {'bert-base-cased': 'cased_L-12_H-768_A-12', 'bert-base-uncased': 'uncased_L-12_H-768_A-12', 'bert-large-cased': 'cased_L-24_H-1024_A-16', 'bert-large-uncased': 'uncased_L-24_H-1024_A-16', 'bert-base-uncased-1024': 'uncased_L-12_H-768_A-12'}
