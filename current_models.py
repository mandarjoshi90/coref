CURRENT_MODELS = {
    'google_large_cased': ('bert-large-cased', '/checkpoint/danqi/coref_eval/bert_models/cased_L-24_H-1024_A-16/bert_model.ckpt'),
    'small_batch_random': ('bert-large-cased', '/checkpoint/omerlevy/span_bert_models/cased/random/checkpoint_best.pt'),
    'small_batch_no_nsp_random': ('bert-large-cased', '/checkpoint/omerlevy/mandar_data/pretraining_models/no_nsp_random/checkpoint_best.pt'),
    'small_batch_no_nsp_pair': ('bert-large-cased', '/checkpoint/omerlevy/mandar_data/pretraining_models/no_nsp_pair/checkpoint_best.pt'),
    'small_batch_no_nsp_pair_1.2m': ('bert-large-cased', '/checkpoint/omerlevy/mandar_data/pretraining_models/no_nsp_pair/checkpoint_37_1200000.pt'),
    'small_batch_no_nsp_geo_1.2m': ('bert-large-cased', '/checkpoint/omerlevy/slow_models/geo/checkpoint_37_1200000.pt'),
    'small_batch_random_1.2m': ('bert-large-cased', '/checkpoint/omerlevy/span_bert_models/cased/random/checkpoint_27_1200000.pt'),
    'small_batch_geo_1.2m': ('bert-large-cased', '/checkpoint/omerlevy/mandar_data/pretraining_models/geo_span_0.2/checkpoint_27_1200000.pt'),
    'small_batch_np_1.2m': ('bert-large-cased', '/checkpoint/omerlevy/mandar_data/pretraining_models/np_span/checkpoint_27_1200000.pt'),
    'small_batch_ner_1.2m': ('bert-large-cased', '/checkpoint/omerlevy/mandar_data/pretraining_models/ner_span/checkpoint_27_1200000.pt'),
    'small_batch_word_1.2m': ('bert-large-cased', '/checkpoint/omerlevy/mandar_data/pretraining_models/word/checkpoint_27_1200000.pt')
}

MODEL_CAT_TO_GOOGLE_DIR = {'bert-base-cased': 'cased_L-12_H-768_A-12', 'bert-base-uncased': 'uncased_L-12_H-768_A-12', 'bert-large-cased': 'cased_L-24_H-1024_A-16', 'bert-large-uncased': 'uncased_L-24_H-1024_A-16', 'bert-base-uncased-1024': 'uncased_L-12_H-768_A-12'}

