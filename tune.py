import os
import time
#max_sents = {64: 23, 128:11, 256: 5, 384: 3, 512: 3}
#max_sents = {256: 5, 384: 3, 512: 3}
max_sents = {384: 3, 512: 3}
lang_to_bert = {'english': 'cased_L-24_H-1024_A-16'} #, 'chinese': 'chinese_L-12_H-768_A-12', 'arabic': 'multi_cased_L-12_H-768_A-12'}
lang_to_data = {'english': 'data/seg_len_expts'} #, 'chinese': 'chinese_L-12_H-768_A-12', 'arabic': 'multi_cased_L-12_H-768_A-12'}
ffnn = [3000]
num_docs = {'english': 2802, 'chinese': 1810, 'arabic': 359}
bert_lrs = [1e-5, 2e-5]
task_lrs = [1e-4, 2e-4, 3e-4] #, 5e-4, 1e-3]
checkpoints = ['../bert/cased_L-24_H-1024_A-16/bert_model.ckpt', '../bert/wwm_cased_L-24_H-1024_A-16/bert_model.ckpt', '../huggingface-fairseq/final/geo_span_0.2/checkpoint_23_1000000.pt', '../huggingface-fairseq/final/geo_span/checkpoint_23_1000000.pt', '../huggingface-fairseq/final/no_nsp_geo_span/checkpoint_31_1000000.pt', 
        '../huggingface-fairseq/final/no_nsp_pair/checkpoint_31_1000000.pt', '../huggingface-fairseq/final/no_nsp_random/checkpoint_31_1000000.pt', '../huggingface-fairseq/final/word/checkpoint_23_1000000.pt', '/checkpoint/omerlevy/span_bert_models/cased/random/checkpoint_23_1000000.pt', '../huggingface-fairseq/final/ner_span/checkpoint_23_1000000.pt', '../huggingface-fairseq/final/np_span/checkpoint_23_1000000.pt']
checkpoints = ['../bert/wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt',
'../bert/uncased_L-24_H-1024_A-16/bert_model.ckpt',
'/checkpoint/yinhanliu/bert_large/checkpoint_24_1040000.pt',
'/private/home/omerlevy/cookie/uncased_models/word/checkpoint_24_1000000.pt',
'/private/home/omerlevy/cookie/uncased_models/geo_span/checkpoint_24_1000000.pt',
'/private/home/omerlevy/cookie/uncased_models/ner_span/checkpoint_24_1000000.pt',
'/private/home/omerlevy/cookie/uncased_models/np_span/checkpoint_24_1000000.pt',
'/private/home/omerlevy/cookie/uncased_models/no_nsp_random/checkpoint_32_1000000.pt',
'/private/home/omerlevy/cookie/uncased_models/no_nsp_geo_span/checkpoint_32_1000000.pt',
'/private/home/mandarj/workspace/huggingface-fairseq/final/uncased_no_nsp_0.2/checkpoint_32_1000000.pt',
'/checkpoint/ghazvini/geo_large/geol4/checkpoint_24_960000.pt',
'/private/home/mandarj/workspace/huggingface-fairseq/data/lspan_replace/checkpoint_24_1000000.pt']

checkpoints = ['../huggingface-fairseq/final/no_nsp_pair/checkpoint_31_1000000.pt', '../huggingface-fairseq/final/no_nsp_random/checkpoint_31_1000000.pt']
# def get_conf_name(lang, seg_len, ffnn_size, bert_lr, task_lr):
    # return '{}_sl{}_ff{}_blr{}_tlr{}'.format(lang, seg_len, ffnn_size, bert_lr, task_lr)

def get_conf_name(model, seg_len, bert_lr, task_lr):
    return '{}_sl{}_blr{}_tlr{}'.format(model, seg_len, bert_lr, task_lr)

def get_conf_name(model, seg_len, bert_lr, task_lr, task_optimizer=None, eps=None):
    if task_optimizer is None and eps is None:
        return '{}_sl{}_blr{}_tlr{}'.format(model, seg_len, bert_lr, task_lr)
    else:
        return '{}_sl{}_blr{}_tlr{}_to{}_eps{}'.format(model, seg_len, bert_lr, task_lr, task_optimizer, eps)

def get_conf_lines(lang, seg_len, ffnn_size, bert_lr, task_lr, checkpoint, task_optimizer=None, eps=None):
    model = checkpoint.split('/')[-2]
    lines = []
    # lines += ['{} = $\{best\}\{'.format(get_conf_name(lang, seg_len, ffnn_size))]
    lines += [get_conf_name(model, seg_len, bert_lr, task_lr, task_optimizer, eps) + ' = ${best} {']
    lines += ['  num_docs = {}'.format(num_docs[lang])]
    lines += ['  bert_learning_rate = {}'.format(bert_lr)]
    lines += ['  task_learning_rate = {}'.format(task_lr)]
    lines += ['  max_segment_len = {}'.format(seg_len)]
    lines += ['  ffnn_size = {}'.format(ffnn_size)]
    lines += ['  train_path = {}/train.{}.{}.jsonlines'.format(lang_to_data[lang], lang, seg_len)]
    lines += ['  eval_path = {}/dev.{}.{}.jsonlines'.format(lang_to_data[lang], lang, seg_len)]
    lines += ['  conll_eval_path = dev.{}.v4_gold_conll'.format(lang)]
    lines += ['  max_training_sentences = {}'.format(max_sents[seg_len])]
    lines += ['  bert_config_file = ../bert/{}/bert_config.json'.format(lang_to_bert[lang])]
    lines += ['  vocab_file = ../bert/{}/vocab.txt'.format(lang_to_bert[lang])]
    lines += ['  init_checkpoint = {}'.format(checkpoint)]
    if task_optimizer is not None:
        lines += ['  task_optimizer = {}'.format(task_optimizer)]
    if eps is not None:
        lines += ['  adam_eps = {}'.format(eps)]

    lines += ['}\n']
    return lines

def generate(output_file):
    lang = 'english'
    with open(output_file, 'a') as f:
        for checkpoint in checkpoints:
            for sl in max_sents.keys():
                if lang == 'chinese' and sl == 64:
                    continue
                for ff in ffnn:
                    for bert_lr in bert_lrs:
                        for task_lr in task_lrs:
                            lines = get_conf_lines(lang, sl, ff, bert_lr, task_lr, checkpoint)
                            print('\n'.join(lines) + '\n')

def generate_opt(output_file):
    lang = 'english'
    optimizers = ['adam', 'adam_weight_decay']
    eps = [1e-8, 1e-6]
    with open(output_file, 'a') as f:
        for checkpoint in checkpoints:
            for sl in max_sents.keys():
                for optimizer in optimizers:
                    for e in eps:
                        if lang == 'chinese' and sl == 64:
                            continue
                        for ff in ffnn:
                            for bert_lr in bert_lrs:
                                for task_lr in task_lrs:
                                    lines = get_conf_lines(lang, sl, ff, bert_lr, task_lr, checkpoint, optimizer, e)
                                    f.write('\n'.join(lines) + '\n')

def get_last_line(fname):
    with open(fname) as f:
        for line in f:
            pass
    return line

def run_slrm(to_run_file, slrm_file):
    with open(to_run_file) as f:
        for i, line in enumerate(f):
            if os.path.isdir('with_init/fiddle/' + line.strip()):
                last_line = get_last_line(os.path.join('with_init/fiddle/', line.strip(), 'stdout.log'))
                if '[57000]' in last_line:
                    continue
            os.system('sbatch {} {}'.format( slrm_file, line))
            print('starting job', line)
            #time.sleep(5)
            #break
            # if i == 2:
                # break

if __name__ == '__main__':
    #generate_opt('experiments.conf')
    run_slrm('torun.txt', 'slurm_coref.slrm')

