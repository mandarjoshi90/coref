import os
import time
max_sents = {64: 23, 128:11, 256: 5, 384: 3, 512: 3}
lang_to_bert = {'english': 'cased_L-12_H-768_A-12', 'chinese': 'chinese_L-12_H-768_A-12', 'arabic': 'multi_cased_L-12_H-768_A-12'}
ffnn = [1000, 3000]
num_docs = {'english': 2802, 'chinese': 1810, 'arabic': 359}
bert_lrs = [1e-5, 5e-5]
task_lrs = [2e-4, 5e-4, 1e-3]

def get_conf_name(lang, seg_len, ffnn_size, bert_lr, task_lr):
    return '{}_sl{}_ff{}_blr{}_tlr{}'.format(lang, seg_len, ffnn_size, bert_lr, task_lr)

def get_conf_lines(lang, seg_len, ffnn_size, bert_lr, task_lr):
    lines = []
    # lines += ['{} = $\{best\}\{'.format(get_conf_name(lang, seg_len, ffnn_size))]
    lines += [get_conf_name(lang, seg_len, ffnn_size, bert_lr, task_lr) + ' = ${best} {']
    lines += ['  num_docs = {}'.format(num_docs[lang])]
    lines += ['  bert_learning_rate = {}'.format(bert_lr)]
    lines += ['  task_learning_rate = {}'.format(task_lr)]
    lines += ['  max_segment_len = {}'.format(seg_len)]
    lines += ['  ffnn_size = {}'.format(ffnn_size)]
    lines += ['  train_path = data/seg_len_expts/train.{}.{}.jsonlines'.format(lang, seg_len)]
    lines += ['  eval_path = data/seg_len_expts/dev.{}.{}.jsonlines'.format(lang, seg_len)]
    lines += ['  conll_eval_path = dev.{}.v4_gold_conll'.format(lang)]
    lines += ['  max_training_sentences = {}'.format(max_sents[seg_len])]
    lines += ['  bert_config_file = ../bert/{}/bert_config.json'.format(lang_to_bert[lang])]
    lines += ['  vocab_file = ../bert/{}/vocab.txt'.format(lang_to_bert[lang])]
    lines += ['  init_checkpoint = ../bert/{}/bert_model.ckpt'.format(lang_to_bert[lang])]
    lines += ['}\n']
    return lines

def generate(output_file):
    with open(output_file, 'a') as f:
        for lang in lang_to_bert.keys():
            for sl in max_sents.keys():
                if lang == 'chinese' and sl == 64:
                    continue
                for ff in ffnn:
                    for bert_lr in bert_lrs:
                        for task_lr in task_lrs:
                            lines = get_conf_lines(lang, sl, ff, bert_lr, task_lr)
                            f.write('\n'.join(lines) + '\n')

def tune(to_run_file, sh_file):
    sh = '#!/bin/sh\npython train.py '
    with open(to_run_file) as f:
        for i, line in enumerate(f):
            with open(sh_file, 'w') as fw:
                fw.write(sh + line.strip())
                print('starting job', line)
            os.system('sbatch ' + sh_file.replace('.sh', '.slrm'))
            time.sleep(5)
            # if i == 2:
                # break

if __name__ == '__main__':
    #generate('experiments.conf')
    tune('torun.txt', 'slurm_coref.sh')

