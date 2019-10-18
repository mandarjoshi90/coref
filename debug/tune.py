import os
import argparse
from current_models import CURRENT_MODELS, MODEL_CAT_TO_GOOGLE_DIR
# 512 always performs best for our models.
max_sents = {128: 11, 256: 5, 384: 3, 512: 3}
# max_sents = {64: 23, 128:11, 256: 5, 384: 3, 512: 3}

bert_lrs = [1e-5, 2e-5]
task_lrs = [1e-4, 2e-4, 3e-4] #, 5e-4, 1e-3]

def get_conf_name(model, seg_len, bert_lr, task_lr, task_optimizer=None, eps=None):
    if task_optimizer is None and eps is None:
        return '{}_sl{}_blr{}_tlr{}'.format(model, seg_len, bert_lr, task_lr)
    else:
        return '{}_sl{}_blr{}_tlr{}_to{}_eps{}'.format(model, seg_len, bert_lr, task_lr, task_optimizer, eps)

def get_conf_lines(model, seg_len, bert_lr, task_lr, bert_model_dir, checkpoint, task_optimizer=None, eps=None):
    lines = []
    casing = 'uncased' if 'uncased' in bert_model_dir else 'cased'
    lines += [get_conf_name(model, seg_len, bert_lr, task_lr, task_optimizer, eps) + ' = ${best} {']
    lines += ['  num_docs = {}'.format(args.num_docs)]
    lines += ['  bert_learning_rate = {}'.format(bert_lr)]
    lines += ['  task_learning_rate = {}'.format(task_lr)]
    lines += ['  max_segment_len = {}'.format(seg_len)]
    lines += ['  ffnn_size = {}'.format(args.ffnn_size)]
    lines += ['  train_path = {}/{}/train.{}.{}.jsonlines'.format(args.data_dir, casing, args.lang, seg_len)]
    lines += ['  eval_path = {}/{}/dev.{}.{}.jsonlines'.format(args.data_dir, casing, args.lang, seg_len)]
    lines += ['  conll_eval_path = {}/gold_conll/dev.{}.v4_gold_conll'.format(args.data_dir, args.lang)]
    lines += ['  max_training_sentences = {}'.format(max_sents[seg_len])]
    lines += ['  bert_config_file = {}/bert_config.json'.format(bert_model_dir)]
    lines += ['  vocab_file = {}/vocab.txt'.format(bert_model_dir)]
    lines += ['  tf_checkpoint = {}/bert_model.ckpt'.format(bert_model_dir)]
    lines += ['  init_checkpoint = {}'.format(checkpoint)]
    if task_optimizer is not None:
        lines += ['  task_optimizer = {}'.format(task_optimizer)]
    if eps is not None:
        lines += ['  adam_eps = {}'.format(eps)]

    lines += ['}\n']
    return lines

def generate(args):
    num_confs = 0
    with open(args.conf_file, 'a') as f:
        for (model, (model_cat, ckpt_file)) in CURRENT_MODELS.items():
            bert_model_dir = os.path.join(args.data_dir, 'bert_models', MODEL_CAT_TO_GOOGLE_DIR[model_cat])
            for sl in max_sents.keys():
                for bert_lr in bert_lrs:
                    for task_lr in task_lrs:
                        lines = get_conf_lines(model, sl, bert_lr, task_lr, bert_model_dir, ckpt_file)
                        if args.trial:
                            print('\n'.join(lines) + '\n')
                        else:
                            f.write('\n'.join(lines) + '\n')
                        num_confs += 1
    print('{} configs written to {}'.format(num_confs, args.conf_file))


def run_slrm(args):
    with open(args.jobs_file) as f:
        for i, line in enumerate(f):
            job = line.strip()
            os.system('sbatch -J {} {} {}'.format('coref_' + job, args.slrm_file, job))
            print('starting job {}'.format(job))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help='High level coref data dir')
    parser.add_argument("--generate_configs", action='store_true', help='appends configs to --conf_file')
    parser.add_argument("--run_jobs", action='store_true', help='send jobs from --jobs_file to the cluster')

    # you mostly don't need to touch these below
    parser.add_argument("--trial", action='store_true', help='Print config to stdout if true')
    parser.add_argument("--conf_file", default='experiments.conf', type=str, help='Output config file')
    parser.add_argument("--jobs_file", default='torun.txt', type=str, help='file contraining list of jobs')
    parser.add_argument("--slrm_file", default='slurm_coref.slrm', type=str, help='Slrm file')
    parser.add_argument("--num_docs", default=2802, type=int)
    parser.add_argument("--ffnn_size", default=3000, type=int)
    parser.add_argument("--lang", default='english', type=str)
    args = parser.parse_args()
    if not args.generate_configs and not args.run_jobs:
        print('Only one of --generate_configs and --run_jobs should be true')
    elif args.generate_configs and args.run_jobs:
        print('Only one of --generate_configs and --run_jobs should be true. First generate the configs with --generate_configs only. Make sure you have the right list in the jobs_file. The run with --run_jobs.')
    elif args.generate_configs:
        generate(args)
    else:
        run_slrm(args)

