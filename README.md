# BERT and SpanBERT for Coreference Resolution
This repository contains code and models for the paper, [BERT for Coreference Resolution: Baselines and Analysis](https://homes.cs.washington.edu/~mandar90/). Additionally, we also include the coreference resolution model from the paper [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529), which is the current state of the art on OntoNotes (79.6 F1). The model architecture itself is an extension of the [e2e-coref](https://github.com/kentonl/e2e-coref) model.

## Setup
* Install python3 requirements: `pip install -r requirements.txt`
* `export data_dir=</path/to/data_dir>`
* `./setup_all.sh`: This builds the custom kernels

## Pretrained Coreference Models
Please download following files to use the pretrained models on your data.

| Model          | F1 (%) |
| -------------- |:------:|
| BERT-base      | 73.9   |
| SpanBERT-base  | 77.7   |
| BERT-large     | 76.9   |
| SpanBERT-large | 79.6   |

`./download_pretrained.sh <model_name>` (e.g,: bert-base, bert-large, spanbert-base, spanbert-large; assumes that `$data_dir` is set)
The non-finetuned version of SpanBERT weights can be downloaded from, [here]()


## Training / Finetuning Instructions
* Finetuning a BERT/SpanBERT *large* model on OntoNotes requires access to a 32GB GPU. You might be able to train the large model with a smaller `max_seq_length`, `ffnn_size`, and `model_heads = false` on a 16GB machine; this will almost certainly result in relatively poorer performance as measured on OntoNotes.
* Running/testing a large pretrained model is still possible on a 16GB GPU. You should be able to finetune the base models on smaller GPUs.

### Setup for training
This assumes access to OntoNotes 5.0.
`./setup_training.sh <ontonotes/path/ontonotes-release-5.0> $data_dir`

* Experiment configurations are found in `experiments.conf`. Choose an experiment that you would like to run, e.g. `best`
* Training: `GPU=0 python train.py <experiment>`
* Results are stored in the `log_root` directory (see `experiments.conf`) and can be viewed via TensorBoard.
* Evaluation: `GPU=0 python evaluate.py <experiment>`


## Batched Prediction Instructions

* Create a file where each line is in the following json format (make sure to strip the newlines so each line is well-formed json):
```
{
  "clusters": [],
  "doc_key": "nw",
  "sentences": [["This", "is", "the", "first", "sentence", "."], ["This", "is", "the", "second", "."]],
  "speakers": [["spk1", "spk1", "spk1", "spk1", "spk1", "spk1"], ["spk2", "spk2", "spk2", "spk2", "spk2"]]
}
```
  * `clusters` should be left empty and is only used for evaluation purposes.
  * `doc_key` indicates the genre, which can be one of the following: `"bc", "bn", "mz", "nw", "pt", "tc", "wb"`
  * `speakers` indicates the speaker of each word. These can be all empty strings if there is only one known speaker.
* Run `GPU=0 python predict.py <experiment> <input_file> <output_file>`, which outputs the input jsonlines with predicted clusters.

## Notes
* The current config runs the Independent model.
* When running on test, change the `eval_path` and `conll_eval_path` from dev to test.
* The `model_dir` inside the `log_root` contains `stdout.log`. Check the `max_f1` after 57000 steps. For example
``
2019-06-12 12:43:11,926 - INFO - __main__ - [57000] evaL_f1=0.7694, max_f1=0.7697
``
* You can also load pytorch based model files (ending in `.pt`) which share BERT's architecture. See `pytorch_to_tf.py` for details.

### Important Config Keys
* `log_root`: This is where all models and logs are stored. Check this before running anything.
* `bert_learning_rate`: The learning rate for the BERT parameters. Typically, `1e-5` and `2e-5` work well.
* `task_learning_rate`: The learning rate for the other parameters. Typically, LRs between `0.0001` to `0.0003` work well.
* `init_checkpoint`: The checkpoint file from which BERT parameters are initialized. Both TF and Pytorch checkpoints work as long as they use the same BERT architecture. Use `*ckpt` files for TF and `*pt` for Pytorch.
* `max_segment_len`: The maximum size of the BERT context window. Larger segments work better for SpanBERT while BERT suffers a sharp drop at 512.

### Slurm
If you have access to a slurm GPU cluster, you could use the following for set of commands for training.
* `python tune.py  --generate_configs --data_dir <coref_data_dir>`: This generates multiple configs for tuning (BERT and task) learning rates, embedding models, and `max_segment_len`. This modifies `experiments.conf`. Use `--trial` to print to stdout instead. If you need to generate this from scratch, refer to `basic.conf`.
* `grep "\{best\}" experiments.conf | cut -d = -f 1 > torun.txt`: This creates a list of configs that can be used by the script to launch jobs. You can use a regexp to restrict the list of configs. For example, `grep "\{best\}" experiments.conf | grep "sl512*" | cut -d = -f 1 > torun.txt` will select configs with `max_segment_len = 512`.
* `python tune.py --data_dir <coref_data_dir> --run_jobs`: This launches jobs from torun.txt on the slurm cluster.


## Citations
If you use the pretrained *BERT*-based coreference model (or this implementation), please cite the paper, [BERT for Coreference Resolution: Baselines and Analysis](https://homes.cs.washington.edu/~mandar90/).
```
@inproceedings{joshi2019coref,
    title={{BERT} for Coreference Resolution: Baselines and Analysis},
    author={Mandar Joshi and Omer Levy and Daniel S. Weld and Luke Zettlemoyer and Omer Levy},
    year={2019},
    booktitle={Empirical Methods in Natural Language Processing (EMNLP)}
}
```

Additionally, if you use the pretrained *SpanBERT* coreference model, please cite the paper, [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529).
```
@article{joshi2019spanbert,
    title={{SpanBERT}: Improving Pre-training by Representing and Predicting Spans},
    author={Mandar Joshi and Danqi Chen and Yinhan Liu and Daniel S. Weld and Luke Zettlemoyer and Omer Levy},
    year={2019},
    journal={arXiv preprint arXiv:1907.10529}
}
```
