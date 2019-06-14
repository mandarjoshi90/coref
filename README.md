# BERT for Coreference Resolution: Baselines and Analysis


## Introduction
This repository contains the code for replicating results from

* [BERT for Coreference Resolution: Baselines and Analysis](!!!!!!!!!!!!!Insert Link!!!!!!!!) 
* OpenReview Anonymous Prepreint
We apply BERT to coreference resolution, achieving a new state of the art on the GAP (+11.5 F1) and OntoNotes (+3.9 F1) benchmarks.

## Getting Started
* Install python (either 2 or 3) requirements: `pip install -r requirements.txt`
  * There are 3 platform-dependent ways to build custom TensorFlow kernels. Please comment/uncomment the appropriate lines in the script.
* Preprocessing
  * `python minimize.py <bert_vocab_file> <ontonotes_data_dir> <output_dir> <do_lower_case>`: Ensure that `<do_lower_case>` is `true` for uncased models and `false` for cased models. The `<output_dir>` should contain `*.english.v4_gold_conll` files. See the [e2e-coref](https://github.com/kentonl/e2e-coref/tree/e2e) for further details.

## Training Instructions

* Experiment configurations are found in `experiments.conf`
* Choose an experiment that you would like to run, e.g. `best`
* Training: `GPU=0 python train.py <experiment>`
* Results are stored in the `logs` directory and can be viewed via TensorBoard.
* Evaluation: `python evaluate.py <experiment>`


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
* Run `python predict.py <experiment> <input_file> <output_file>`, which outputs the input jsonlines with predicted clusters.

## Tune Hyperparameters
* `python tune_models.py configs`: This generates multiple configs for tuning (BERT and task) learning rates, embedding models, and `max_segment_len`. This modifies `experiments.conf`. Use `--trial` to print to stdout instead.
* `grep "\{best\}" experiments.conf | cut -d = -f 1 > torun.txt`: This creates a list of configs that can be used by the script to launch jobs. You can use a reg exp to restrict the list of configs. For example, `grep "\{best\}" experiments.conf | grep "*sl512*" | cut -d = -f 1 > torun.txt` will select configs with `max_segment_len = 512`.
* `python tune_models.py run`: This launches jobs from torun.txt on the slurm cluster.

## Important Hyperpameters
* `bert_learning_rate`: The learning rate for the BERT parameters. Typically, `1e-5` and `2e-5` work well.
* `task_learning_rate`: The learning rate for the other parameters. Typically, LRs between `0.0001` to `0.0003` work well.
* `init_checkpoint`: The checkpoint file from which BERT parameters are initialized. Both TF and Pytorch checkpoints work as long as they use the same BERT architecture. Use `*ckpt` files for TF and `*pt` for Pytorch.
* `max_segment_len`: The maximum size of the BERT segment. 
