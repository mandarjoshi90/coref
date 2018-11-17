from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import operator
import random
import math
import json
import threading
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import h5py
import os
import sys


import util
import coref_ops
import conll
import metrics
sys.path.append(os.path.abspath('../bert'))
import tokenization
import modeling
import optimization

class CorefModel(object):
  def __init__(self, config):
    self.config = config
    self.max_span_width = config["max_span_width"]
    self.genres = { g:i for i,g in enumerate(config["genres"]) }
    self.eval_data = None # Load eval data lazily.
    self.bert_config = modeling.BertConfig.from_json_file(config["bert_config_file"])
    self.tokenizer = tokenization.FullTokenizer(
                vocab_file=config['vocab_file'], do_lower_case=False)

    input_props = []
    input_props.append((tf.int32, [None, None])) # input_ids.
    input_props.append((tf.int32, [None, None])) # input_mask
    input_props.append((tf.int32, [None])) # Text lengths.
    input_props.append((tf.int32, [None])) # Speaker IDs.
    input_props.append((tf.int32, [])) # Genre.
    input_props.append((tf.bool, [])) # Is training.
    input_props.append((tf.int32, [None])) # Gold starts.
    input_props.append((tf.int32, [None])) # Gold ends.
    input_props.append((tf.int32, [None])) # Cluster ids.

    self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
    dtypes, shapes = zip(*input_props)
    queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
    self.enqueue_op = queue.enqueue(self.queue_input_tensors)
    self.input_tensors = queue.dequeue()

    self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
    #self.predictions = self.get_predictions(*self.input_tensors)
    #self.loss, self.predictions = self.get_loss(*self.input_tensors)
    # bert stuff
    tvars = tf.trainable_variables()
    assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, config['init_checkpoint'])
    tf.train.init_from_checkpoint(config['init_checkpoint'], assignment_map)
    print("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      # tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      # init_string)
      print("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    num_train_steps = int(
                    2802 * 100)
    num_warmup_steps = int(num_train_steps * 0.1)
    #self.global_step = tf.train.get_or_create_global_step()
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.train_op = optimization.create_optimizer(
                      self.loss, 5e-5, num_train_steps, num_warmup_steps, False, self.global_step)
    # self.reset_global_step = tf.assign(self.global_step, 0)
    # learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                               # self.config["decay_frequency"], self.config["decay_rate"], staircase=True)
    # trainable_params = tf.trainable_variables()
    # gradients = tf.gradients(self.loss, trainable_params)
    # gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
    # optimizers = {
      # "adam" : tf.train.AdamOptimizer,
      # "sgd" : tf.train.GradientDescentOptimizer
    # }
    # optimizer = optimizers[self.config["optimizer"]](learning_rate)
    # self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)

  def start_enqueue_thread(self, session):
    with open(self.config["train_path"]) as f:
      train_examples = [json.loads(jsonline) for jsonline in f.readlines()]
    def _enqueue_loop():
      while True:
        random.shuffle(train_examples)
        for example in train_examples:
          tensorized_example = self.tensorize_example(example, is_training=True)
          feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
          session.run(self.enqueue_op, feed_dict=feed_dict)
    enqueue_thread = threading.Thread(target=_enqueue_loop)
    enqueue_thread.daemon = True
    enqueue_thread.start()

  def restore(self, session):
    # Don't try to restore unused variables from the TF-Hub ELMo module.
    vars_to_restore = [v for v in tf.global_variables() if "module/" not in v.name]
    saver = tf.train.Saver(vars_to_restore)
    checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
    print("Restoring from {}".format(checkpoint_path))
    session.run(tf.global_variables_initializer())
    saver.restore(session, checkpoint_path)


  def tensorize_mentions(self, mentions):
    if len(mentions) > 0:
      starts, ends = zip(*mentions)
    else:
      starts, ends = [], []
    return np.array(starts), np.array(ends)

  def tensorize_span_labels(self, tuples, label_dict):
    if len(tuples) > 0:
      starts, ends, labels = zip(*tuples)
    else:
      starts, ends, labels = [], [], []
    return np.array(starts), np.array(ends), np.array([label_dict[c] for c in labels])

  def tensorize_example(self, example, is_training):
    clusters = example["clusters"]

    gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
    gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}
    cluster_ids = np.zeros(len(gold_mentions))
    for cluster_id, cluster in enumerate(clusters):
      for mention in cluster:
        cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

    sentences = example["sentences"]
    num_words = sum(len(s) for s in sentences)
    speakers = util.flatten(example["speakers"])

    assert num_words == len(speakers), (num_words, len(speakers))

    max_sentence_length = max(len(s) for s in sentences)
    text_len = np.array([len(s) for s in sentences])

    input_ids, input_mask = [], []
    for i, sentence in enumerate(sentences):
      sent_input_ids = self.tokenizer.convert_tokens_to_ids(sentence)
      sent_input_mask = [1] * len(sent_input_ids)
      while len(sent_input_ids) < max_sentence_length:
          sent_input_ids.append(0)
          sent_input_mask.append(0)
      input_ids.append(sent_input_ids)
      input_mask.append(sent_input_mask)
    input_ids = np.array(input_ids)
    input_mask = np.array(input_mask)
    assert num_words == np.sum(input_mask), (num_words, np.sum(input_mask))

    speaker_dict = { s:i for i,s in enumerate(set(speakers)) }
    speaker_ids = np.array([speaker_dict[s] for s in speakers])

    doc_key = example["doc_key"]
    genre = self.genres[doc_key[:2]]

    gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)
    example_tensors = (input_ids, input_mask,  text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids)

    if is_training and len(sentences) > self.config["max_training_sentences"]:
      return self.truncate_example(*example_tensors)
    else:
      return example_tensors

  def truncate_example(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids):
    max_training_sentences = self.config["max_training_sentences"]
    num_sentences = input_ids.shape[0]
    assert num_sentences > max_training_sentences

    sentence_offset = random.randint(0, num_sentences - max_training_sentences)
    word_offset = text_len[:sentence_offset].sum()
    num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
    input_ids = input_ids[sentence_offset:sentence_offset + max_training_sentences, :]
    input_mask = input_mask[sentence_offset:sentence_offset + max_training_sentences, :]
    text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

    speaker_ids = speaker_ids[word_offset: word_offset + num_words]
    gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
    gold_starts = gold_starts[gold_spans] - word_offset
    gold_ends = gold_ends[gold_spans] - word_offset
    cluster_ids = cluster_ids[gold_spans]

    return input_ids, input_mask, text_len, speaker_ids, genre, is_training,  gold_starts, gold_ends, cluster_ids

  def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
    same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0)) # [num_labeled, num_candidates]
    same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0)) # [num_labeled, num_candidates]
    same_span = tf.logical_and(same_start, same_end) # [num_labeled, num_candidates]
    candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span)) # [1, num_candidates]
    candidate_labels = tf.squeeze(candidate_labels, 0) # [num_candidates]
    return candidate_labels

  def get_dropout(self, dropout_rate, is_training):
    return 1 - (tf.to_float(is_training) * dropout_rate)

  def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c):
    k = util.shape(top_span_emb, 0)
    top_span_range = tf.range(k) # [k]
    antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0) # [k, k]
    antecedents_mask = antecedent_offsets >= 1 # [k, k]
    fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores, 0) # [k, k]
    fast_antecedent_scores += tf.log(tf.to_float(antecedents_mask)) # [k, k]
    fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb) # [k, k]

    _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False) # [k, c]
    top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents) # [k, c]
    top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents) # [k, c]
    top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents) # [k, c]
    return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets



  # def get_predictions(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids):
    # # is_training = tf.Print(is_training, [is_training], 'istraining')
    # predictions, _ = self.get_predictions_and_loss(input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, False)
    # return predictions

  # def get_loss(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids):
    # pred, loss = self.get_predictions_and_loss(input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, True)
    # return loss, pred

  def get_predictions_and_loss(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids):
    # is_training = tf.Print(is_training, [is_training], 'istraining')
    # do_train = tf.cond(tf.equal(is_training, tf.constant(True)), lambda: tf.constant(True), lambda: tf.constant(False))
    # is_training = tf.Print(is_training, [is_training, do_train], 'istraining')
    # input_ids = tf.Print(input_ids, [tf.shape(input_ids)], 'input shape')
    model = modeling.BertModel(
      config=self.bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      # use_tpu is False
      use_one_hot_embeddings=False)
    context_outputs = model.get_sequence_output()
    self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
    # self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
    # self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

    num_sentences = tf.shape(context_outputs)[0]
    max_sentence_length = tf.shape(context_outputs)[1]
    # context_outputs = tf.nn.dropout(context_outputs, self.lexical_dropout) # [num_sentences, max_sentence_length, emb]
    # text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length) # [num_sentence, max_sentence_length]
    context_outputs = self.flatten_emb_by_sentence(context_outputs, input_mask)
    num_words = util.shape(context_outputs, 0)

    #genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len(self.genres), self.config["feature_size"]]), genre) # [emb]

    # mask out cross-sentence candidates
    sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1), [1, max_sentence_length]) # [num_sentences, max_sentence_length]
    flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, input_mask) # [num_words]
    # flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask) # [num_words]

    candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1), [1, self.max_span_width]) # [num_words, max_span_width]
    candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width), 0) # [num_words, max_span_width]
    candidate_start_sentence_indices = tf.gather(flattened_sentence_indices, candidate_starts) # [num_words, max_span_width]
    candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends, num_words - 1)) # [num_words, max_span_width]
    candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices, candidate_end_sentence_indices)) # [num_words, max_span_width]
    flattened_candidate_mask = tf.reshape(candidate_mask, [-1]) # [num_words * max_span_width]
    candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]), flattened_candidate_mask) # [num_candidates]
    candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask) # [num_candidates]
    candidate_sentence_indices = tf.boolean_mask(tf.reshape(candidate_start_sentence_indices, [-1]), flattened_candidate_mask) # [num_candidates]

    candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends, cluster_ids) # [num_candidates]

    # compute span embeddings -- don't need this
    # candidate_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, candidate_starts, candidate_ends) # [num_candidates, emb]
    # conpute mention scores -- change this
    # context_outputs = tf.Print(context_outputs, [tf.shape(context_outputs)], 'context_outputs')
    # candidate_mention_scores =  self.get_mention_scores_old(candidate_span_emb)
    candidate_mention_scores =  self.get_mention_scores(context_outputs, candidate_starts, candidate_ends) # [k, 1]
    # candidate_mention_scores = tf.Print(candidate_mention_scores, [tf.shape(candidate_mention_scores)], 'cand mention scores')
    candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1) # [k]
    #candidate_mention_scores = tf.boolean_mask(candidate_mention_scores, flattened_candidate_mask) # [num_candidates]

    # beam size
    k = tf.to_int32(tf.floor(tf.to_float(tf.shape(context_outputs)[0]) * self.config["top_span_ratio"]))
    # pull from beam
    top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                               tf.expand_dims(candidate_starts, 0),
                                               tf.expand_dims(candidate_ends, 0),
                                               tf.expand_dims(k, 0),
                                               util.shape(context_outputs, 0),
                                               True) # [1, k]
    top_span_indices.set_shape([1, None])
    top_span_indices = tf.squeeze(top_span_indices, 0) # [k]

    top_span_starts = tf.gather(candidate_starts, top_span_indices) # [k]
    top_span_ends = tf.gather(candidate_ends, top_span_indices) # [k]
    # don't need this
    #top_span_emb = tf.gather(candidate_span_emb, top_span_indices) # [k, emb]
    top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices) # [k]
    top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices) # [k]
    top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts) # [k]

    # c = tf.minimum(self.config["max_top_antecedents"], k)

    # if self.config["coarse_to_fine"]:
      # top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_to_fine_pruning(top_span_emb, top_span_mention_scores, c)
    # else:
      # top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.distance_pruning(top_span_emb, top_span_mention_scores, c)

    # antecedent scores -- change this
    dummy_scores = tf.zeros([k, 1]) # [k, 1]
    # top_span_starts = tf.Print(top_span_starts, [top_span_starts], 'top start')
    top_antecedent_scores, top_antecedents, top_antecedents_mask = self.get_antecedent_scores(context_outputs, top_span_starts, top_span_ends, top_span_mention_scores)
    # for i in range(self.config["coref_depth"]):
      # with tf.variable_scope("coref_layer", reuse=(i > 0)):
        # top_antecedent_emb = tf.gather(top_span_emb, top_antecedents) # [k, c, emb]
        # top_antecedent_scores = top_fast_antecedent_scores + self.get_slow_antecedent_scores(top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets, top_span_speaker_ids, genre_emb) # [k, c]
        # top_antecedent_weights = tf.nn.softmax(tf.concat([dummy_scores, top_antecedent_scores], 1)) # [k, c + 1]
        # top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), top_antecedent_emb], 1) # [k, c + 1, emb]
        # attended_span_emb = tf.reduce_sum(tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb, 1) # [k, emb]
        # with tf.variable_scope("f"):
          # f = tf.sigmoid(util.projection(tf.concat([top_span_emb, attended_span_emb], 1), util.shape(top_span_emb, -1))) # [k, emb]
          # top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb # [k, emb]

    top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 1) # [k, c + 1]
    # top_antecedent_scores = tf.Print(top_antecedent_scores, [tf.shape(context_outputs), tf.shape(candidate_ends), top_antecedent_scores, tf.shape(top_antecedent_scores)], 'top_antecedent_scores')

    top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents) # [k, c]
    top_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(top_antecedents_mask))) # [k, c]
    same_cluster_indicator = tf.equal(top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1)) # [k, c]
    non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1) # [k, 1]
    pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator) # [k, c]
    dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True)) # [k, 1]
    top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1) # [k, c + 1]
    loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels) # [k]
    loss = tf.reduce_sum(loss) # []

    return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores], loss


  def get_mention_scores_old(self, span_emb):
      with tf.variable_scope("mention_scores", reuse=tf.AUTO_REUSE):
        return util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, 1]

  def get_mention_scores(self, encoded_doc, span_starts, span_ends):
      num_words = util.shape(encoded_doc, 0) # T
      # span_starts = tf.Print(span_starts, [tf.shape(span_starts)], 'span_starts')
      with tf.variable_scope("start_scores", reuse=tf.AUTO_REUSE):
        start_scores =  tf.squeeze(util.ffnn(encoded_doc, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout), 1) # [T]
      with tf.variable_scope("end_scores", reuse=tf.AUTO_REUSE):
        end_scores =  tf.squeeze(util.ffnn(encoded_doc, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout), 1) # [T]
      start_end_scores = tf.tile(tf.expand_dims(start_scores, 1), [1, num_words]) + tf.tile(tf.expand_dims(end_scores, 0), [num_words, 1]) # [T, T]
      # start_end_scores = tf.Print(start_end_scores, [tf.shape(start_end_scores)], 'start_end')
      # span_start_doc_scores = tf.gather(start_end_scores, tf.tile(tf.expand_dims(span_starts, 1), [1, num_words])) #[NC, T]
      # span_scores = tf.gather(span_start_doc_scores, tf.expand_dims(span_ends, 1), axis=1) # [NC, 1]`
      span_start_doc_scores = tf.gather(start_end_scores, span_starts) #[NC, T]
      # span_start_doc_scores = tf.Print(span_start_doc_scores, [tf.shape(span_start_doc_scores)], 'start_start_doc')
      span_scores = util.batch_gather(span_start_doc_scores, tf.expand_dims(span_ends, 1)) # [NC, 1]`
      span_width = 1 + span_ends - span_starts # [NC]
      if self.config["use_features"]:
        span_width_index = span_width - 1 # [k]
        #span_width_emb = tf.gather(tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]), span_width_index) # [k, emb]
        span_width_emb = tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]) # [W, emb]
        span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
        with tf.variable_scope("width_scores", reuse=tf.AUTO_REUSE):
          width_scores =  util.ffnn(span_width_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [W, 1]
        width_scores = tf.gather(width_scores, span_width_index)
        span_scores += width_scores
      # span_scores = tf.Print(span_scores, [tf.shape(span_scores)], 'span_scores')
      return span_scores

  def get_antecedent_scores(self, encoded_doc, span_starts, span_ends, candidate_mention_scores):
      num_words = util.shape(encoded_doc, 0) # T
      num_c = util.shape(span_starts, 0) # NC
      antecedents_mask, antecedent_offsets = self.get_antecedent_mask(num_c)
      encoded_doc_end = encoded_doc_start = encoded_doc
      # with tf.variable_scope("start_ffnn", reuse=tf.AUTO_REUSE):
        # encoded_doc_start =  util.ffnn(encoded_doc, self.config["ffnn_depth"], self.config["ffnn_size"], self.config["ffnn_size"] , self.dropout)# [T]
      # with tf.variable_scope("end_ffnn", reuse=tf.AUTO_REUSE):
        # encoded_doc_end =  util.ffnn(encoded_doc, self.config["ffnn_depth"], self.config["ffnn_size"], self.config["ffnn_size"], self.dropout) # [T]
      ac_scores = self.gather_twice(self.get_bilinear_scores_xWy(encoded_doc_start, 'W_ac', encoded_doc_start), span_starts, tf.tile(tf.expand_dims(span_starts, 0), [num_c, 1]))
      ad_scores = self.gather_twice(self.get_bilinear_scores_xWy(encoded_doc_start, 'W_ad', encoded_doc_end), span_starts,  tf.tile(tf.expand_dims(span_ends, 0), [num_c, 1]))
      bc_scores = self.gather_twice(self.get_bilinear_scores_xWy(encoded_doc_end, 'W_bc', encoded_doc_start), span_ends,  tf.tile(tf.expand_dims(span_starts, 0), [num_c, 1]))
      bd_scores = self.gather_twice(self.get_bilinear_scores_xWy(encoded_doc_end, 'W_bd', encoded_doc_end), span_ends,  tf.tile(tf.expand_dims(span_ends, 0), [num_c, 1]))
      # top_antecedent_scores =  tf.tile(tf.expand_dims(candidate_mention_scores, 1), [1, num_c]) + tf.tile(tf.expand_dims(candidate_mention_scores, 0), [num_c, 1])  
      top_antecedent_scores =  tf.expand_dims(candidate_mention_scores, 1) + tf.expand_dims(candidate_mention_scores, 0) # [k, k]
      top_antecedent_scores  += tf.log(tf.to_float(antecedents_mask)) # [k, k]
      top_antecedent_scores +=  ac_scores + ad_scores + bc_scores + bd_scores#[nc, nc]
      _, top_antecedents = tf.nn.top_k(top_antecedent_scores, num_c, sorted=False) # [nc, c]
      #k = util.shape(top_antecedent_scores, 0)
      top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents) # [nc, c]
      top_antecedent_scores = util.batch_gather(top_antecedent_scores, top_antecedents) # [k, c]
      # print( top_antecedents.get_shape() , top_antecedent_scores.get_shape()) 
      #return top_antecedent_scores, antecedent_offsets, antecedents_mask
      return top_antecedent_scores, top_antecedents, top_antecedents_mask

  def get_antecedent_mask(self, k):
    top_span_range = tf.range(k) # [k]
    antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0) # [k, k]
    antecedents_mask = antecedent_offsets >= 1 # [k, k]
    return antecedents_mask, antecedent_offsets

  def gather_twice(self, params, indices1, indices2):
      # print('twice', params.get_shape(), indices1.get_shape(), indices2.get_shape())
      gather1_op = tf.gather(params, indices1, axis=0)
      # gather1_op = tf.Print(gather1_op, [tf.shape(gather1_op)], 'gather1')
      output = util.batch_gather(gather1_op, indices2)
      # output = tf.Print(output, [tf.shape(output)], 'gather2')
      return output

  def get_bilinear_scores_xWy(self, x, scope_W, y):
      with tf.variable_scope(scope_W, reuse=tf.AUTO_REUSE):
        xW = tf.nn.dropout(util.projection(x, util.shape(x, -1)), self.dropout) # [k, emb]
        y = tf.nn.dropout(y, self.dropout) # [k, emb]
        output = tf.matmul(xW, y, transpose_b=True) # [k, k]
        # print(output.get_shape())
        return output


  def softmax_loss(self, antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels)) # [k, max_ant + 1]
    marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1]) # [k]
    log_norm = tf.reduce_logsumexp(antecedent_scores, [1]) # [k]
    return log_norm - marginalized_gold_scores # [k]

  def bucket_distance(self, distances):
    """
    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
    logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances))/math.log(2))) + 3
    use_identity = tf.to_int32(distances <= 4)
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return tf.clip_by_value(combined_idx, 0, 9)

  def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets, top_span_speaker_ids, genre_emb):
    k = util.shape(top_span_emb, 0)
    c = util.shape(top_antecedents, 1)

    feature_emb_list = []

    if self.config["use_metadata"]:
      top_antecedent_speaker_ids = tf.gather(top_span_speaker_ids, top_antecedents) # [k, c]
      same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), top_antecedent_speaker_ids) # [k, c]
      speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]), tf.to_int32(same_speaker)) # [k, c, emb]
      feature_emb_list.append(speaker_pair_emb)

      tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, c, 1]) # [k, c, emb]
      feature_emb_list.append(tiled_genre_emb)

    if self.config["use_features"]:
      antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets) # [k, c]
      antecedent_distance_emb = tf.gather(tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]]), antecedent_distance_buckets) # [k, c]
      feature_emb_list.append(antecedent_distance_emb)

    feature_emb = tf.concat(feature_emb_list, 2) # [k, c, emb]
    feature_emb = tf.nn.dropout(feature_emb, self.dropout) # [k, c, emb]

    target_emb = tf.expand_dims(top_span_emb, 1) # [k, 1, emb]
    similarity_emb = top_antecedent_emb * target_emb # [k, c, emb]
    target_emb = tf.tile(target_emb, [1, c, 1]) # [k, c, emb]

    pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2) # [k, c, emb]

    with tf.variable_scope("slow_antecedent_scores"):
      slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, c, 1]
    slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2) # [k, c]
    return slow_antecedent_scores # [k, c]

  def get_fast_antecedent_scores(self, top_span_emb):
    with tf.variable_scope("src_projection"):
      source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)), self.dropout) # [k, emb]
    target_top_span_emb = tf.nn.dropout(top_span_emb, self.dropout) # [k, emb]
    return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True) # [k, k]

  def flatten_emb_by_sentence(self, emb, text_len_mask):
    num_sentences = tf.shape(emb)[0]
    max_sentence_length = tf.shape(emb)[1]

    emb_rank = len(emb.get_shape())
    if emb_rank  == 2:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
    elif emb_rank == 3:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
    else:
      raise ValueError("Unsupported rank: {}".format(emb_rank))
    return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))


  def get_predicted_antecedents(self, antecedents, antecedent_scores):
    predicted_antecedents = []
    for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
      if index < 0:
        predicted_antecedents.append(-1)
      else:
        predicted_antecedents.append(antecedents[i, index])
    return predicted_antecedents

  def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
      if predicted_index < 0:
        continue
      assert i > predicted_index, (i, predicted_index)
      predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
      if predicted_antecedent in mention_to_predicted:
        predicted_cluster = mention_to_predicted[predicted_antecedent]
      else:
        predicted_cluster = len(predicted_clusters)
        predicted_clusters.append([predicted_antecedent])
        mention_to_predicted[predicted_antecedent] = predicted_cluster

      mention = (int(top_span_starts[i]), int(top_span_ends[i]))
      predicted_clusters[predicted_cluster].append(mention)
      mention_to_predicted[mention] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }

    return predicted_clusters, mention_to_predicted

  def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator):
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
    mention_to_gold = {}
    for gc in gold_clusters:
      for mention in gc:
        mention_to_gold[mention] = gc

    predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
    return predicted_clusters

  def load_eval_data(self):
    if self.eval_data is None:
      def load_line(line):
        example = json.loads(line)
        return self.tensorize_example(example, is_training=False), example
      with open(self.config["eval_path"]) as f:
        self.eval_data = [load_line(l) for l in f.readlines()]
      num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
      print("Loaded {} eval examples.".format(len(self.eval_data)))

  def evaluate(self, session, official_stdout=False):
    self.load_eval_data()

    coref_predictions = {}
    coref_evaluator = metrics.CorefEvaluator()
    losses = []

    for example_num, (tensorized_example, example) in enumerate(self.eval_data):
      _, _, _, _, _, _, gold_starts, gold_ends, _ = tensorized_example
      feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}
      candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(self.predictions, feed_dict=feed_dict)
      losses.append(session.run(self.loss, feed_dict=feed_dict))
      predicted_antecedents = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
      coref_predictions[example["doc_key"]] = self.evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, example["clusters"], coref_evaluator)
      if example_num % 10 == 0:
        print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

    summary_dict = {}
    print(sum(losses) / len(losses))
    # conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, official_stdout)
    # average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
    # summary_dict["Average F1 (conll)"] = average_f1
    # print("Average F1 (conll): {:.2f}%".format(average_f1))

    p,r,f = coref_evaluator.get_prf()
    summary_dict["Average F1 (py)"] = f
    print("Average F1 (py): {:.2f}%".format(f * 100))
    summary_dict["Average precision (py)"] = p
    print("Average precision (py): {:.2f}%".format(p * 100))
    summary_dict["Average recall (py)"] = r
    print("Average recall (py): {:.2f}%".format(r * 100))

    return util.make_summary(summary_dict), 0 #average_f1
