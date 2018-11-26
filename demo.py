from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import input
import tensorflow as tf
import coref_model as cm
import util

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize
import sys
import os
sys.path.append(os.path.abspath('../bert'))
import tokenization

tokenizer = tokenization.FullTokenizer(
                vocab_file='../bert/cased_L-12_H-768_A-12/vocab.txt', do_lower_case=False)
def create_example(text):
  #raw_sentences = sent_tokenize(text)
  sentences = [['[CLS]'] + tokenizer.tokenize(text) + ['[SEP]']]
  sentence_map = [0] * len(sentences[0])
  speakers = [["" for _ in sentence] for sentence in sentences]
  return {
    "doc_key": "nw",
    "clusters": [],
    "sentences": sentences,
    "speakers": speakers,
    'sentence_map': sentence_map
  }

def print_predictions(example):
  words = util.flatten(example["sentences"])
  for cluster in example["predicted_clusters"]:
    print(u"Predicted cluster: {}".format([" ".join(words[m[0]:m[1]+1]) for m in cluster]))

def make_predictions(text, model):
  example = create_example(text)
  tensorized_example = model.tensorize_example(example, is_training=False)
  feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
  # print(feed_dict)
  mention_starts, mention_ends, candidate_mention_scores, top_span_starts, top_span_ends, antecedents, antecedent_scores  = session.run(model.predictions, feed_dict=feed_dict)

  predicted_antecedents = model.get_predicted_antecedents(antecedents, antecedent_scores)

  example["predicted_clusters"], _ = model.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)
  example["top_spans"] = zip((int(i) for i in mention_starts), (int(i) for i in mention_ends))
  return example

if __name__ == "__main__":
  config = util.initialize_from_env()
  log_dir = config["log_dir"]
  model = cm.CorefModel(config)
  saver = tf.train.Saver()
  with tf.Session() as session:
    # model.restore(session)
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print("Restoring from: {}".format(ckpt.model_checkpoint_path))
      saver.restore(session, ckpt.model_checkpoint_path)

    while True:
      text = input("Document text: ")
      print_predictions(make_predictions(text, model))
