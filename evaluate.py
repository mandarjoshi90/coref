#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
# import coref_model as cm
import truncated_coref_model  as cm
# import overlap as cm
import util

def read_doc_keys(fname):
    keys = set()
    with open(fname) as f:
        for line in f:
            keys.add(line.strip())
    return keys

if __name__ == "__main__":
  config = util.initialize_from_env()
  model = cm.CorefModel(config)
  saver = tf.train.Saver()
  log_dir = config["log_dir"]
  keys = read_doc_keys('../kenton-coref-elmo-2018/doc_keys_512.txt')
  with tf.Session() as session:
    # ckpt = tf.train.get_checkpoint_state(log_dir, latest_filename='model.max.ckpt')
    # print(ckpt.model_checkpoint_path)
    # if ckpt and ckpt.model_checkpoint_path:
      # print("Restoring from: {}".format(ckpt.model_checkpoint_path))
      # saver.restore(session, ckpt.model_checkpoint_path)
    model.restore(session)
    model.evaluate(session, official_stdout=True)
