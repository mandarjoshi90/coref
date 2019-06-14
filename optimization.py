import tensorflow as tf
from bert.optimization import AdamWeightDecayOptimizer

def create_custom_optimizer(tvars, loss, bert_init_lr, task_init_lr, num_train_steps, num_warmup_steps, use_tpu, global_step=None, freeze=-1, task_opt='adam', eps=1e-6):
  """Creates an optimizer training op."""
  if global_step is None:
    global_step = tf.train.get_or_create_global_step()

  bert_learning_rate = tf.constant(value=bert_init_lr, shape=[], dtype=tf.float32)
  task_learning_rate = tf.constant(value=task_init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  bert_learning_rate = tf.train.polynomial_decay(
      bert_learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)
  task_learning_rate = tf.train.polynomial_decay(
      task_learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    bert_warmup_learning_rate = bert_init_lr * warmup_percent_done
    task_warmup_learning_rate = task_init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    bert_learning_rate = (
        (1.0 - is_warmup) * bert_learning_rate + is_warmup * bert_warmup_learning_rate)

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  bert_optimizer = AdamWeightDecayOptimizer(
      learning_rate=bert_learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=eps,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  if task_opt == 'adam_weight_decay':
    task_optimizer = AdamWeightDecayOptimizer(
          learning_rate=task_learning_rate,
          weight_decay_rate=0.01,
          beta_1=0.9,
          beta_2=0.999,
          epsilon=eps
    )
  elif task_opt == 'adam':
    task_optimizer = tf.train.AdamOptimizer(
      learning_rate=task_learning_rate)
  else:
    raise NotImplementedError('Check optimizer. {} is invalid.'.format(task_opt))

  # tvars = tf.trainable_variables()
  bert_vars, task_vars = [], []
  for var in tvars:
    if var.name.startswith('bert'):
        can_optimize = False
        if var.name.startswith('bert/encoder/layer_') and  int(var.name.split('/')[2][len('layer_'):]) >= freeze:
          can_optimize = True
        if freeze == -1 or can_optimize:
          bert_vars.append(var)
    else:
        task_vars.append(var)
  print('bert:task', len(bert_vars), len(task_vars))
  grads = tf.gradients(loss, bert_vars + task_vars)
  bert_grads = grads[:len(bert_vars)]
  task_grads = grads[len(bert_vars):]

  # This is how the model was pre-trained.
  (bert_grads, _) = tf.clip_by_global_norm(bert_grads, clip_norm=1.0)
  (task_grads, _) = tf.clip_by_global_norm(task_grads, clip_norm=1.0)

  # global_step1 = tf.Print(global_step, [global_step], 'before')
  bert_train_op = bert_optimizer.apply_gradients(
      zip(bert_grads, bert_vars), global_step=global_step)
  task_train_op = task_optimizer.apply_gradients(
      zip(task_grads, task_vars), global_step=global_step)
  if task_opt == 'adam_weight_decay':
    new_global_step = global_step + 1
    train_op = tf.group(bert_train_op, task_train_op, [global_step.assign(new_global_step)])
  else:
    train_op = tf.group(bert_train_op, task_train_op)
  return train_op
