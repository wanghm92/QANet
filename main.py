import tensorflow as tf
import ujson as json
import numpy as np
from tqdm import tqdm
import os, sys

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''

from model import Model
from demo import Demo
from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset

def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    '''
        Iterator: Represents the state of iterating through a Dataset.
        
        from_string_handle(): https://www.tensorflow.org/api_docs/python/tf/data/Iterator
            This method allows you to define a "feedable" iterator where you can choose between concrete iterators 
            by feeding a value in a tf.Session.run call. In that case, string_handle would a tf.placeholder, 
            and you would feed it with the value of tf.data.Iterator.string_handle in each step.
        
        make_one_shot_iterator():Creates an Iterator for enumerating the elements of this dataset. 
            The returned iterator will be initialized automatically. 
            A "one-shot" iterator does not currently support re-initialization.
    '''

    dev_total = meta["total"]
    print("Building model...")
    parser = get_record_parser(config)
    graph = tf.Graph()
    with graph.as_default() as g:
        train_dataset = get_batch_dataset(config.train_record_file, parser, config)
        dev_dataset = get_dataset(config.dev_record_file, parser, config)
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

        model = Model(config, iterator, word_mat, char_mat, graph = g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        loss_save = 100.0
        patience = 0
        best_em = 0.0
        best_f1 = 0.0

        with tf.Session(config=sess_config) as sess:

            sess.run(tf.global_variables_initializer())

            train_handle = sess.run(train_iterator.string_handle())
            dev_handle = sess.run(dev_iterator.string_handle())

            saver = tf.train.Saver(max_to_keep=100)
            if os.path.exists(os.path.join(config.save_dir, "checkpoint")):
                saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))

            writer = tf.summary.FileWriter(config.log_dir)

            global_step = max(sess.run(model.global_step), 1)
            for _ in tqdm(range(global_step, config.num_steps + 1)):
                global_step = sess.run(model.global_step) + 1

                '''
                enc, start_logits, end_logits, logits1, logits2, outer, yp1, yp2, losses, losses2, loss \
                    = sess.run(model.debug_ops,feed_dict={handle: train_handle, model.dropout: config.dropout})
                print(outer)
                print(yp1)
                print(yp2)
                for i in [enc, start_logits, end_logits, logits1, logits2, outer, yp1, yp2, losses, losses2, loss]:
                    print(i.shape)
                '''

                loss, train_op = sess.run([model.loss, model.train_op],
                                          feed_dict={handle: train_handle, model.dropout: config.dropout})
                if global_step % config.period == 0:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, global_step)
                if global_step % config.checkpoint == 0:
                    _, summ = evaluate_batch(model, config.val_num_batches, train_eval_file, sess,
                                             "train", handle, train_handle)
                    for s in summ:
                        writer.add_summary(s, global_step)

                    metrics, summ = evaluate_batch(model, dev_total // config.batch_size + 1,
                                                   dev_eval_file, sess, "dev", handle, dev_handle)

                    # optimization from jasonwbw
                    # early stop
                    dev_f1 = metrics["f1"]
                    dev_em = metrics["exact_match"]
                    if dev_f1 < best_f1 and dev_em < best_em:
                        patience += 1
                        if patience > config.early_stop:
                            print('>>>>>> WARNING !!! <<<<<<< Early_stop reached!!!')
                    # save best model
                    else:
                        patience = 0
                        if dev_em >= best_em:
                            best_em = dev_em
                            filename = os.path.join(config.save_dir, "model_{}.bestem".format(global_step))
                            saver.save(sess, filename)
                            if dev_f1 >= best_f1:
                                best_f1 = dev_f1
                                filename = os.path.join(config.save_dir, "model_{}.bestf1".format(global_step))
                                saver.save(sess, filename)

                    for s in summ:
                        writer.add_summary(s, global_step)
                    writer.flush()
                    filename = os.path.join(config.save_dir, "model_{}.ckpt".format(global_step))
                    saver.save(sess, filename)
                    # save best dev model



def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    answer_dict = {}
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, yp1, yp2, = sess.run([model.qa_id, model.loss, model.yp1, model.yp2],
                                          feed_dict={handle: str_handle})
        answer_dict_, _ = convert_tokens(eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
    return metrics, [loss_sum, f1_sum, em_sum]


def demo(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)

    model = Model(config, None, word_mat, char_mat, trainable=False, demo = True)
    demo = Demo(model, config)


def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)

    total = meta["total"]

    graph = tf.Graph()
    with graph.as_default() as g:
        test_batch = get_dataset(config.test_record_file, get_record_parser(config, is_test=True),
                                 config).make_one_shot_iterator()

        model = Model(config, test_batch, word_mat, char_mat, trainable=False, graph = g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # model_path = os.path.join(config.save_dir, 'model_79000.bestckpt')
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            print("Model Loaded from --> {}".format(tf.train.latest_checkpoint(config.save_dir)))
            # saver.restore(sess, model_path)
            # print("Model Loaded from --> {}".format(model_path))
            if config.decay < 1.0:
                sess.run(model.assign_vars)
            losses = []
            answer_dict = {}
            remapped_dict = {}
            for step in tqdm(range(total // config.batch_size + 1)):
                qa_id, loss, yp1, yp2 = sess.run(
                    [model.qa_id, model.loss, model.yp1, model.yp2])
                answer_dict_, remapped_dict_ = convert_tokens(
                    eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
                answer_dict.update(answer_dict_)
                remapped_dict.update(remapped_dict_)
                losses.append(loss)
            loss = np.mean(losses)
            metrics = evaluate(eval_file, answer_dict)
            with open(config.answer_file, "w") as fh:
                json.dump(remapped_dict, fh)
            print("Exact Match: {}, F1: {}".format(
                metrics['exact_match'], metrics['f1']))
