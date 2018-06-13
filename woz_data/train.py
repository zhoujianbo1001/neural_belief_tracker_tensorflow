import tensorflow as tf
import random

from woz_data import utils
from neural_belief_tracker import nbt_cell

FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("num_filters", 300, "the number of cnn filters.")

# Optimizer parameters
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10,
                      "Epsilon used for RMSProp optimizer.")

# Task parameters
tf.flags.DEFINE_integer("batch_size", 512, "Bach size for training.")

# Training options.
tf.flags.DEFINE_integer("max_epoch", 4, "The max epoch.")


def train_each_slot(batch_size=FLAGS.batch_size,
                    target_slot="food",
                    label_count=92,
                    longest_utterance_length=40,
                    vector_dimension=300):
    """Reading data."""
    data_set = utils.read_data(target_slot)

    """Setting inputs."""
    user_utterance_full = tf.placeholder(
        dtype=tf.float32,
        shape=[batch_size, longest_utterance_length, vector_dimension]
    )
    slot_vectors = tf.placeholder(
        dtype=tf.float32,
        shape=[label_count+1, vector_dimension]
    )
    value_vectors = tf.placeholder(
        dtype=tf.float32,
        shape=[label_count+1, vector_dimension]
    )
    system_act_request_slots = tf.placeholder(
        dtype=tf.float32,
        shape=[batch_size, vector_dimension]
    )
    system_act_confirm_slots = tf.placeholder(
        dtype=tf.float32,
        shape=[batch_size, vector_dimension]
    )
    system_act_confirm_values = tf.placeholder(
        dtype=tf.float32,
        shape=[batch_size, vector_dimension]
    )
    y_past_state = tf.placeholder(
        dtype=tf.float32,
        shape=[batch_size, label_count+1]
    )

    """Setting outputs."""
    y_state = tf.placeholder(
        dtype=tf.float32,
        shape=[batch_size, label_count+1]
    )

    """Setting neural belief tracker"""
    nbt = nbt_cell.NbtCell(
        batch_size=FLAGS.batch_size,
        vector_dimension=vector_dimension,
        label_count=label_count,
        longest_utterance_length=longest_utterance_length
    )

    y_combine, logits = nbt.build(
        user_utterance_full=user_utterance_full,
        slot_vectors=slot_vectors,
        value_vectors=value_vectors,
        system_act_request_slots=system_act_request_slots,
        system_act_confirm_slots=system_act_confirm_slots,
        system_act_confirm_values=system_act_confirm_values,
        y_past_state=y_past_state
    )

    """Setting loss function."""
    train_loss = tf.contrib.losses.softmax_cross_entropy(
        logits=y_combine,
        onehot_labels=y_state
    )

    """Setting optimizer"""
    trainable_variables = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(train_loss, trainable_variables),
        FLAGS.max_grad_norm
    )

    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP]
    )

    optimizer = tf.train.RMSPropOptimizer(
        FLAGS.learning_rate, epsilon=FLAGS.optimizer_epsilon
    )
    train_step = optimizer.apply_gradients(
        grads_and_vars=zip(grads, trainable_variables),
        global_step=global_step
    )

    """Setting saver"""
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        epoch = 0
        while epoch < FLAGS.max_epoch:
            epoch += 1

            # Reading data.
            random.shuffle(data_set)

            # Batch training.
            batch_interation = 0
            while 1:
                batch_data, index = utils.read_batch_data(
                    pos=batch_interation*batch_size,
                    batch_size=batch_size,
                    data_set=data_set
                )
                if index != batch_size:
                    print("report: %d", epoch)
                    break

                batch_interation += 1

                # Run the session.
                sess.run(fetches=[train_step, logits],
                         feed_dict={
                             user_utterance_full: batch_data["user_utterance_full_batch"],
                             slot_vectors: batch_data["slot_vectors_batch"],
                             value_vectors: batch_data["value_vectors_batch"],
                             system_act_request_slots: batch_data["system_act_request_slots_batch"],
                             system_act_confirm_slots: batch_data["system_act_confirm_slots_batch"],
                             system_act_confirm_values: batch_data["system_act_confirm_values_batch"],
                             y_past_state: batch_data["y_past_state_batch"],
                             y_state: batch_data["y_state_batch"]
                         })

        # Saving model.
        saver.save(sess=sess,
                   save_path="./checkpoint/"+target_slot+".ckpt")


def main():
    train_each_slot()


if __name__ == "__main__":
    main()
