import tensorflow as tf
import numpy as np


class NbtCell():
    def __init__(self,
                 batch_size,
                 vector_dimension,
                 label_count,
                 num_filters=300,
                 longest_utterance_length=40,
                 name="nbtCell"):
        self.name = name
        self.batch_size = batch_size
        self.vector_dimension = vector_dimension
        self.label_count = label_count
        self.num_filters = num_filters
        self.longest_utterance_length = longest_utterance_length

    def build(self,
               user_utterance_full,
               slot_vectors,
               value_vectors,
               system_act_request_slots,
               system_act_confirm_slots,
               system_act_confirm_values,
               y_past_state):
        """
        :param user_utterance_full:
                the utterance for user, whose shape is
                [batch_size, longest_utterance_length, vector_dimension]
                and type is tf.float32.
        :param slot_vectors:
                the slot for candidate slot-value pair, whose shape is
                [label_count + 1, vector_dimension]
                and type is tf.float32
        :param value_vectors:
                the value for candidate slot-value pair, whose shape is
                [label_count + 1, vector_dimension]
                and type is tf.float32
        :param system_act_request_slots:
                the request slots for system act, whose shape is
                [batch_size, vector_dimension]
                and type is tf.float32
        :param system_act_confirm_slots:
                the confirm slots for system act, whose shape is
                [batch_size, vector_dimension]
                and type is tf.float32
        :param system_act_confirm_values:
                the confirm value for system act, whose shape is
                [batch_size, vector_dimension]
                and type is tf.float32
        :param y_past_state:
                the past user-goal state, whose shape is
                [batch_size, label_count + 1]
                and the number is 0 or 1.
        :return:
            y_combine: the predicted user-goal state which don't use softmax,
                    whose shape is [batch_size, label_count+1]
            y: the predicted user-goal state, whose shape is
                [batch_size, label_count+1]
        """
        # Network parameters.
        hidden_units_1 = 100
        keep_prob = 0.6

        # Getting user utterance representation.
        with tf.variable_scope("utterance_representation"):
            utterance_representation = self._CNN_model(user_utterance_full)

        # Getting candidate slot-value pair representation.
        with tf.variable_scope("candidate_slot_value_representation"):
            weight_candidates = tf.Variable(
                initial_value=tf.random_normal([self.vector_dimension, self.vector_dimension]))
            bias_candidates = tf.Variable(
                initial_value=tf.zeros([self.vector_dimension]))
            candidate_slot_value_pair = slot_vectors + value_vectors
            candidate_slot_value_pair_representation = tf.nn.sigmoid(
                tf.matmul(candidate_slot_value_pair, weight_candidates) + bias_candidates)

        # Computing the contributions to user utterance which slot-value pair make.
        with tf.variable_scope("user_utterance_and_slot_value_interaction"):
            list_of_candidate_contributions = []
            for index in range(0, self.label_count):
                list_of_candidate_contributions.append(
                    tf.multiply(utterance_representation, candidate_slot_value_pair_representation[index, :])
                )
            user_utterance_and_slot_value_interaction = tf.reshape(
                tf.transpose(tf.stack(list_of_candidate_contributions),
                            [1, 0, 2]),
                [-1, self.vector_dimension]
            )

        # Semantic Decoding
        with tf.variable_scope("Semantic_Decoding_Network"):
            # hidden layer
            utterance_weight_hidden_layer = tf.Variable(
                tf.random_normal([self.vector_dimension, hidden_units_1])
            )
            utterance_bias_hidden_layer = tf.Variable(
                tf.zeros([hidden_units_1])
            )
            utterance_hidden_layer = tf.nn.sigmoid(
                tf.reshape(
                    tf.matmul(
                        user_utterance_and_slot_value_interaction,
                        utterance_weight_hidden_layer
                    ) + utterance_bias_hidden_layer,
                    [-1, self.label_count, hidden_units_1]
                )
            )
            utterance_hidden_layer_with_dropout = tf.nn.dropout(
                utterance_hidden_layer, keep_prob
            )

            # output layer
            utterance_weight_presoftmax = tf.Variable(
                tf.random_normal([hidden_units_1, 1])
            )
            utterance_bias_presoftmax = tf.Variable(
                tf.zeros([1])
            )
            utterance_presoftmax = tf.reshape(
                tf.matmul(
                    tf.reshape(utterance_hidden_layer_with_dropout, [-1, hidden_units_1]),
                    utterance_weight_presoftmax
                ) + utterance_bias_presoftmax,
                [-1, self.label_count]
            )

        # Interaction network for system requests.
        with tf.variable_scope("network_for_sys_requests"):
            list_of_sys_request_contributions = []
            sys_request_candidate_interaction = tf.multiply(slot_vectors[0, :], system_act_request_slots)
            dot_product_sys_request = tf.reduce_mean(sys_request_candidate_interaction, 1)

            decision = tf.multiply(tf.expand_dims(dot_product_sys_request, 1), utterance_representation)

            sysreq_weight_hidden_layer = tf.Variable(
                tf.random_normal([self.vector_dimension, hidden_units_1])
            )
            sysreq_bias_hidden_layer = tf.Variable(tf.zeros([1]))

            for value_index in range(0, self.label_count):
                sysreq_hidden_layer_1 = tf.nn.sigmoid(
                    tf.matmul(decision, sysreq_weight_hidden_layer) + sysreq_bias_hidden_layer)
                sysreq_hidden_layer_1_with_dropout = tf.nn.dropout(sysreq_hidden_layer_1, keep_prob)

                sysreq_weight_softmax = tf.Variable(
                    tf.random_normal([hidden_units_1, 1]))
                sysreq_bias_softmax = tf.Variable(tf.zeros([1]))

                sysreq_contribution = tf.matmul(
                    sysreq_hidden_layer_1_with_dropout,
                    sysreq_weight_softmax) + sysreq_bias_softmax

                list_of_sys_request_contributions.append(sysreq_contribution)

            sysreq = tf.concat(list_of_sys_request_contributions, 1)

        # Interaction network for system confirmations.
        with tf.variable_scope("network_for_sys_confirmations"):
            list_of_sys_confirmations_contributions = []
            confirm_w1_hidden_layer = tf.Variable(tf.random_normal([self.vector_dimension, hidden_units_1]))
            confirm_b1_hidden_layer = tf.Variable(tf.zeros([hidden_units_1]))

            confirm_w1_softmax = tf.Variable(tf.random_normal([hidden_units_1, 1]))
            confirm_b1_softmax = tf.Variable(tf.zeros([1]))

            for value_idx in range(0, self.label_count):
                dot_product = tf.multiply(tf.reduce_mean(tf.multiply(slot_vectors[0, :], system_act_confirm_slots), 1),
                                          tf.reduce_mean(tf.multiply(value_vectors[value_idx, :],
                                                                     system_act_confirm_values),
                                                         1)
                                          )  # dot product: slot equality and value equality

                full_ones = tf.ones(tf.shape(dot_product))
                dot_product = tf.cast(tf.equal(dot_product, full_ones), "float32")

                decision = tf.multiply(tf.expand_dims(dot_product, 1), utterance_representation)

                confirm_hidden_layer_1 = tf.nn.sigmoid(
                    tf.matmul(decision, confirm_w1_hidden_layer) + confirm_b1_hidden_layer)
                confirm_hidden_layer_1_with_dropout = tf.nn.dropout(confirm_hidden_layer_1, keep_prob)

                confirm_contribution = tf.matmul(confirm_hidden_layer_1_with_dropout,
                                                 confirm_w1_softmax) + confirm_b1_softmax
                list_of_sys_confirmations_contributions.append(confirm_contribution)

            sysconf = tf.concat(list_of_sys_confirmations_contributions, 1)

        # Context Modelling
        append_zeros_none = tf.zeros([tf.shape(utterance_presoftmax)[0], 1])
        utterance_presoftmax = tf.concat([utterance_presoftmax, append_zeros_none], 1)

        append_zeros = tf.zeros([tf.shape(utterance_presoftmax)[0], 1])
        sysreq = tf.concat([sysreq, append_zeros], 1)
        sysconf = tf.concat([sysconf, append_zeros], 1)

        y_presoftmax = utterance_presoftmax + sysconf + sysreq

        update_coefficient = tf.Variable(0.5)  # this scales the contribution of the current turn.
        y_combine = update_coefficient * y_presoftmax + (1 - update_coefficient) * y_past_state
        y = tf.nn.softmax(y_combine)

        return y_combine, y

    def _CNN_model(self, user_utterance_full):
        """
        Representation learning for user utterance
        :param user_utterance_full:
                the utterance for user, whose shape is
                [batch_size, longest_utterance_length, vector_dimension]
                and type is tf.float32
        :return:
            utterance_representation_cnn:
                the representation for user utterance, whose shape is
                [batch_size, num_filters]
                and type is tf.float32
        """
        with tf.variable_scope('cnn_model'):
            filter_sizes = [1, 2, 3]
            utterance_representation_cnn = tf.zeros([self.num_filters], tf.float32)
            for i, filter_size in enumerate(filter_sizes):
                filter_shape = [filter_size, self.vector_dimension, 1, self.num_filters]
                filter_weight = tf.Variable(
                    tf.truncated_normal(filter_shape, stddev=0.1)
                )
                filter_bias = tf.Variable(
                    tf.constant(0.1, shape=[self.num_filters])
                )
                conv = tf.nn.conv2d(
                    input=tf.expand_dims(user_utterance_full, -1),
                    filter=filter_weight,
                    strides=[1, 1, 1, 1],
                    padding="VALID"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, filter_bias), name="relu")
                pooled = tf.nn.max_pool(
                    value=h,
                    ksize=[1, self.longest_utterance_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID'
                )
                utterance_representation_cnn += tf.reshape(pooled, [-1, self.num_filters])

        return utterance_representation_cnn


def main():
    """Setting parameters."""
    batch_size = 512
    vector_dimension = 300
    label_count = 92
    num_filters = 300
    longest_utterance_length = 40

    """Creating nbtCell."""
    nbtCell = NbtCell(batch_size=batch_size,
                      vector_dimension=vector_dimension,
                      label_count=label_count,
                      num_filters=num_filters,
                      longest_utterance_length=longest_utterance_length)

    """Testing cnn model."""
    # user_utterance_full = tf.constant(
    #     np.random.rand(batch_size, longest_utterance_length, vector_dimension).astype(np.float32)
    # )
    # utterance_representation = nbtCell._CNN_model(user_utterance_full=user_utterance_full)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print(sess.run(utterance_representation).shape)

    """Testing nbtCell.build()"""
    user_utterance_full = tf.constant(
        np.random.rand(batch_size, longest_utterance_length, vector_dimension).astype(np.float32)
    )
    slot_vectors = tf.constant(
        np.random.rand(label_count+1, vector_dimension).astype(np.float32)
    )
    value_vectors = tf.constant(
        np.random.rand(label_count+1, vector_dimension).astype(np.float32)
    )
    system_act_request_slots = tf.constant(
        np.random.rand(batch_size, vector_dimension).astype(np.float32)
    )
    system_act_confirm_slots = tf.constant(
        np.random.rand(batch_size, vector_dimension).astype(np.float32)
    )
    system_act_confirm_values = tf.constant(
        np.random.rand(batch_size, vector_dimension).astype(np.float32)
    )
    y_past_state = tf.constant(
        np.eye(N=batch_size,
               M=label_count+1)[np.random.randint(low=0,
                                                  high=label_count,
                                                  size=batch_size)].astype(np.float32)
    )

    y_combine, y = nbtCell.build(user_utterance_full=user_utterance_full,
                   slot_vectors=slot_vectors,
                   value_vectors=value_vectors,
                   system_act_request_slots=system_act_request_slots,
                   system_act_confirm_slots=system_act_confirm_slots,
                   system_act_confirm_values=system_act_confirm_values,
                   y_past_state=y_past_state)
    print(y_combine)
    print(y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_combine_value, y_value = sess.run([y_combine, y])
        print(y_combine_value)
        print(y_value)




if __name__ == "__main__":
    main()