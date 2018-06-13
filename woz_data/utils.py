import numpy as np

longest_utterance_length = 40
vector_dimension = 300
label_count = 92


def read_data(target_slot):
    return [20, 16, 10, 5]


def read_batch_data(pos,
                    batch_size,
                    data_set):
    user_utterance_full = np.random.rand(batch_size, longest_utterance_length, vector_dimension).astype(np.float32)
    slot_vectors = np.random.rand(label_count + 1, vector_dimension).astype(np.float32)
    value_vectors = np.random.rand(label_count + 1, vector_dimension).astype(np.float32)
    system_act_request_slots = np.random.rand(batch_size, vector_dimension).astype(np.float32)
    system_act_confirm_slots = np.random.rand(batch_size, vector_dimension).astype(np.float32)
    system_act_confirm_values = np.random.rand(batch_size, vector_dimension).astype(np.float32)
    y_past_state = np.eye(N=batch_size,
               M=label_count + 1)[np.random.randint(low=0,
                                                    high=label_count,
                                                    size=batch_size)].astype(np.float32)
    y_state = np.eye(N=batch_size,
                          M=label_count + 1)[np.random.randint(low=0,
                                                               high=label_count,
                                                               size=batch_size)].astype(np.float32)

    batch_data = {}
    batch_data["user_utterance_full_batch"] = user_utterance_full
    batch_data["slot_vectors_batch"] = slot_vectors
    batch_data["value_vectors_batch"] = value_vectors
    batch_data["system_act_request_slots_batch"] = system_act_request_slots
    batch_data["system_act_confirm_slots_batch"] = system_act_confirm_slots
    batch_data["system_act_confirm_values_batch"] = system_act_confirm_values
    batch_data["y_past_state_batch"] = y_past_state
    batch_data["y_state_batch"] = y_state
    index = 500

    return batch_data, index
