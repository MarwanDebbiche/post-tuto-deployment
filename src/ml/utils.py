import numpy as np
import torch

def preprocess_input(text, alphabet, extra_characters, number_of_characters, max_length):
    text = text.lower()
    text = text.strip()

    number_of_characters = number_of_characters + len(extra_characters)
    identity_mat = np.identity(number_of_characters)
    vocabulary = list(alphabet) + list(extra_characters)
    max_length = max_length

    processed_output = np.array([identity_mat[vocabulary.index(i)] for i in list(
        text[::-1]) if i in vocabulary], dtype=np.float32)
    if len(processed_output) > max_length:
        processed_output = processed_output[:max_length]
    elif 0 < len(processed_output) < max_length:
        processed_output = np.concatenate((processed_output, np.zeros(
            (max_length - len(processed_output), number_of_characters), dtype=np.float32)))
    elif len(processed_output) == 0:
        processed_output = np.zeros(
            (max_length, number_of_characters), dtype=np.float32)

    processed_output = torch.tensor(processed_output)
    processed_output = processed_output.unsqueeze(0)
    return processed_output
