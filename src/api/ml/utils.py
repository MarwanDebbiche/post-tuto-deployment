import numpy as np
import torch
import torch.nn.functional as F


def predict_sentiment(model, text, alphabet, extra_characters, number_of_characters, max_length, num_classes):

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

    prediction = model(processed_output)
    probabilities = F.softmax(prediction, dim=1)
    proba, index = torch.max(probabilities, dim=1)
    proba = proba.item()
    index = index.item()

    if num_classes == 3:

        if index == 0:
            score = (0.33 - 0) * (1 - proba) + 0

        elif index == 1:
            score = (0.67 - 0.33) * proba + 0.33

        elif index == 2:
            score = (1 - 0.67) * proba + 0.67
        
    elif num_classes == 2:
        score = proba

    return score
