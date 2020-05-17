import random
import torch
from rnn_utils import sample_temperature, torch_device


def generate_phrase(rnn, length, voc, starting_characters=None, temperature=1.):
    if starting_characters is None:
        character = voc.random_character()
    else:
        character = random.choice(starting_characters)

    result = [character]

    rnn.zero_state()
    x = voc.string2one_hot(character)

    with torch.no_grad():
        rnn.eval()

        for _ in range(length):
            probabilities = rnn(x.to(torch_device))
            x = sample_temperature(probabilities, temperature)
            character = voc.tensor2string(x)
            x = voc.tensor2one_hot(x)
            result.append(character)

        rnn.eval()

    return ''.join(result)
