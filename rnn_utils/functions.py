from torch.distributions import Categorical


def read_file_string(file_path, encoding=None):
    return open(file_path, 'r', encoding=encoding).read()


def sample_temperature(probabilities, temperature=1.):
    letter_probabilities = (probabilities / temperature).exp()
    tensor = Categorical(letter_probabilities).sample()
    return tensor
