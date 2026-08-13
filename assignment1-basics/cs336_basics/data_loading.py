import numpy.typing as npt

import numpy
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # note that the range is [low, high)
    starting_indexes = numpy.random.randint(0, dataset.size - context_length, size=batch_size)
    inputs = []
    expects = []
    for index in starting_indexes:
        input = dataset[index : index + context_length]
        expect = dataset[index + 1 : index + context_length + 1]
        inputs.append(input.astype(numpy.int64))
        expects.append(expect.astype(numpy.int64))

    # numpy.stack() turns B arrays of shape (m,) into one array of shape (B, m).
    i = torch.from_numpy(numpy.stack(inputs)).to(device)
    e = torch.from_numpy(numpy.stack(expects)).to(device)
    return tuple[torch.Tensor, torch.Tensor]([i, e])
