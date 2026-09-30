import torch
from model import DefectDetectorCNN


def test_output_shape():
    model = DefectDetectorCNN(2)
    output = model(torch.zeros(2, 3, 128, 128))
    assert tuple(output.shape) == (2, 2)
