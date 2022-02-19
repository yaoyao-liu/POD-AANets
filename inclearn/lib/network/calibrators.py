import torch
from torch import nn


class CalibrationWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        self.start_indexes = []
        self.end_indexes = []
        self.models = nn.ModuleList([])

    def add_model(self, model, start_index, end_index):
        self.models.append(model)
        self.start_indexes.append(start_index)
        self.end_indexes.append(end_index)

    def forward(self, inputs):
        corrected_inputs = []

        if self.start_indexes[0] != 0:
            corrected_inputs.append(inputs[..., :self.start_indexes[0]])

        for model, start_index, end_index in zip(self.models, self.start_indexes, self.end_indexes):
            corrected_inputs.append(model(inputs[..., start_index:end_index]))

        if self.end_indexes[-1] != inputs.shape[1]:
            corrected_inputs.append(inputs[..., self.end_indexes[-1]:])

        corrected_inputs = torch.cat(corrected_inputs, dim=-1)

        return corrected_inputs


class LinearModel(nn.Module):
    def __init__(self, alpha=1., beta=0.):
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, inputs):
        return self.alpha * inputs + self.beta


class TemperatureScaling(nn.Module):

    def __init__(self, temperature=1):
        super().__init__()

        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, inputs):
        return inputs / self.temperature
