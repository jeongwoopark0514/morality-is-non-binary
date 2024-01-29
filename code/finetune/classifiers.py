import torch
import torch.nn as nn
import torch.optim as optim


class LinearCLS(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(LinearCLS, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits


class LinearReg(nn.Module):
    # Object Constructor
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    # define the forward function for prediction
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layer_1 = nn.Linear(self.input_size, self.hidden_size[0])
        self.layer_2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.layer_3 = nn.Linear(self.hidden_size[1], self.output_size)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)

        return x
