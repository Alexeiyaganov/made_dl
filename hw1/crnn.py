import torch
import torch.nn as nn
from torchvision.models import resnet18

class CRNN(nn.Module):

    def __init__(self, num_chars, rnn_hidden_size=256, dropout=0.1):
        super(CRNN, self).__init__()
        self.num_chars = num_chars
        self.rnn_hidden_size = rnn_hidden_size
        self.dropout = dropout

        resnet = resnet18(pretrained=False)

        resnet_modules = list(resnet.children())[:-3]
        self.cnn_p1 = nn.Sequential(*resnet_modules)

        # CNN Part 2
        self.cnn_p2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 6), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.linear1 = nn.Linear(1024, 256)

        self.lstm1 = nn.LSTM(input_size=rnn_hidden_size,
                             hidden_size=rnn_hidden_size,
                             bidirectional=True,
                             batch_first=True)
        self.lstm2 = nn.LSTM(input_size=rnn_hidden_size,
                             hidden_size=rnn_hidden_size,
                             bidirectional=True,
                             batch_first=True)
        self.linear2 = nn.Linear(self.rnn_hidden_size * 2, num_chars)

    def forward(self, batch):
        batch = self.cnn_p1(batch)

        batch = self.cnn_p2(batch)

        batch = batch.permute(0, 3, 1, 2)

        batch_size = batch.size(0)
        T = batch.size(1)
        batch = batch.view(batch_size, T, -1)

        batch = self.linear1(batch)

        batch, hidden = self.lstm1(batch)
        feature_size = batch.size(2)
        batch = batch[:, :, :feature_size // 2] + batch[:, :, feature_size // 2:]

        batch, hidden = self.lstm2(batch)

        batch = self.linear2(batch)

        batch = batch.permute(1, 0, 2)

        return batch