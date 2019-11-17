import json
import torch
import torch.nn as nn


class CharacterLevelCNN(nn.Module):
    def __init__(self):
        super(CharacterLevelCNN, self).__init__()

        # model parameters
        self.number_of_characters = 69
        # self.extra_characters = "éàèùâêîôûçëïü"
        self.extra_characters = ""
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
        self.max_length = 1014
        self.number_of_classes = 3
        self.dropout_input_p = 0

        # define dropout input

        self.dropout_input = nn.Dropout2d(self.dropout_input_p)

        # define conv layers

        self.conv1 = nn.Sequential(nn.Conv1d(self.number_of_characters + len(self.extra_characters),
                                             256,
                                             kernel_size=7,
                                             padding=0),
                                   nn.ReLU(),
                                   nn.MaxPool1d(3)
                                   )

        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=7, padding=0),
                                   nn.ReLU(),
                                   nn.MaxPool1d(3)
                                   )

        self.conv3 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3, padding=0),
                                   nn.ReLU()
                                   )

        self.conv4 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3, padding=0),
                                   nn.ReLU()
                                   )

        self.conv5 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3, padding=0),
                                   nn.ReLU()
                                   )

        self.conv6 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3, padding=0),
                                   nn.ReLU(),
                                   nn.MaxPool1d(3)
                                   )

        # compute the  output shape after forwarding an input to the conv layers

        input_shape = (128,
                       self.max_length,
                       self.number_of_characters + len(self.extra_characters))
        self.output_dimension = self._get_conv_output(input_shape)

        # define linear layers

        self.fc1 = nn.Sequential(
            nn.Linear(self.output_dimension, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc3 = nn.Linear(1024, self.number_of_classes)

        # initialize weights

        self._create_weights()

    # utility private functions

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def _get_conv_output(self, shape):
        x = torch.rand(shape)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        output_dimension = x.size(1)
        return output_dimension

    # get model params:

    def get_model_parameters(self):
        return {
            'alphabet': self.alphabet,
            'extra_characters': self.extra_characters,
            'number_of_characters': self.number_of_characters,
            'max_length': self.max_length,
            'num_classes': self.number_of_classes
        }

    # forward

    def forward(self, x):
        x = self.dropout_input(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
