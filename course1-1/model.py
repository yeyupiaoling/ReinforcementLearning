import paddle.nn as nn
import paddle.nn.functional as F


class Model(nn.Layer):
    def __init__(self, num_inputs, num_actions):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
