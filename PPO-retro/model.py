import paddle.nn as nn
import paddle.nn.functional as F


class Model(nn.Layer):
    def __init__(self, num_inputs, num_actions):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2D(num_inputs, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2D(32)
        self.conv2 = nn.Conv2D(32, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2D(64)
        self.conv3 = nn.Conv2D(64, 64, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2D(64)
        self.conv4 = nn.Conv2D(64, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2D(128)
        self.linear = nn.Linear(128 * 6 * 6, 1024)
        self.flatten = nn.Flatten()
        self.critic_linear = nn.Linear(1024, 1)
        self.actor_linear = nn.Linear(1024, num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.linear(x)
        # 定义动作模型和评估模型
        return self.actor_linear(x), self.critic_linear(x)
