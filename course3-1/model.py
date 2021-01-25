import paddle.nn as nn
import paddle.nn.functional as F


class Model(nn.Layer):
    def __init__(self, num_inputs, num_actions):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2D(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2D(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2D(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2D(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)
        self.flatten = nn.Flatten()
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        x = self.linear(x)
        # 定义动作模型和评估模型
        return self.actor_linear(x), self.critic_linear(x)
