import parl
from parl import layers


class Model(parl.Model):
    def __init__(self, act_dim):
        super().__init__()
        self.conv1 = layers.conv2d(num_filters=32, filter_size=3, stride=1, padding=1, act='relu')
        self.conv2 = layers.conv2d(num_filters=32, filter_size=3, stride=1, padding=1, act='relu')
        self.conv3 = layers.conv2d(num_filters=32, filter_size=3, stride=1, padding=1, act='relu')
        self.conv4 = layers.conv2d(num_filters=32, filter_size=3, stride=1, padding=1, act='relu')

        self.fc1 = layers.fc(size=128, act='relu')
        self.fc2 = layers.fc(size=128, act='relu')
        self.fc3 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        h1 = self.fc1(conv4)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q
