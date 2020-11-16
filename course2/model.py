import parl
from parl import layers


class Model(parl.Model):
    def __init__(self, act_dim):
        super().__init__()
        self.conv1 = layers.conv2d(num_filters=32, filter_size=8, stride=4, padding=1, act='relu')
        self.conv2 = layers.conv2d(num_filters=64, filter_size=4, stride=2, padding=2, act='relu')
        self.conv3 = layers.conv2d(num_filters=64, filter_size=3, stride=1, padding=0, act='relu')

        self.fc1 = layers.fc(size=128, act='relu')
        self.fc2 = layers.fc(size=128, act='relu')
        self.fc3 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        con1 = self.conv1(obs)
        con2 = self.conv2(con1)
        con3 = self.conv3(con2)

        h1 = self.fc1(con3)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q
