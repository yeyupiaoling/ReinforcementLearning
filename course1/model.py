import parl
from parl import layers


class Model(parl.Model):
    def __init__(self, act_dim):
        self.fc1 = layers.fc(size=128, act='relu')
        self.fc2 = layers.fc(size=128, act='relu')
        self.fc3 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q
