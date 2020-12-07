import parl
from parl import layers

LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0


class ActorModel(parl.Model):
    def __init__(self, act_dim):
        self.conv1 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')
        self.conv2 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')
        self.conv3 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')
        self.conv4 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')

        self.fc1 = layers.fc(size=512, act='relu')
        self.mean_linear = layers.fc(size=act_dim)
        self.log_std_linear = layers.fc(size=act_dim)

    def policy(self, obs):
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        fc = self.fc1(conv4)
        means = self.mean_linear(fc)
        log_std = self.log_std_linear(fc)
        log_std = layers.clip(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return means, log_std


class CriticModel(parl.Model):
    def __init__(self):
        self.conv1 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')
        self.conv2 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')
        self.conv3 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')
        self.conv4 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')

        self.fc1 = layers.fc(size=512, act='relu')
        self.fc2 = layers.fc(size=1, act=None)

        self.fc3 = layers.fc(size=512, act='relu')
        self.fc4 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        hid1 = self.fc1(conv4)
        concat1 = layers.concat([hid1, act], axis=1)
        Q1 = self.fc2(concat1)
        Q1 = layers.squeeze(Q1, axes=[1])

        conv5 = self.conv1(obs)
        conv6 = self.conv2(conv5)
        conv7 = self.conv3(conv6)
        conv8 = self.conv3(conv7)

        hid2 = self.fc3(conv8)
        concat2 = layers.concat([hid2, act], axis=1)
        Q2 = self.fc4(concat2)
        Q2 = layers.squeeze(Q2, axes=[1])

        return Q1, Q2
