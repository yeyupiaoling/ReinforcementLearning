import parl
from parl import layers

LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0


class ActorModel(parl.Model):
    def __init__(self, act_dim):
        self.conv1 = layers.conv2d(num_filters=32, filter_size=3, stride=1, padding=0, act='relu')
        self.conv2 = layers.conv2d(num_filters=32, filter_size=3, stride=1, padding=0, act='relu')
        self.conv3 = layers.conv2d(num_filters=32, filter_size=3, stride=1, padding=0, act='relu')

        self.fc1 = layers.fc(size=512, act='relu')
        self.fc2 = layers.fc(size=512, act='relu')
        self.mean_linear = layers.fc(size=act_dim)
        self.log_std_linear = layers.fc(size=act_dim)

    def policy(self, obs):
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        pool = layers.pool2d(input=conv3, pool_size=2, pool_stride=2)

        hid1 = self.fc1(pool)
        hid2 = self.fc2(hid1)
        means = self.mean_linear(hid2)
        log_std = self.log_std_linear(hid2)
        log_std = layers.clip(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return means, log_std


class CriticModel(parl.Model):
    def __init__(self):
        self.conv1 = layers.conv2d(num_filters=32, filter_size=3, stride=1, padding=0, act='relu')
        self.conv2 = layers.conv2d(num_filters=64, filter_size=3, stride=1, padding=0, act='relu')
        self.conv3 = layers.conv2d(num_filters=64, filter_size=3, stride=1, padding=0, act='relu')

        self.fc1 = layers.fc(size=512, act='relu')
        self.fc2 = layers.fc(size=512, act='relu')
        self.fc3 = layers.fc(size=1, act=None)

        self.fc4 = layers.fc(size=512, act='relu')
        self.fc5 = layers.fc(size=512, act='relu')
        self.fc6 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        pool1 = layers.pool2d(input=conv3, pool_size=2, pool_stride=2)

        hid1 = self.fc1(pool1)
        concat1 = layers.concat([hid1, act], axis=1)
        Q1 = self.fc2(concat1)
        Q1 = self.fc3(Q1)
        Q1 = layers.squeeze(Q1, axes=[1])

        conv4 = self.conv1(obs)
        conv5 = self.conv2(conv4)
        conv6 = self.conv3(conv5)

        pool2 = layers.pool2d(input=conv6, pool_size=2, pool_stride=2)

        hid2 = self.fc4(pool2)
        concat2 = layers.concat([hid2, act], axis=1)
        Q2 = self.fc5(concat2)
        Q2 = self.fc6(Q2)
        Q2 = layers.squeeze(Q2, axes=[1])

        return Q1, Q2
