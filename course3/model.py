import parl
from parl import layers


class Model(parl.Model):
    def __init__(self, action_dim):
        self.actor_model = ActorModel(action_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()


class ActorModel(parl.Model):
    def __init__(self, action_dim):
        super().__init__()
        self.conv1 = layers.conv2d(num_filters=32, filter_size=3, stride=1, padding=1, act='relu')
        self.conv2 = layers.conv2d(num_filters=32, filter_size=3, stride=1, padding=1, act='relu')
        self.conv3 = layers.conv2d(num_filters=32, filter_size=3, stride=1, padding=1, act='relu')
        self.conv4 = layers.conv2d(num_filters=32, filter_size=3, stride=1, padding=1, act='relu')
        self.fc1 = layers.fc(size=512, act='relu')
        self.fc2 = layers.fc(size=512, act='relu')
        self.fc3 = layers.fc(size=action_dim, act=None)

    def policy(self, obs):
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        hid1 = self.fc1(conv4)
        hid2 = self.fc2(hid1)
        means = self.fc3(hid2)
        return means


class CriticModel(parl.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.conv2d(num_filters=32, filter_size=3, stride=1, padding=0, act='relu')
        self.conv2 = layers.conv2d(num_filters=32, filter_size=3, stride=1, padding=0, act='relu')
        self.conv3 = layers.conv2d(num_filters=32, filter_size=3, stride=1, padding=0, act='relu')
        self.conv4 = layers.conv2d(num_filters=32, filter_size=3, stride=1, padding=0, act='relu')
        self.fc1 = layers.fc(size=512, act='relu')
        self.fc2 = layers.fc(size=512, act='relu')
        self.fc3 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        hid1 = self.fc1(conv4)
        concat = layers.concat([hid1, act], axis=1)
        hid2 = self.fc2(concat)
        Q = self.fc3(hid2)
        Q = layers.squeeze(Q, axes=[1])
        return Q
