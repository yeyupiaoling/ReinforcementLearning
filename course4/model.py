import parl
from paddle import fluid
from parl import layers


class Model(parl.Model):
    def __init__(self, obs_dim, act_dim, policy_lr=1e-9, value_lr=1e-9, init_logvar=-1.0):
        self.policy_model = PolicyModel(act_dim, init_logvar)
        self.value_model = ValueModel()
        self.policy_lr = policy_lr
        self.value_lr = value_lr

    def policy(self, obs):
        return self.policy_model.policy(obs)

    def policy_sample(self, obs):
        return self.policy_model.sample(obs)

    def value(self, obs):
        return self.value_model.value(obs)


class PolicyModel(parl.Model):
    def __init__(self, act_dim, init_logvar):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')
        self.conv2 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')
        self.conv3 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')
        self.conv4 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')

        self.fc1 = layers.fc(size=256, act='tanh')
        self.fc2 = layers.fc(size=512, act='tanh')
        self.fc3 = layers.fc(size=128, act='tanh')
        self.fc4 = layers.fc(size=act_dim, act='tanh')

        self.logvars = layers.create_parameter(
            shape=[act_dim],
            dtype='float32',
            default_initializer=fluid.initializer.ConstantInitializer(init_logvar))

    def policy(self, obs):
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        hid1 = self.fc1(conv4)
        hid2 = self.fc2(hid1)
        hid3 = self.fc3(hid2)
        means = self.fc4(hid3)
        logvars = self.logvars()
        return means, logvars

    def sample(self, obs):
        means, logvars = self.policy(obs)
        sampled_act = means + (
                layers.exp(logvars / 2.0) *  # stddev
                layers.gaussian_random(shape=(self.act_dim,), dtype='float32'))
        return sampled_act


class ValueModel(parl.Model):
    def __init__(self):
        super(ValueModel, self).__init__()

        self.conv1 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')
        self.conv2 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')
        self.conv3 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')
        self.conv4 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')

        self.fc1 = layers.fc(size=256, act='tanh')
        self.fc2 = layers.fc(size=512, act='tanh')
        self.fc3 = layers.fc(size=128, act='tanh')
        self.fc4 = layers.fc(size=1)

    def value(self, obs):
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        hid1 = self.fc1(conv4)
        hid2 = self.fc2(hid1)
        hid3 = self.fc3(hid2)
        V = self.fc4(hid3)
        V = layers.squeeze(V, axes=[])
        return V
