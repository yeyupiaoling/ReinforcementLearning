import parl
from parl import layers


class Model(parl.Model):
    def __init__(self, act_dim):
        self.conv1 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')
        self.conv2 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')
        self.conv3 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')
        self.conv4 = layers.conv2d(num_filters=32, filter_size=3, stride=2, padding=1, act='relu')

        self.fc = layers.fc(size=512, act='relu')

        self.policy_fc = layers.fc(size=act_dim)
        self.value_fc = layers.fc(size=1)

    def policy(self, obs):
        """
        Args:
            obs: 输入的图像，shape为[N, C, H, W]

        Returns:
            policy_logits: N * ACTION_DIM
        """
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        flatten = layers.flatten(conv4, axis=1)
        fc_output = self.fc(flatten)

        policy_logits = self.policy_fc(fc_output)
        return policy_logits

    def value(self, obs):
        """
        Args:
            obs: 输入的图像，shape为[N, C, H, W]

        Returns:
            values: N
        """
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        flatten = layers.flatten(conv4, axis=1)
        fc_output = self.fc(flatten)

        values = self.value_fc(fc_output)
        values = layers.squeeze(values, axes=[1])
        return values

    def policy_and_value(self, obs):
        """
        Args:
            obs: 输入的图像，shape为[N, C, H, W]

        Returns:
            policy_logits: N * ACTION_DIM
            values: N
        """
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        flatten = layers.flatten(conv4, axis=1)
        fc_output = self.fc(flatten)

        policy_logits = self.policy_fc(fc_output)

        values = self.value_fc(fc_output)
        values = layers.squeeze(values, axes=[1])

        return policy_logits, values
