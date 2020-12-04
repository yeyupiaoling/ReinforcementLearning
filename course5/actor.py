import cv2
import gym
import parl


@parl.remote_class
class Actor(object):
    def __init__(self, config):
        self.config = config

        self.env = gym.make(config['env_name'])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.preprocess(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.preprocess(obs)
        return obs


    # 改变游戏的布局环境，减低输入图像的复杂度
    def change_obs_color(self, obs, src, target):
        for i in range(len(src)):
            index = (obs == src[i])
            obs[index] = target[i]
        return obs

    # 图像预处理
    def preprocess(self, observation):
        assert self.config['obs_shape'][0] == 1 or self.config['obs_shape'][0] == 3
        w, h, c = observation.shape
        observation = observation[25:h, 15:w]
        if self.config['obs_shape'][0] == 1:
            # 把图像转成灰度图
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            # 把其他的亮度调成一种，减低图像的复杂度
            observation = self.change_obs_color(observation, [66, 88, 114, 186, 189, 250], [255, 255, 255, 255, 255, 0])
            observation = cv2.resize(observation, (self.config['obs_shape'][2], self.config['obs_shape'][1]))
            observation = np.expand_dims(observation, axis=0)
        else:
            observation = cv2.resize(observation, (self.config['obs_shape'][2], self.config['obs_shape'][1]))
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
            observation = observation.transpose((2, 0, 1))
        observation = observation / 255.0
        return observation

