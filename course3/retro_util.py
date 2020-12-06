import cv2
import retro
import numpy as np


class RetroEnv(retro.RetroEnv):
    def __init__(self, game, state=retro.State.DEFAULT, scenario=None, info=None,
                 use_restricted_actions=retro.Actions.FILTERED, record=False, players=1,
                 inttype=retro.data.Integrations.STABLE, obs_type=retro.Observations.IMAGE,
                 resize_shape=None, skill_frame=1, render_preprocess=False):
        super(RetroEnv, self).__init__(game, state=state, scenario=scenario, info=info,
                                       use_restricted_actions=use_restricted_actions,
                                       record=record, players=players, inttype=inttype, obs_type=obs_type)
        self.resize_shape = resize_shape
        self.skill_frame = skill_frame
        self.render_preprocess = render_preprocess
        self.observation_space.shape = resize_shape

    def step(self, a):
        for _ in range(self.skill_frame):
            obs, reward, terminal, info = super(RetroEnv, self).step(a)
        obs = self.preprocess(obs, self.render_preprocess)
        return obs, reward, terminal, info

    def reset(self):
        obs = super(RetroEnv, self).reset()
        obs = self.preprocess(obs, self.render_preprocess)
        return obs

    # 改变游戏的布局环境，减低输入图像的复杂度
    def change_obs_color(self, obs, src, target):
        for i in range(len(src)):
            index = (obs == src[i])
            obs[index] = target[i]
        return obs

    # 图像预处理
    def preprocess(self, observation, render=False):
        if self.resize_shape is None:
            if render:
                observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
                # 显示处理过的图像
                cv2.imshow("preprocess", observation)
                cv2.waitKey(1)
            return observation
        assert self.resize_shape[0] == 1 or self.resize_shape[0] == 3
        w, h, c = observation.shape
        observation = observation[25:h, 15:w]
        if self.resize_shape[0] == 1:
            # 把图像转成灰度图
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            # 把其他的亮度调成一种，减低图像的复杂度
            observation = self.change_obs_color(observation, [66, 88, 114, 186, 189, 250], [255, 255, 255, 255, 255, 0])
            observation = cv2.resize(observation, (self.resize_shape[2], self.resize_shape[1]))
            if render:
                # 显示处理过的图像
                cv2.imshow("preprocess", observation)
                cv2.waitKey(1)
            observation = np.expand_dims(observation, axis=0)
        else:
            observation = cv2.resize(observation, (self.resize_shape[2], self.resize_shape[1]))
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
            if render:
                # 显示处理过的图像
                cv2.imshow("preprocess", observation)
                cv2.waitKey(1)
            observation = observation.transpose((2, 0, 1))
        observation = observation / 255.0
        return observation
