import retrowrapper
import retro

if __name__ == "__main__":
    # 使用retrowrapper可以同时创建多个retro游戏
    env1 = retrowrapper.RetroWrapper(game='SuperMarioBros-Nes',
                                     use_restricted_actions=retro.Actions.DISCRETE,
                                     skill_frame=1,
                                     resize_shape=(1, 112, 112),
                                     render_preprocess=True)
    env2 = retrowrapper.RetroWrapper(game='SuperMarioBros-Nes',
                                     use_restricted_actions=retro.Actions.DISCRETE,
                                     skill_frame=1,
                                     resize_shape=(1, 112, 112),
                                     render_preprocess=True)
    _obs = env1.reset()
    _obs2 = env2.reset()

    while True:
        action = env1.action_space.sample()
        _obs, _rew, done, _info = env1.step(action)
        env1.render()
        if done:
            env1.reset()

        action = env2.action_space.sample()
        _obs2, _rew2, done2, _info2 = env2.step(action)
        env2.render()
        if done2:
            env1.reset()
