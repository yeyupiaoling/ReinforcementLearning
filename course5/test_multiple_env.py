import retrowrapper
import retro

if __name__ == "__main__":
    env1 = retrowrapper.RetroWrapper(game='SnowBrothers-Nes',
                                     state=retro.State.DEFAULT,
                                     use_restricted_actions=retro.Actions.DISCRETE,
                                     players=1,
                                     obs_type=retro.Observations.IMAGE)
    env2 = retrowrapper.RetroWrapper(game='SnowBrothers-Nes',
                                     state=retro.State.DEFAULT,
                                     use_restricted_actions=retro.Actions.DISCRETE,
                                     players=1,
                                     obs_type=retro.Observations.IMAGE)
    _obs = env1.reset()
    _obs2 = env2.reset()

    done = False
    while not done:
        action = env1.action_space.sample()
        print(action)
        _obs, _rew, done, _info = env1.step(action)
        env1.render()

        action = env2.action_space.sample()
        _obs2, _rew2, done2, _info2 = env2.step(action)
        env2.render()
