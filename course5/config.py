config = {

    # ==========  remote config ==========
    'master_address': 'localhost:8010',

    # ==========  env config ==========
    'env_name': 'SnowBrothers-Nes',
    'obs_shape': (1, 112, 112),

    # ==========  actor config ==========
    'actor_num': 1,
    'env_num': 1,
    'sample_batch_steps': 20,

    # ==========  learner config ==========
    'max_sample_steps': int(1e7),
    'gamma': 0.99,
    'lambda': 1.0,  # GAE

    # start learning rate
    'start_lr': 0.001,
    'model_path': 'models/model.ckpt',

    # coefficient of policy entropy adjustment schedule: (train_step, coefficient)
    'entropy_coeff_scheduler': [(0, -0.01)],
    'vf_loss_coeff': 0.5,
    'log_metrics_interval_s': 10,
    'save_model_interval_s': 60 * 60 * 3,
}
