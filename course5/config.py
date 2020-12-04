config = {
    # ==========  remote config ==========
    'master_address': 'localhost:8010',

    # ==========  env config ==========
    'env_name': 'PongNoFrameskip-v4',
    'env_dim': 9,
    'obs_shape': (1, 112, 112),

    # ==========  learner config ==========
    'actor_num': 4,
    'train_batch_size': 128,
    'max_predict_batch_size': 16,
    'predict_thread_num': 2,
    't_max': 5,
    'gamma': 0.99,
    'lambda': 1.0,  # GAE

    # learning rate adjustment schedule: (train_step, learning_rate)
    'lr_scheduler': [(0, 0.0005), (100000, 0.0003), (200000, 0.0001)],

    # coefficient of policy entropy adjustment schedule: (train_step, coefficient)
    'entropy_coeff_scheduler': [(0, -0.01)],
    'vf_loss_coeff': 0.5,
    'log_metrics_interval_s': 10,
}
