config = {

    # ========== 主机地址 ==========
    'master_address': 'localhost:8010',

    # ========== 游戏环境参数 ==========
    # 游戏名字
    'env_name': 'SuperMarioBros-Nes',
    # 处理后的游戏图像大小
    'obs_shape': (1, 112, 112),
    # Actor的数量
    'actor_num': 12,
    # 每个Actor生成的游戏数量
    'env_num': 10,
    # 每一次执行的游戏步数
    'sample_batch_steps': 20,

    # ========== 训练模型参数 ==========
    'max_sample_steps': int(1e7),
    'gamma': 0.99,
    'lambda': 1.0,
    # 策略模型的参数
    'entropy_coeff_scheduler': [(0, -0.01)],
    'vf_loss_coeff': 0.5,

    # 初始学习率
    'start_lr': 0.001,
    # 是否加载预训练模型
    'restore_model': False,
    # 模型的保存路径
    'model_path': 'models/',
    # 保存日志的时间间隔，单位秒
    'log_metrics_interval_s': 10,
    # 保存模型的时间间隔，单位秒
    'save_model_interval_s': 60 * 60 * 3,
}
