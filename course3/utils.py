import os

os.environ['OMP_NUM_THREADS'] = '1'
import paddle
from model import Model
from env import create_train_env
import paddle.nn.functional as F
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from visualdl import LogWriter
import hashlib


# 评估模型
def eval(args, num_states, num_actions):
    log_writer = LogWriter(logdir='log')
    # 固定初始化状态
    paddle.seed(123)
    # 使用 GPU预测
    if paddle.is_compiled_with_cuda():
        paddle.set_device("gpu:0")
    # 判断游戏动作类型
    if args.action_type == "right":
        actions = RIGHT_ONLY
    elif args.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    # 创建游戏动作
    env = create_train_env(args.world, args.stage, actions)
    # 获取网络模型
    local_model = Model(num_states, num_actions)
    # 切换为评估状态
    local_model.eval()
    # 将图像转换为Paddle的数据类型
    state = paddle.to_tensor(env.reset(), dtype="float32")
    # 一开始就更新模型参数
    done = True
    # 日志的记录步数
    step = 0
    # 旧模型的MD5
    old_model_file_md5 = ''
    # 游戏总得分
    total_reward = 0
    while True:
        # 每结束一次就更新模型参数
        if done:
            try:
                model_path = "{}/model_{}_{}.pdparams".format(args.saved_path, args.world, args.stage)
                # 使用文件的MD5保证每个模型只用一次
                with open(model_path, 'rb') as f:
                    file = f.read()
                file_md5 = hashlib.md5(file).hexdigest()
                if file_md5 == old_model_file_md5:
                    continue
                else:
                    model_dict = paddle.load(model_path)
                    old_model_file_md5 = file_md5
            except:
                continue
            total_reward = 0
            local_model.load_dict(model_dict)
        # 预测动作概率和评估值
        logits, value = local_model(state)
        # 获取动作的序号
        policy = F.softmax(logits, axis=1)
        action = paddle.argmax(policy)[0]
        # 执行游戏
        state, reward, done, info = env.step(int(action))
        total_reward += reward
        # 显示界面
        if args.show_play:
            env.render()
        # 游戏通关
        if info["flag_get"]:
            print("World {} stage {} 通关".format(args.world, args.stage))
            paddle.save(local_model.state_dict(),
                        "{}/model_{}_{}_finish.pdparams".format(args.saved_path, args.world, args.stage))
        # 重置游戏状态
        if done:
            step += 1
            state = env.reset()
            print('总得分是：%f' % total_reward)
            log_writer.add_scalar(tag='Eval reward', value=total_reward, step=step)
        # 转换每一步都游戏状态
        state = paddle.to_tensor(state, dtype="float32")


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")
