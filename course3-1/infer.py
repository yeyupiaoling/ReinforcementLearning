import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import paddle
from env import create_train_env
from model import Model
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import paddle.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Contra Nes""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument("--saved_path", type=str, default="models")
    parser.add_argument("--output_path", type=str, default="output")
    args = parser.parse_args()
    return args


def infer(args):
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
    # 创建游戏环境
    env = create_train_env(args.world, args.stage, actions)
    # 创建模型
    model = Model(env.observation_space.shape[0], len(actions))
    # 加载模型参数文件
    model.load_dict(paddle.load("{}/model_{}_{}.pdparams".format(args.saved_path, args.world, args.stage)))
    # 切换评估模式
    model.eval()
    # 获取刚开始的游戏图像
    state = paddle.to_tensor(env.reset(), dtype="float32")
    total_reward = 0
    while True:
        # 显示界面
        env.render()
        # 预测动作概率和评估值
        logits, value = model(state)
        # 获取动作的序号
        policy = F.softmax(logits, axis=1)
        action = paddle.argmax(policy)[0]
        # 执行游戏
        state, reward, done, info = env.step(int(action))
        total_reward += reward
        # 转换每一步都游戏状态
        state = paddle.to_tensor(state, dtype="float32")
        # 游戏通关
        if info["flag_get"]:
            print("World {} stage {} 通关".format(args.world, args.stage))
            break
        if done:
            print("游戏结束，得分：%f, 未能通过！" % total_reward)
            break


if __name__ == "__main__":
    opt = get_args()
    infer(opt)
