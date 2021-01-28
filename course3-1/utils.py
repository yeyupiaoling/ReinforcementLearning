import os

os.environ['OMP_NUM_THREADS'] = '1'
import paddle
from model import Model
from collections import deque
from env import create_train_env
import paddle.nn.functional as F
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY


# 评估模型
def eval(args, num_states, num_actions):
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
    curr_step = 0
    # 执行动作的容器
    actions = deque(maxlen=args.max_actions)
    total_reward = 0
    while True:
        curr_step += 1
        # 每结束一次就更新模型参数
        if done:
            try:
                model_dict = paddle.load("{}/model_{}_{}.pdparams".format(args.saved_path, args.world, args.stage))
            except:
                continue
            total_reward = 0
            # local_model.load_dict(model.state_dict())
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
        # env.render()
        # 记录动作
        actions.append(action)
        if curr_step > args.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        # 游戏通关
        if info["flag_get"]:
            print("World {} stage {} 通关".format(args.world, args.stage))
            paddle.save(local_model.state_dict(),
                        "{}/model_{}_{}_finish.pdparams".format(args.saved_path, args.world, args.stage))

        # 重置游戏状态
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
            print('总得分是：%f' % total_reward)
        # 转换每一步都游戏状态
        state = paddle.to_tensor(state, dtype="float32")


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")
