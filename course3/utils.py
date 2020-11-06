import numpy as np


# 十进制转二进制
def dec_to_bin(num, act_dim):
    arry = []
    while True:
        arry.append(str(num % 2))
        num = num // 2
        if num == 0:
            break
    for i in range(act_dim - len(arry)):
        arry.append('0')
    arry = [int(a) for a in arry[::-1]]
    print(arry)
    return arry


# 创建所有的动作组合
def create_actions(act_dim):
    actions = []
    print("===========所有的动作组合===========")
    for i in range(2**act_dim):
        a = dec_to_bin(i, act_dim)
        actions.append(a)
    return np.array(actions)

