# 马里奥

## 安装
```shell
pip install gym-super-mario-bros -i https://mirrors.aliyun.com/pypi/simple/
```

## 游戏场景
<table>
<thead>
<tr>
<th align="left">Environment</th>
<th align="left">Game</th>
<th align="left">ROM</th>
<th align="left">Screenshot</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><code>SuperMarioBros-v0</code></td>
<td align="left">SMB</td>
<td align="left">standard</td>
<td align="left"><img alt="" src="https://warehouse-camo.ingress.cmh1.psfhosted.org/c4717c633d3823dda390ebc21bac34b18e7c22c3/68747470733a2f2f757365722d696d616765732e67697468756275736572636f6e74656e742e636f6d2f323138343436392f34303934383832302d33643135653563322d363833302d313165382d383164342d6563666166666565306131342e706e67"></td>
</tr>
<tr>
<td align="left"><code>SuperMarioBros-v1</code></td>
<td align="left">SMB</td>
<td align="left">downsample</td>
<td align="left"><img alt="" src="https://warehouse-camo.ingress.cmh1.psfhosted.org/e8eda56caeefcada9af67f43385ef1f48d0ac394/68747470733a2f2f757365722d696d616765732e67697468756275736572636f6e74656e742e636f6d2f323138343436392f34303934383831392d33636666366334382d363833302d313165382d383337332d3866616431363635616337322e706e67"></td>
</tr>
<tr>
<td align="left"><code>SuperMarioBros-v2</code></td>
<td align="left">SMB</td>
<td align="left">pixel</td>
<td align="left"><img alt="" src="https://warehouse-camo.ingress.cmh1.psfhosted.org/7f53e40eb716be49673cf41fb833486ab3ee104a/68747470733a2f2f757365722d696d616765732e67697468756275736572636f6e74656e742e636f6d2f323138343436392f34303934383831382d33636561303964342d363833302d313165382d386566612d3866333464386230356231312e706e67"></td>
</tr>
<tr>
<td align="left"><code>SuperMarioBros-v3</code></td>
<td align="left">SMB</td>
<td align="left">rectangle</td>
<td align="left"><img alt="" src="https://warehouse-camo.ingress.cmh1.psfhosted.org/51975e7cc634efb02ed92acfb56368733b25f4d9/68747470733a2f2f757365722d696d616765732e67697468756275736572636f6e74656e742e636f6d2f323138343436392f34303934383831372d33636436363030612d363833302d313165382d386162622d3963656536613331643337372e706e67"></td>
</tr>
<tr>
<td align="left"><code>SuperMarioBros2-v0</code></td>
<td align="left">SMB2</td>
<td align="left">standard</td>
<td align="left"><img alt="" src="https://warehouse-camo.ingress.cmh1.psfhosted.org/0618011a5c6cedb9dba051b8cf134ba51dd0777a/68747470733a2f2f757365722d696d616765732e67697468756275736572636f6e74656e742e636f6d2f323138343436392f34303934383832322d33643362383431322d363833302d313165382d383630622d6166333830326635333733662e706e67"></td>
</tr>
<tr>
<td align="left"><code>SuperMarioBros2-v1</code></td>
<td align="left">SMB2</td>
<td align="left">downsample</td>
<td align="left"><img alt="" src="https://warehouse-camo.ingress.cmh1.psfhosted.org/7c42437f4d2f447e192c088eab22739534c2d9be/68747470733a2f2f757365722d696d616765732e67697468756275736572636f6e74656e742e636f6d2f323138343436392f34303934383832312d33643264363161322d363833302d313165382d383738392d6139326537353061613961382e706e67"></td>
</tr></tbody></table>


## 返回info内容解读


<table>
<thead>
<tr>
<th align="left">Key</th>
<th align="left">Type</th>
<th align="left">Description</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><code>coins</code></td>
<td align="left"><code>int</code></td>
<td align="left">收集硬币的数量</td>
</tr>
<tr>
<td align="left"><code>flag_get</code></td>
<td align="left"><code>bool</code></td>
<td align="left">如果马里奥拿到了旗帜或斧头，则为真</td>
</tr>
<tr>
<td align="left"><code>life</code></td>
<td align="left"><code>int</code></td>
<td align="left">剩余生命数，即<em>{3,2,1}</em></td>
</tr>
<tr>
<td align="left"><code>score</code></td>
<td align="left"><code>int</code></td>
<td align="left">游戏内累积分数</td>
</tr>
<tr>
<td align="left"><code>stage</code></td>
<td align="left"><code>int</code></td>
<td align="left">当前阶段，即, <em>{1, ..., 4}</em></td>
</tr>
<tr>
<td align="left"><code>status</code></td>
<td align="left"><code>str</code></td>
<td align="left">马里奥的状态，即, <em>{'small', 'tall', 'fireball'}</em></td>
</tr>
<tr>
<td align="left"><code>time</code></td>
<td align="left"><code>int</code></td>
<td align="left">游戏剩余时间</td>
</tr>
<tr>
<td align="left"><code>world</code></td>
<td align="left"><code>int</code></td>
<td align="left">当前的世界，即., <em>{1, ..., 8}</em></td>
</tr>
<tr>
<td align="left"><code>x_pos</code></td>
<td align="left"><code>int</code></td>
<td align="left">马里奥的<em>x</em>在舞台上的位置(从左边开始)</td>
</tr>
<tr>
<td align="left"><code>y_pos</code></td>
<td align="left"><code>int</code></td>
<td align="left">马里奥的<em>y</em>在舞台中的位置(从底部开始)</td>
</tr></tbody></table>

# 测试
```shell
python test_env.py
```

# 训练

## 安装PaddlePaddle
```shell
pip install paddlepaddle-gpu==2.0.0 -i https://mirrors.aliyun.com/pypi/simple/
```

## 执行训练
通过传参可以选择各个关卡
```shell
python train.py
```

## 执行预测
```shell
python infer.py
```

