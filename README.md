# RL_Game 项目说明

本项目基于 `gym-super-mario-bros` 实现马里奥强化学习训练，包含：

- 环境快速验证脚本
- DDQN（2动作 / 4动作）Notebook 训练版本
- PPO 脚本版本
- 已训练权重（`weights/`）

---

## 1. 文件与目录作用

### `1.env_test.py`
- 最小可运行环境测试脚本。
- 创建 `SuperMarioBros-v0` 环境并使用 `COMPLEX_MOVEMENT` 动作空间随机采样。
- 作用是验证：环境是否安装成功、渲染是否正常、`env.step()` 是否可执行。

### `2.train_ddqn_4_action.ipynb`
- 核心教学与训练 Notebook（本 README 重点讲解对象）。
- 采用 **DDQN（Double DQN）**，动作空间为 4 个：`left`、`left+A`、`right`、`right+A`。
- 包含：环境预处理、Q 网络定义、经验回放、目标网络同步、训练主循环。
- 特点：`SkipFrame` 中加入了 reward 分解调试（`x / time / death`），便于分析奖励来源。

### `3.train_ddqn_2_action.ipynb`
- 与 4-action 结构基本一致，但动作空间简化为 2 个：`right`、`right+A`。
- 用于降低动作复杂度、加快早期收敛。
- 相比 4-action 版本，reward 分解调试逻辑更简洁（默认只做 skip 累加）。

### `4.mario_ppo.py`
- PPO 训练脚本（非 Notebook）。
- Actor-Critic 结构 + GAE（广义优势估计）+ clipping 目标。
- 适合持续脚本训练与 checkpoint 管理。

### `weights/`
- `weights/ddqn/`：DDQN 模型参数（如 `checkpoint_4action_12060.pth`）。
- `weights/ppo/`：PPO 模型参数（如 `checkpoint_3490.pth`）。

---

## 2. 以 `2.train_ddqn_4_action.ipynb` 讲强化学习步骤与原理

下面按“代码结构 -> 算法含义”的顺序解释。

### Step A：构建环境与动作空间

Notebook 先创建：

- `env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')`
- `env = JoypadSpace(env, [["left"],["left","A"],["right"],["right","A"]])`

含义：
- 智能体每一步只能从这 4 个离散动作中选 1 个。
- 强化学习中的动作集合 $\mathcal{A}$ 被离散化，便于 DQN 输出每个动作的 Q 值。

### Step B：状态预处理（降低维度 + 保留运动信息）

通过三个 wrapper：

1. `SkipFrame(skip=4)`：每次动作执行 4 帧并累加奖励。
2. `GrayScaleObservation`：RGB 转灰度，降低输入冗余。
3. `ResizeObservation(shape=84)`：统一到 $84\times84$。
4. `FrameStack(num_stack=4)`：拼接连续 4 帧，得到状态形状 `(4,84,84)`。

原理：
- 单帧难以感知速度方向，4 帧堆叠可近似提供“时序信息”。
- 这是 Atari / Mario 类视觉强化学习常见输入范式。

### Step C：Q 网络与 Double DQN 结构

`DDQNSolver` 内有两个同结构 CNN：

- `online`：用于当前策略评估与更新。
- `target`：用于计算稳定目标值（参数冻结，周期性同步）。

核心思想（Double DQN）：

- 动作选择由 `online` 完成：
  $$a^* = \arg\max_a Q_{online}(s', a)$$
- 动作评估由 `target` 完成：
  $$Q_{target}(s', a^*)$$

这样可减少普通 DQN 中 max 操作导致的过估计问题。

### Step D：经验回放与目标值计算

`DDQNAgent` 的关键函数：

- `remember(...)`：把 `(s, s', a, r, done)` 存入 replay buffer。
- `recall()`：随机采样 batch。
- `experience_replay()`：计算损失并反向传播。

目标值计算代码对应：

$$
y = r + (1-done)\cdot\gamma\cdot Q_{target}(s',\arg\max_a Q_{online}(s',a))
$$

其中：
- `done=1` 时，后续价值截断，只保留即时奖励 $r$。
- `gamma=0.95` 控制未来回报折扣。

损失函数使用 `SmoothL1Loss`（Huber Loss），对异常值更稳。

### Step E：探索-利用（epsilon-greedy）

`act()` 中：

- 以概率 `exploration_rate` 随机动作（探索）。
- 否则选 `argmax Q`（利用）。
- `exploration_rate` 每步衰减到下限 `0.01`。

这解决了强化学习早期“只利用当前差策略”导致陷入局部最优的问题。

### Step F：训练主循环

主循环逻辑：

1. `state = env.reset()`
2. `action = agent.act(state)`
3. `next_state, reward, done, info = env.step(action)`
4. `agent.remember(...)`
5. `agent.experience_replay(reward)`
6. `state = next_state`，直到 `done`
7. 每 `checkpoint_period` 回合记录均值奖励并保存 checkpoint

这就是完整的“交互 -> 存储 -> 学习 -> 重复”闭环。

---

## 3. 重点：reward 组成、计算方法、对应文件

本项目 reward 相关可以分成 3 层看：**环境底层定义**、**wrapper 聚合**、**训练使用方式**。

### 3.1 环境底层 reward（Mario 环境）

在 `2.train_ddqn_4_action.ipynb` 末尾单元通过 `inspect.getsource(...)` 打印了底层实现：

```python
def _get_reward(self):
    return self._x_reward + self._time_penalty + self._death_penalty
```

即底层奖励由三项组成：

1. `x_reward`：位置推进奖励（一般越向右推进越高）。
2. `time_penalty`：时间相关惩罚（拖延会被惩罚）。
3. `death_penalty`：死亡惩罚。

> 对应文件位置：
> - 外部包文件：`gym_super_mario_bros/smb_env.py`（在你本地环境路径中，不在本仓库）
> - 仓库中的定位代码：`2.train_ddqn_4_action.ipynb` 最后一个代码单元（`inspect` 打印）。

### 3.2 Wrapper 层 reward 聚合（本仓库可见）

`2.train_ddqn_4_action.ipynb` 的 `SkipFrame.step()`：

- 连续执行 `self._skip` 次环境步进；
- 将每个 inner step 的 reward 做求和：

$$
R_{skip} = \sum_{i=1}^{k} r_i
$$

其中 $k=4$（默认）。这意味着 agent 一次决策对应 4 帧累计回报。

### 3.3 4-action 版本的 reward 分解调试

`SkipFrame` 中新增 `_estimate_reward_terms(...)`，通过 `info['x_pos']`、`info['time']`、`done/flag_get` 估算三项：

- `x_delta = x_pos - last_x_pos`
- `x_reward = x_delta`（并对异常跳变做截断）
- `time_delta = time_left - last_time`
- `time_penalty = time_delta`（时间减少通常为负）
- `death_penalty = -25`（死亡且未通关）

这部分主要用于 **解释和打印调试**，帮助你观察奖励来源，不直接替代底层环境 reward。

### 3.4 训练时如何使用 reward

在 DDQN 中，`reward` 直接参与目标值：

$$
q_{target} = r + (1-done)\gamma\cdot next\_q
$$

也就是说：
- reward 越能体现“向目标推进”的质量，学习越快；
- reward 噪声或偏置过大，会导致 Q 值估计不稳定。

---

## 4. 奖励相关代码索引（快速定位）

- `2.train_ddqn_4_action.ipynb`
  - `SkipFrame.step()`：skip 帧奖励累加
  - `_estimate_reward_terms()`：分项估算与 debug 打印
  - 训练循环中的 `reward` -> `remember` -> `experience_replay`
  - 末尾 `inspect` 单元：打印底层 `_get_reward`

- `3.train_ddqn_2_action.ipynb`
  - `SkipFrame.step()`：skip 帧奖励累加（无详细分解调试）
  - DDQN 目标值计算同 4-action

- `4.mario_ppo.py`
  - `SkipFrame.step()`：同样做 skip 帧奖励累加
  - reward 后续用于优势估计（GAE）和 PPO 更新

---

## 5. 运行建议

- 本地训练时可开启 `env.render()` 观察行为；云端训练建议关闭渲染以提高速度。
- 若继续训练，请在对应脚本/Notebook 中设置 checkpoint 开关与文件名。
- 建议优先从 2-action 版本验证流程，再迁移到 4-action 提升策略表达能力。
