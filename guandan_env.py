# 2025/3/23 15:02
import gym
from gym import spaces
import numpy as np
from Game import GuandanGame
from typing import Optional

class GuandanEnv(gym.Env):
    def __init__(self):
        super(GuandanEnv, self).__init__()
        self.game = GuandanGame()

        # 定义状态空间：3049 维
        self.observation_space = spaces.Box(low=0, high=1, shape=(3049,), dtype=np.float32)

        # 动作空间：假设最多 54 种可能出牌方式（可以根据实际情况调整）
        self.action_space = spaces.Discrete(54)

    def _get_obs(self):
        """
        构造状态向量，总共 3049 维
        """
        obs = np.zeros(3049)  # ✅ 修正 obs 长度

        # 1️⃣ **当前玩家的手牌 (108 维)**
        for card in self.game.players[self.game.current_player].hand:
            obs[self.card_to_index(card)] = 1  # ✅ 确保 `hand` 是列表

        # 2️⃣ **其他玩家手牌 (3 维，表示手牌数量)**
        offset = 108
        for i, player in enumerate(self.game.players):
            if i != self.game.current_player:
                obs[offset + i] = min(len(player.hand), 26) / 26.0  # ✅ 归一化到 [0,1]
        offset += 3  # ✅ 其他玩家手牌数量 (3 维)

        # 3️⃣ **每个玩家最近动作 (108 * 4 = 432 维)**
        for i in range(4):
            action = self.game.recent_actions.get(i, [])  # ✅ 确保 `recent_actions[i]` 是列表
            for card in action:
                obs[offset + i * 108 + self.card_to_index(card)] = 1
        offset += 108 * 4

        # 4️⃣ **其他玩家已出的牌 (108 * 3 = 324 维)**
        for i, player in enumerate(self.game.players):
            if i != self.game.current_player:
                for card in player.played_cards:
                    obs[offset + i * 108 + self.card_to_index(card)] = 1
        offset += 108 * 3

        # 5️⃣ **当前级牌 (13 维)**
        obs[offset + self.level_card_to_index(self.game.active_level)] = 1
        offset += 13

        # 6️⃣ **最近 5 轮历史 (108 * 4 * 5 = 2160 维)**
        history_length = min(5, len(self.game.history))  # ✅ 确保访问最近 5 轮
        for round_idx in range(history_length):
            round_actions = self.game.history[-(history_length - round_idx)]  # 取最近 5 轮
            for player_idx, action in enumerate(round_actions):
                for card in action:
                    obs[offset + round_idx * 108 * 4 + player_idx * 108 + self.card_to_index(card)] = 1
        offset += 108 * 4 * 5

        # 7️⃣ **协作/压制/辅助状态 (3×3 = 9 维)**
        coop_status = self.compute_coop_status()  # [1, 0, 0]
        dwarf_status = self.compute_dwarf_status()  # [1, 0, 0]
        assist_status = self.compute_assist_status()  # [1, 0, 0]

        obs[offset:offset + 3] = coop_status
        obs[offset + 3:offset + 6] = dwarf_status
        obs[offset + 6:offset + 9] = assist_status
        offset += 9

        # ✅ 确保 offset 没有超出 3049
        assert offset == 3049, f"⚠️ offset 计算错误: 预期 3049, 实际 {offset}"

        return obs

    def get_action_space(self):
        """
        生成所有可能的出牌动作
        """
        action_space = []

        # ✅ 遍历当前玩家的所有可能出牌
        player_hand = self.game.players[self.game.current_player].hand
        for size in range(1, min(5, len(player_hand)) + 1):  # 只允许最多出 5 张
            for i in range(len(player_hand) - size + 1):
                move = player_hand[i:i + size]
                if self.game.rules.is_valid_play(move):  # 只添加合法出牌
                    action_space.append(move)

        return action_space

    def action_to_cards(self, action_index):
        """
        将 DQN 选择的索引转换为具体的出牌
        """
        action_space = self.get_action_space()  # ✅ 获取所有可能的出牌
        if 0 <= action_index < len(action_space):
            return action_space[action_index]  # ✅ 返回索引对应的出牌列表
        else:
            raise ValueError(f"⚠️ 无效的 action_index: {action_index}")

    def step(self, action):
        """
        执行 AI 选择的动作
        """
        # ✅ **确保 `action` 是列表**
        if isinstance(action, (int, np.int64)):
            action = self.action_to_cards(action)  # 把索引转换成对应的出牌

        # ✅ 确保 `action` 是合法的列表
        if not isinstance(action, list):
            raise ValueError(f"⚠️ 非法 action 类型: {type(action)}, 需要是列表！action={action}")

        # **检查是否是合法出牌**
        if not self.game.rules.is_valid_play(action):
            raise ValueError(f"⚠️ AI 选择了不合法的出牌: {action}")

        # **让当前玩家出牌**
        current_player = self.game.current_player
        self.game.recent_actions[current_player] = action  # ✅ 记录出牌
        for card in action:
            self.game.players[current_player].hand.remove(card)  # ✅ 从手牌移除

        # **执行完整回合**
        self.game.play_turn()  # ✅ 现在 play_turn() 不需要 `action` 了

        # **获取新状态**
        obs = self._get_obs()

        # **计算奖励**
        reward = self.compute_reward()

        # **判断游戏是否结束**
        done = self.game_over()

        return obs, reward, done, {}

    def compute_reward(self):
        """计算当前的奖励"""
        if self.game_over():
            # 如果游戏结束，给胜利队伍正奖励，失败队伍负奖励
            return 100 if self.current_player in self.winning_team else -100

        # **鼓励 AI 先出完手牌**
        hand_size = len(self.players[self.current_player].hand)
        return -hand_size  # 手牌越少，奖励越高

    def reset(self, *, seed=None, options=None):
        """
        重置环境，返回状态 `obs` 和 空字典 `info`
        """
        super().reset(seed=seed)  # ✅ 确保调用 Gym 的 `reset()`
        self.game = GuandanGame(active_level=self.game.active_level)  # ✅ 继承上一局的级牌
        obs = self._get_obs()

        return obs, {}  # ✅ 返回 (obs, info)

    def render(self, mode="human"):
        """
        渲染当前状态
        """
        print(f"当前玩家: {self.game.current_player + 1}")
        print(f"玩家手牌: {self.game.players[self.game.current_player]}")

    def card_to_index(self, card):
        """
        牌面转换为索引
        """
        card_map = {card: i for i, card in enumerate(self.game.rules.CARD_RANKS.keys())}
        return card_map.get(card, 0)

    def level_card_to_index(self, level_card):
        """
        级牌转换为 one-hot 索引 (2 -> 0, 3 -> 1, ..., A -> 12)
        """
        levels = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        return levels.index(str(level_card)) if str(level_card) in levels else 0

    def compute_coop_status(self):
        """
        计算协作状态：
        [1, 0, 0] -> 不能协作
        [0, 1, 0] -> 选择协作
        [0, 0, 1] -> 拒绝协作
        """
        return [1, 0, 0]  # 目前默认"不能协作"，后续可修改逻辑

    def compute_dwarf_status(self):
        """
        计算压制状态：
        [1, 0, 0] -> 不能压制
        [0, 1, 0] -> 选择压制
        [0, 0, 1] -> 拒绝压制
        """
        return [1, 0, 0]  # 目前默认"不能压制"，后续可修改逻辑

    def compute_assist_status(self):
        """
        计算辅助状态：
        [1, 0, 0] -> 不能辅助
        [0, 1, 0] -> 选择辅助
        [0, 0, 1] -> 拒绝辅助
        """
        return [1, 0, 0]  # 目前默认"不能辅助"，后续可修改逻辑

