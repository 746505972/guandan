# 2025/3/25 10:58
import gym
import numpy as np
import random

class Player:
    """玩家类，包含手牌信息"""
    def __init__(self, hand):
        self.hand = hand  # 存储玩家手牌

def create_deck():
    """创建两副扑克牌"""
    suits = ['♠', '♥', '♣', '♦']
    values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    deck = [suit + value for suit in suits for value in values] * 2  # 2 副牌
    return deck

def shuffle_deck(deck):
    """洗牌"""
    random.shuffle(deck)

def deal_cards():
    """发牌：平均分配给 4 名玩家"""
    deck = create_deck()
    shuffle_deck(deck)
    return [Player(deck[i::4]) for i in range(4)]

class GuandanTrainEnv(gym.Env):
    """掼蛋简化训练环境"""

    def __init__(self):
        super(GuandanTrainEnv, self).__init__()
        self.players = deal_cards()  # 发牌
        self.current_player = 0  # 轮到谁出牌
        self.ranking = []  # 记录玩家出完牌的顺序
        self.active_level = None  # 当前级牌
        self.winning_team = None  # 获胜队伍
        self.score = 0  # 得分

    def reset(self):
        """重置游戏"""
        self.active_level = random.choice(['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'])
        self.players = deal_cards()
        self.current_player = 0
        self.ranking = []
        self.winning_team = None
        self.score = 0
        return self._get_obs()

    def _get_obs(self):
        """获取当前状态（DQN 训练使用）"""
        obs = np.zeros(3049)  # 示例状态维度，真实实现需要修改
        return obs

    def step(self, action):
        """执行 AI 选择的出牌动作"""
        self.execute_action(self.current_player, action)  # 处理出牌
        done = self.check_game_over()
        reward = self.compute_reward()
        obs = self._get_obs()
        return obs, reward, done, {}

    def execute_action(self, player_idx, action):
        """处理出牌逻辑"""
        player = self.players[player_idx]

        # 确保 action 合法
        for card in action:
            if card in player.hand:
                player.hand.remove(card)

        # 记录出完牌的玩家
        if len(player.hand) == 0 and player_idx not in self.ranking:
            self.ranking.append(player_idx)

        # 切换到下一个玩家
        self.current_player = (self.current_player + 1) % 4

    def check_game_over(self):
        """检查游戏是否结束（3 人出完牌）"""
        if len(self.ranking) < 3:
            return False  # 还没 3 个人出完牌，游戏继续

        first_player = self.ranking[0]  # 第一个出完牌的玩家
        team = [first_player, (first_player + 2) % 4]  # 确定队友（0&2，1&3）

        # 检查队友名次
        teammate_rank = None
        for i, player in enumerate(self.ranking):
            if player in team:
                teammate_rank = i + 1  # 排名从 1 开始
                break

        # 计算得分
        if teammate_rank == 4:
            self.score = 1
        elif teammate_rank == 3:
            self.score = 2
        elif teammate_rank == 2:
            self.score = 3
            return True  # 提前结束游戏（同队玩家都出完）

        return len(self.ranking) == 3  # 3 人出完牌，游戏结束

    def compute_reward(self):
        """计算奖励：
        - 胜利队伍: 100 + self.score * 10
        - 失败队伍: -100
        - 牌越少，奖励越高
        """
        if not self.check_game_over():
            return -len(self.players[self.current_player].hand)  # 鼓励早点出牌

        return (100 + self.score * 10) if self.current_player in self.winning_team else -100

    def ai_play(self, player_idx):
        """AI 出牌逻辑（随机出牌或 DQN 选择动作）"""
        player_hand = self.players[player_idx].hand
        if not player_hand:
            return []  # 没牌可出

        # 随机选择 1-5 张牌出
        move_size = random.randint(1, min(5, len(player_hand)))
        move = random.sample(player_hand, move_size)
        return move
