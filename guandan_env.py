import numpy as np
from get_actions import enumerate_colorful_actions
import random
from collections import Counter
try:
    from c_rule import Rules  # 导入 Cython 版本
except ImportError:
    from rule import Rules  # 退回 Python 版本
try:
    from c_give_cards import create_deck, shuffle_deck, deal_cards
except ImportError:
    from give_cards import create_deck, shuffle_deck, deal_cards

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

class Player:
    def __init__(self, hand):
        """
        程序里的玩家是从0开始的，输出时会+1
        """
        self.hand = hand  # 手牌
        self.played_cards = []  # 记录已出的牌

class GuandanGame:
    def __init__(self, user_player=None, active_level=None,verbose=True , print_history=False):
        # **两队各自的级牌**
        self.print_history = print_history
        self.active_level = active_level if active_level else random.choice(range(2, 15))
        # 历史记录，记录最近 20 轮的出牌情况（每轮包含 4 个玩家的出牌）
        self.history = []
        # **只传当前局的有效级牌**
        self.rules = Rules(self.active_level)
        self.players = [Player(hand) for hand in deal_cards(shuffle_deck(create_deck()))]# 发牌
        self.current_player = 0  # 当前出牌玩家
        self.last_play = None  # 记录上一手牌
        self.last_player = -1  # 记录上一手是谁出的
        self.pass_count = 0  # 记录连续 Pass 的次数
        self.user_player = user_player - 1 if user_player else None  # 转换为索引（0~3）
        self.ranking = []  # 存储出完牌的顺序
        self.recent_actions = [[],[],[],[]]
        self.verbose = verbose  # 控制是否输出文本
        self.team_1 = {0, 2}
        self.team_2 = {1, 3}
        self.is_free_turn=True
        self.jiefeng = False
        self.winning_team = None
        self.is_game_over = False

        # **手牌排序**
        for player in self.players:
            player.hand = self.sort_cards(player.hand)

    def log(self, message):
        """控制是否打印消息"""
        if self.verbose:
            print(message)

    def sort_cards(self, cards):
        """按牌的大小排序（从大到小）"""
        return sorted(cards, key=lambda card: self.rules.get_rank(card), reverse=True)

    def play_turn(self):
        """执行当前玩家的回合"""

        player = self.players[self.current_player]  # 获取当前玩家对象

        # **计算当前仍有手牌的玩家数**
        active_players = 4-len(self.ranking)

        # **如果 Pass 的人 == "当前有手牌的玩家数 - 1"，就重置轮次**
        if self.pass_count >= (active_players - 1) and self.current_player not in self.ranking:
            if self.jiefeng:
                first_player = self.ranking[-1]
                teammate = 2 if first_player == 0 else 0 if first_player == 2 else 3 if first_player == 1 else 1
                self.log(f"\n🆕 轮次重置！玩家 {teammate + 1} 接风。\n")
                self.recent_actions[self.current_player] = []  # 记录空列表
                self.current_player = (self.current_player + 1) % 4
                self.last_play = None  # ✅ 允许新的自由出牌
                self.pass_count = 0  # ✅ Pass 计数归零
                self.is_free_turn = True
                self.jiefeng = False
            else:
                self.log(f"\n🆕 轮次重置！玩家 {self.current_player + 1} 可以自由出牌。\n")
                self.last_play = None  # ✅ 允许新的自由出牌
                self.pass_count = 0  # ✅ Pass 计数归零
                self.is_free_turn = True

        result = self.ai_play(player)
        # **记录最近 5 轮历史**
        if self.current_player == 0:
            round_history = [self.recent_actions[i] for i in range(4)]
            self.history.append(round_history)
            self.recent_actions=[['None'],['None'],['None'],['None']]
            '''
            if len(self.history) > 20:
                self.history.pop(0)
            '''
        return result

    def get_possible_moves(self, player_hand):
        """获取所有可能的合法出牌，包括顺子（5 张）、连对（aabbcc）、钢板（aaabbb）"""

        possible_moves = []
        hand_points = [self.rules.get_rank(card) for card in player_hand]  # 仅点数（去掉花色）
        hand_counter = Counter(hand_points)  # 统计点数出现次数
        unique_points = sorted(set(hand_points))  # 仅保留唯一点数，排序

        # 1. **原逻辑（单张、对子、三条、炸弹等）**
        for size in [1, 2, 3, 4, 5, 6, 7, 8]:
            for i in range(len(player_hand) - size + 1):
                move = player_hand[i:i + size]
                if self.rules.can_beat(self.last_play, move):
                    possible_moves.append(move)

        # 2. **检查顺子（固定 5 张）**
        for i in range(len(unique_points) - 4):  # 只找长度=5 的顺子
            seq = unique_points[i:i + 5]
            if self.rules._is_consecutive(seq) and 15 not in seq:  # 不能有大小王
                move = self._map_back_to_suit(seq, player_hand)  # 还原带花色的牌
                if self.rules.can_beat(self.last_play, move):
                    possible_moves.append(move)

        # 3. **检查连对（aabbcc）**
        for i in range(len(unique_points) - 2):  # 只找 3 组对子
            seq = unique_points[i:i + 3]
            if all(hand_counter[p] >= 2 for p in seq):  # 每张至少两张
                move = self._map_back_to_suit(seq, player_hand, count=2)  # 每点数取 2 张
                if self.rules.can_beat(self.last_play, move):
                    possible_moves.append(move)

        # 4. **检查钢板（aaabbb）**
        for i in range(len(unique_points) - 1):  # 只找 2 组三张
            seq = unique_points[i:i + 2]
            if all(hand_counter[p] >= 3 for p in seq):  # 每张至少 3 张
                move = self._map_back_to_suit(seq, player_hand, count=3)  # 每点数取 3 张
                if self.rules.can_beat(self.last_play, move):
                    possible_moves.append(move)

        return possible_moves

    def _map_back_to_suit(self, seq, sorted_hand, count=1):
        """从手牌映射回带花色的牌"""
        move = []
        hand_copy = sorted_hand[:]  # 复制手牌
        for p in seq:
            for _ in range(count):  # 取 count 张
                for card in hand_copy:
                    if self.rules.get_rank(card) == p:
                        move.append(card)
                        hand_copy.remove(card)
                        break
        return move

    def get_valid_action_mask(self,hand, M, level_rank):
        mask = np.zeros(len(M), dtype=np.float32)
        for action in M:
            action_id = action['id']
            combos = enumerate_colorful_actions(action, hand, level_rank)
            if combos:  # 至少有一种组合可用
                mask[action_id] = 1.0
        return mask

    def ai_play(self, player):
        """AI 出牌逻辑（随机选择合法且能压过上家的出牌）"""

        # **如果玩家已经打完，仍然记录一个空列表，然后跳过**
        if self.current_player in self.ranking:
            self.recent_actions[self.current_player] = []  # 记录空列表
            self.current_player = (self.current_player + 1) % 4

            return self.check_game_over()

        player_hand = player.hand

        possible_moves = self.get_possible_moves(player_hand)
        if not self.is_free_turn:
            possible_moves.append([])

        if not possible_moves:
            self.log(f"玩家 {self.current_player + 1} Pass")
            self.pass_count += 1
            self.recent_actions[self.current_player] = ['Pass']  # 记录 Pass
        else:
            chosen_move = random.choice(possible_moves) # 随机选择一个合法的牌型
            if not chosen_move:
                self.log(f"玩家 {self.current_player + 1} Pass")
                self.pass_count += 1
                self.recent_actions[self.current_player] = ['Pass']  # 记录 Pass
            else:
                # 如果 chosen_move 不为空，继续进行正常的出牌逻辑
                self.last_play = chosen_move
                self.last_player = self.current_player
                for card in chosen_move:
                    player_hand.remove(card)
                self.log(f"玩家 {self.current_player + 1} 出牌: {' '.join(chosen_move)}")
                self.recent_actions[self.current_player] = list(chosen_move)  # 记录出牌
                self.jiefeng = False
                if not player_hand:  # 玩家出完牌
                    self.log(f"\n🎉 玩家 {self.current_player + 1} 出完所有牌！\n")
                    self.ranking.append(self.current_player)
                    if len(self.ranking)<=2:
                        self.jiefeng=True

                self.pass_count = 0
                if not player_hand:
                    self.pass_count -= 1

                if self.is_free_turn:
                    self.is_free_turn = False

        self.current_player = (self.current_player + 1) % 4
        return self.check_game_over()

    def check_game_over(self):
        """检查游戏是否结束"""
        # **如果有 2 个人出完牌，并且他们是同一队伍，游戏立即结束**
        if len(self.ranking) >= 2:
            first_player, second_player = self.ranking[0], self.ranking[1]
            if (first_player in self.team_1 and second_player in self.team_1) or (
                    first_player in self.team_2 and second_player in self.team_2):
                self.ranking.extend(i for i in range(4) if i not in self.ranking)  # 剩下的按出牌顺序补全
                self.update_level()
                self.is_game_over = True
                return True

        # **如果 3 人出完了，自动补全最后一名，游戏结束**
        if len(self.ranking) == 3:
            self.ranking.append(next(i for i in range(4) if i not in self.ranking))  # 找出最后一个玩家
            self.update_level()
            self.is_game_over = True
            return True

        return False

    def update_level(self):
        """升级级牌"""
        first_player = self.ranking[0]  # 第一个打完牌的玩家
        winning_team = 1 if first_player in self.team_1 else 2
        # 确定队友
        teammate = 2 if first_player == 0 else 0 if first_player == 2 else 3 if first_player == 1 else 1

        # 找到队友在排名中的位置
        teammate_position = self.ranking.index(teammate)

        # 头游 + 队友的名次，确定得分
        upgrade_map = {1: 3, 2: 2, 3: 1}  # 头游 + (队友的名次) 对应的升级规则
        upgrade_amount = upgrade_map[teammate_position]

        self.log(f"\n🏆 {winning_team} 号队伍获胜！得 {upgrade_amount} 分")
        # 显示最终排名
        ranks = ["头游", "二游", "三游", "末游"]
        for i, player in enumerate(self.ranking):
            self.log(f"{ranks[i]}：玩家 {player + 1}")

    def play_game(self):
        """执行一整局游戏"""
        self.log(f"\n🎮 游戏开始！当前级牌：{RANKS[self.active_level - 2]}")

        while True:
            if self.play_turn():
                if self.current_player != 0:
                    round_history = [self.recent_actions[i] for i in range(4)]
                    self.history.append(round_history)
                if self.print_history:
                    for i in range(len(self.history)):
                        self.log(self.history[i])
                break

    def show_user_hand(self):
        """显示用户手牌（按排序后的顺序）"""
        sorted_hand = self.players[self.user_player].hand
        print("\n你的手牌：", " ".join(sorted_hand))
        if self.last_play:
            print(f"场上最新出牌：{' '.join(self.last_play)}\n")

    def _get_obs(self):
        """
        构造状态向量，总共 3049 维
        """
        obs = np.zeros(3049)  # ✅ 修正 obs 长度

        # 1️⃣ **当前玩家的手牌 (108 维)**
        for card in self.players[self.current_player].hand:
            obs[self.card_to_index(card)] = 1  # ✅ 确保 `hand` 是列表

        # 2️⃣ **其他玩家手牌 (3 维，表示手牌数量)**
        offset = 108
        for i, player in enumerate(self.players):
            if i != self.current_player:
                obs[offset + i] = min(len(player.hand), 26) / 26.0  # ✅ 归一化到 [0,1]
        offset += 3  # ✅ 其他玩家手牌数量 (3 维)

        # 3️⃣ **每个玩家最近动作 (108 * 4 = 432 维)**
        for i in range(4):
            action = self.recent_actions[i] if self.recent_actions[i] else []
            for card in action:
                obs[offset + i * 108 + self.card_to_index(card)] = 1
        offset += 108 * 4

        # 4️⃣ **其他玩家已出的牌 (108 * 3 = 324 维)**
        for i, player in enumerate(self.players):
            if i != self.current_player:
                for card in player.played_cards:
                    obs[offset + i * 108 + self.card_to_index(card)] = 1
        offset += 108 * 3

        # 5️⃣ **当前级牌 (13 维)**
        obs[offset + self.level_card_to_index(self.active_level)] = 1
        offset += 13

        # 6️⃣ **最近 5 轮历史 (108 * 4 * 5 = 2160 维)**
        history_length = min(5, len(self.history))  # ✅ 确保访问最近 5 轮
        for round_idx in range(history_length):
            round_actions = self.history[-(history_length - round_idx)]  # 取最近 5 轮
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

    def compute_reward(self):
        """计算当前的奖励"""
        if self.check_game_over():
            # 如果游戏结束，给胜利队伍正奖励，失败队伍负奖励
            return 100 if self.current_player in self.winning_team else -100

        # **鼓励 AI 先出完手牌**
        hand_size = len(self.players[self.current_player].hand)
        return -hand_size  # 手牌越少，奖励越高

    def card_to_index(self, card):
        """
        牌面转换为索引
        """
        card_map = {card: i for i, card in enumerate(self.rules.CARD_RANKS.keys())}
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
if __name__ == "__main__":
    game = GuandanGame(user_player=None,active_level=None,verbose=True,print_history=True)
    game.play_game()
