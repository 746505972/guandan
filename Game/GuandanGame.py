# 2026/8/1 19:35
# 单局掼蛋
import random
from collections import Counter, defaultdict
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from Game.Player import Player
from Game.constants import RANKS, CARD_RANKS, SUITS, type_to_chinese
from Game.models import load_actor_model
from Game.utils import *

M, M_id_dict = load_actions()

class GuandanGame:
    def __init__(self, config: GameConfig):
        self.config: GameConfig = config
        # 历史记录，记录最近 20 轮的出牌情况（每轮包含 4 个玩家的出牌）
        self.history: list = []
        self.rules: Rules = Rules(self.config.active_level)
        self.players: list[Player] = [Player(hand) for hand in give_cards()]  # 发牌
        self.current_player: Literal[0, 1, 2, 3] = 0  # 当前出牌玩家
        self.last_play = []  # 记录上一手牌
        self.last_player = -1  # 记录上一手是谁出的
        self.pass_count = 0  # 记录连续 Pass 的次数
        self.user_player = None
        for i in range(4):
            if getattr(self.config, f'player_{i}') == 'Human':
                self.user_player = i
                break
        self.ranking = []  # 存储出完牌的顺序
        self.recent_actions = [[], [], [], []]
        self.team_1 = {0, 2}
        self.team_2 = {1, 3}
        self.is_free_turn = True
        self.jiefeng = False
        self.winning_team = 0
        self.is_game_over = False
        self.upgrade_amount = 0
        self.actor = load_actor_model(self.config.model)
        self.R = RANKS + [RANKS[self.config.active_level-2]] + ['小王', '大王']

        # **手牌排序**
        for player in self.players:
            player.hand = self.sort_cards(player.hand)

    def point_to_card(self,point) -> [str, list]:
        if isinstance(point, list):
            return [str(self.R[int(p) - 2]) for p in point]
        else:
            return str(self.R[point - 2])

    def log(self, message):
        """控制是否打印消息"""
        if self.config.verbose:
            print(message)

    def sort_cards(self, cards):
        """按牌的大小排序（从大到小）"""
        return sorted(cards, key=lambda card: self.rules.get_rank(card), reverse=True)

    def map_cards_to_action(self, cards, M, level_rank):
        """
        从实际出过的牌中（带花色），判断其结构动作（含同花顺识别）。
        """
        point_count = defaultdict(int)
        suits = set()
        if not cards:
            cards = []
        for card in cards:
            for rank in RANKS + ['小王', '大王']:
                if rank in card:
                    raw_point = CARD_RANKS[rank]
                    logic_point = 15 if raw_point == level_rank else raw_point
                    point_count[logic_point] += 1
                    break
            # 提取花色
            for s in SUITS:
                if card.startswith(s):
                    suits.add(s)
                    break

        # 构建点数序列（带重复）
        logic_points = []
        for pt, count in sorted(point_count.items()):
            logic_points.extend([pt] * count)

        # 🔍 同花顺检测
        if len(cards) == 5 and len(point_count) == 5:
            sorted_points = sorted(point_count.keys())
            if all(sorted_points[i + 1] - sorted_points[i] == 1 for i in range(4)):
                if len(suits) == 1:
                    # 是同花顺 → 去 M 中找类型为 straight_flush
                    for action in M:
                        if action['type'] == 'flush_rocket' and sorted(action['points']) == sorted_points:
                            return action

        # 🔁 普通结构匹配
        for action in M:
            if sorted(action['points']) == sorted(logic_points):
                return action

        return None

    def maybe_reset_turn(self):
        """计算当前仍有手牌的玩家数、处理接风等复杂逻辑"""
        active_players = 4 - len(self.ranking)
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
        # **记录最近 5 轮历史**
        if self.current_player == 0:
            round_history = [self.recent_actions[i] for i in range(4)]
            self.history.append(round_history)
            self.recent_actions = [['None'], ['None'], ['None'], ['None']]
            '''
            if len(self.history) > 20:
                self.history.pop(0)
            '''

    def play_turn(self):
        """执行当前玩家的回合"""
        player = self.players[self.current_player]  # 获取当前玩家对象
        # TODO: 添加选座位接口
        if self.user_player == self.current_player:
            result = self.user_play(player)
        else:
            result = self.actor_play(player)

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

    def can_beat(self, curr_action, prev_action):
        """
        判断结构动作 curr_action 是否能压过 prev_action
        """
        # 如果没人出牌，当前动作永远可以出
        if prev_action["type"] == "None":
            if curr_action["type"] == "None":
                return False
            else:
                return True

        curr_type = curr_action["type"]
        prev_type = prev_action["type"]

        # 炸弹类型（根据牌力表）
        bomb_power = {
            "joker_bomb": 6,
            "8_bomb": 5,
            "7_bomb": 4,
            "6_bomb": 3,
            "flush_rocket": 2,
            "5_bomb": 1,
            "4_bomb": 0
        }

        is_curr_bomb = curr_type in bomb_power
        is_prev_bomb = prev_type in bomb_power

        # ✅ 炸弹能压非炸弹
        if is_curr_bomb and not is_prev_bomb:
            return True
        if not is_curr_bomb and is_prev_bomb:
            return False

        # ✅ 两个都是炸弹 → 比炸弹牌力 → 再比 logic_point
        if is_curr_bomb and is_prev_bomb:
            if bomb_power[curr_type] > bomb_power[prev_type]:
                return True
            elif bomb_power[curr_type] < bomb_power[prev_type]:
                return False
            else:  # 相同牌力 → 比点数
                return curr_action["logic_point"] > prev_action["logic_point"]

        # ✅ 非炸弹时，牌型必须相同才可比
        if curr_type != prev_type:
            return False

        # ✅ 非炸弹，牌型相同 → 比 logic_point
        return curr_action["logic_point"] > prev_action["logic_point"]

    def get_valid_action_mask(self, hand, M, level_rank, last_action):
        """
        返回 mask 向量，标记每个结构动作在当前手牌下是否合法。
        如果 last_action 为 None，则为主动出牌，可出任意合法牌型；
        否则为跟牌回合，只能出能压过 last_action 的合法牌。
        """
        mask = np.zeros(len(M), dtype=np.float32)
        if not last_action:
            last_action = []
        last_action = self.map_cards_to_action(last_action, M, level_rank)
        for action in M:
            action_id = action['id']
            combos = enumerate_colorful_actions(action, hand, level_rank)
            if not combos:
                continue  # 当前手牌无法组成该结构

            if last_action is None:
                # 主动出牌：只要能组成即可
                mask[action_id] = 1.0
            else:
                # 跟牌出牌：还要能压上上家
                if self.can_beat(action, last_action):
                    mask[action_id] = 1.0
        if not self.is_free_turn:
            # 永远允许出 “None” 结构（pass）
            for action in M:
                if action['type'] == 'None':
                    mask[action['id']] = 1.0
                    break

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
            chosen_move = random.choice(possible_moves)  # 随机选择一个合法的牌型
            if not chosen_move:
                self.log(f"玩家 {self.current_player + 1} Pass")
                self.pass_count += 1
                self.recent_actions[self.current_player] = ['Pass']  # 记录 Pass
            else:
                # 如果 chosen_move 不为空，继续进行正常的出牌逻辑
                self.last_play = chosen_move
                self.last_player = self.current_player
                for card in chosen_move:
                    player.played_cards.append(card)
                    player_hand.remove(card)
                self.log(f"玩家 {self.current_player + 1} 出牌: {' '.join(chosen_move)}")
                self.recent_actions[self.current_player] = list(chosen_move)  # 记录出牌
                self.jiefeng = False
                if not player_hand:  # 玩家出完牌
                    self.log(f"\n🎉 玩家 {self.current_player + 1} 出完所有牌！\n")
                    self.ranking.append(self.current_player)
                    if len(self.ranking) <= 2:
                        self.jiefeng = True

                self.pass_count = 0
                if not player_hand:
                    self.pass_count -= 1

                if self.is_free_turn:
                    self.is_free_turn = False
        player.last_played_cards = self.recent_actions[self.current_player]
        self.current_player = (self.current_player + 1) % 4
        return self.check_game_over()

    def actor_play(self, player):
        # 1. 模型推理
        state = self._get_obs()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(self.get_valid_action_mask(player.hand, M, self.config.active_level, self.last_play)).unsqueeze(0)
        probs = self.actor(state_tensor, mask)
        action_id = torch.multinomial(probs, 1).item()
        action_struct = M_id_dict[action_id]
        # 2. 枚举所有合法出牌组合（带花色）
        combos = enumerate_colorful_actions(action_struct, player.hand, self.config.active_level)
        if combos:
            chosen_move = random.choice(combos)
            if not chosen_move:
                self.log(f"玩家 {self.current_player + 1} Pass")
                self.pass_count += 1
                self.recent_actions[self.current_player] = ['Pass']  # 记录 Pass
            else:
                # 如果 chosen_move 不为空，继续进行正常的出牌逻辑
                self.last_play = chosen_move
                self.last_player = self.current_player
                for card in chosen_move:
                    player.played_cards.append(card)
                    player.hand.remove(card)
                self.log(f"玩家 {self.current_player + 1} 出牌: {' '.join(chosen_move)}")
                self.recent_actions[self.current_player] = list(chosen_move)  # 记录出牌
                self.jiefeng = False
                if not player.hand:  # 玩家出完牌
                    self.log(f"\n🎉 玩家 {self.current_player + 1} 出完所有牌！\n")
                    self.ranking.append(self.current_player)
                    if len(self.ranking) <= 2:
                        self.jiefeng = True

                self.pass_count = 0
                if not player.hand:
                    self.pass_count -= 1

                if self.is_free_turn:
                    self.is_free_turn = False
        else:
            self.log(f"玩家 {self.current_player + 1} Pass")
            self.pass_count += 1
            self.recent_actions[self.current_player] = ['Pass']  # 记录 Pass
        player.last_played_cards = self.recent_actions[self.current_player]
        self.current_player = (self.current_player + 1) % 4
        return self.check_game_over()

    def user_play(self, player):
        """用户出牌逻辑"""
        if self.current_player in self.ranking:
            self.recent_actions[self.current_player] = []  # 记录空列表
            self.current_player = (self.current_player + 1) % 4
            return self.check_game_over()

        while True:
            self.show_user_hand()  # 显示手牌
            choice = input("\n请选择要出的牌（用空格分隔），或直接回车跳过（PASS）： ").strip()

            # **用户选择 PASS**
            if choice == "" or choice.lower() == "pass":
                if self.is_free_turn:
                    print("❌ 你的输入无效，自由回合必须出牌！")
                    continue
                print(f"玩家 {self.current_player + 1} 选择 PASS")
                self.pass_count += 1
                self.recent_actions[self.current_player] = ['Pass']  # ✅ 记录 PASS
                break

            # **解析用户输入的牌**
            selected_cards = choice.split()

            # **检查牌是否在手牌中**
            if not all(card in player.hand for card in selected_cards):
                print("❌ 你的输入无效，请确保牌在你的手牌中！")
                continue  # 重新输入

            # **检查牌是否合法**
            if not self.rules.is_valid_play(selected_cards):
                print("❌ 你的出牌不符合规则，请重新选择！")
                continue  # 重新输入

            last_action = self.map_cards_to_action(self.last_play, M, self.config.active_level)
            chosen = self.map_cards_to_action(selected_cards, M, self.config.active_level)
            # **检查是否能压过上一手牌**
            if  not self.can_beat(chosen,last_action):
                print("❌ 你的牌无法压过上一手牌，请重新选择！")
                continue  # 重新输入

            # **成功出牌**
            for card in selected_cards:
                player.played_cards.append(card)
                player.hand.remove(card)  # 从手牌中移除
            self.last_play = selected_cards  # 记录这次出牌
            self.last_player = self.current_player  # 记录是谁出的
            self.recent_actions[self.current_player] = list(selected_cards)  # 记录出牌历史
            self.jiefeng = False
            print(f"玩家 {self.current_player + 1} 出牌: {' '.join(selected_cards)}")

            # **如果手牌为空，玩家出完所有牌**
            if not player.hand:
                print(f"\n🎉 玩家 {self.current_player + 1} 出完所有牌！\n")
                self.ranking.append(self.current_player)
                if len(self.ranking) <= 2:
                    self.jiefeng = True

            # **出牌成功，Pass 计数归零**
            self.pass_count = 0
            if not player.hand:
                self.pass_count -= 1
            if self.is_free_turn:
                self.is_free_turn = False
            break

        # **切换到下一个玩家**
        player.last_played_cards = self.recent_actions[self.current_player]
        self.current_player = (self.current_player + 1) % 4

        return self.check_game_over()

    def get_ai_suggestions(self):
        """返回AI给当前玩家的3个建议字符串"""
        suggestions = []
        player = self.players[self.current_player]
        state = self._get_obs()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(self.get_valid_action_mask(player.hand, M, self.config.active_level, self.last_play)).unsqueeze(0)

        with torch.no_grad():
            all_probs = self.actor(state_tensor, mask)

        top_k_orig_probs, top_k_indices = torch.topk(all_probs, k=self.config.sug_len, dim=-1)

        valid_top_k_probs = top_k_orig_probs[top_k_orig_probs > 0]
        if valid_top_k_probs.numel() > 0:
            normalized_top_k_probs_tensor = F.softmax(valid_top_k_probs, dim=-1)
            normalized_top_k_probs = torch.zeros_like(top_k_orig_probs)
            normalized_top_k_probs[top_k_orig_probs > 0] = normalized_top_k_probs_tensor
        else:
            normalized_top_k_probs = torch.zeros_like(top_k_orig_probs)

        for i in range(top_k_indices.size(1)):
            action_id = top_k_indices[0, i].item()
            normalized_prob = normalized_top_k_probs[0, i].item()

            if top_k_orig_probs[0, i].item() > 0:
                action_struct = M_id_dict.get(action_id)
                if action_struct:
                    action_desc = action_struct.get('name', action_struct.get('type', f'动作ID {action_id}'))
                    points_str = f"[{', '.join(self.point_to_card(action_struct['points']))}]" if action_struct.get('points') else ""
                    if action_struct.get('type') == 'None':
                        action_desc = "Pass (不出)"
                        points_str = ""
                    suggestions.append(f" {type_to_chinese[action_desc]}-{points_str} - {normalized_prob:.2%}")
                else:
                    suggestions.append(f"建议 {i + 1}: 未知动作 ID {action_id} - 相对概率: {normalized_prob:.2%}")
            else:
                suggestions.append(f"无有效动作")

        while len(suggestions) < self.config.sug_len:
            suggestions.append("无可用动作")

        return list(dict.fromkeys(suggestions))

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
        self.winning_team = winning_team
        # 确定队友
        teammate = 2 if first_player == 0 else 0 if first_player == 2 else 3 if first_player == 1 else 1

        # 找到队友在排名中的位置
        teammate_position = self.ranking.index(teammate)

        # 头游 + 队友的名次，确定得分
        upgrade_map = {1: 3, 2: 2, 3: 1}  # 头游 + (队友的名次) 对应的升级规则
        upgrade_amount = upgrade_map[teammate_position]
        self.upgrade_amount=upgrade_amount

        self.log(f"\n🏆 {winning_team} 号队伍获胜！得 {upgrade_amount} 分")
        # 显示最终排名
        ranks = ["头游", "二游", "三游", "末游"]
        for i, player in enumerate(self.ranking):
            self.log(f"{ranks[i]}：玩家 {player + 1}")

    def play_game(self):
        """执行一整局游戏"""
        self.log(f"\n🎮 游戏开始！当前级牌：{RANKS[self.config.active_level - 2]}")

        while True:
            if self.play_turn():
                if self.current_player != 0:
                    round_history = [self.recent_actions[i] for i in range(4)]
                    self.history.append(round_history)
                if self.config.print_history:
                    for i in range(len(self.history)):
                        self.log(self.history[i])
                break

    def show_user_hand(self):
        """显示用户手牌（按排序后的顺序）"""
        sorted_hand = self.players[self.user_player].hand
        print("\n你的手牌：", " ".join(sorted_hand))
        if self.last_play:
            print(f"场上最新出牌：{' '.join(self.last_play)}\n")
    # DONE: 检查不同座位能否正常工作
    def _get_obs(self):
        """
        构造状态向量，总共 3049 维
        """
        obs = np.zeros(3049)

        # 1️⃣ 当前玩家手牌 (108)
        obs[:108]=encode_hand_108(self.players[self.current_player].hand)
        offset = 108

        # 2️⃣ 其他玩家手牌数量 (3)
        for i, player in enumerate(self.players):
            if i != self.current_player:
                obs[offset + i] = min(len(player.hand), 26) / 26.0
        offset += 3

        # 3️⃣ 最近动作 (108 * 4 = 432)
        for i, player in enumerate(self.players):
            obs[offset + i * 108 : offset + (i + 1) * 108] = encode_hand_108(player.last_played_cards)
        offset += 108 * 4

        # 4️⃣ 其他玩家已出牌 (108 * 3 = 324)
        for i, player in enumerate(self.players):
            if i != self.current_player:
                obs[offset + i * 108 : offset + (i + 1) * 108] = encode_hand_108(player.played_cards)
        offset += 108 * 3

        # 5️⃣ 当前级牌 (13)
        obs[offset + self.level_card_to_index(self.config.active_level)] = 1
        offset += 13

        # 6️⃣ 最近 20 步动作历史 (108 * 20 = 2160)
        HISTORY_LEN = 20
        history_flat = []

        # 展平所有轮次中的动作
        for round in self.history:
            for action in round:
                history_flat.append(action)

        # 若不满 20，则在最前补空动作（表示“没人出牌”）
        while len(history_flat) < HISTORY_LEN:
            history_flat.insert(0, [])  # 用空动作填充

        # 取最后 20 个动作
        history_flat = history_flat[-HISTORY_LEN:]

        # 编码入 obs
        for i, action in enumerate(history_flat):
            start = offset + i * 108
            obs[start:start + 108] = encode_hand_108(action)
        offset += 108 * HISTORY_LEN

        # 7️⃣ 状态向量 (9)
        obs[offset:offset + 3] = self.compute_coop_status()
        obs[offset + 3:offset + 6] = self.compute_dwarf_status()
        obs[offset + 6:offset + 9] = self.compute_assist_status()
        offset += 9

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

    def submit_user_move(self, selected_cards):
        """前端提交出牌：selected_cards为list[str]，如 ['红桃3', '黑桃3'] 或 []"""
        if self.is_game_over:
            return {"error": "游戏已结束"}

        player = self.players[self.user_player]

        if not selected_cards:  # 选择 PASS
            if self.is_free_turn:
                return {"error": "自由回合必须出牌"}
            self.pass_count += 1
            self.recent_actions[self.current_player] = ['Pass']
        else:
            if not all(card in player.hand for card in selected_cards):
                return {"error": "出牌不在手牌中"}

            if not self.rules.is_valid_play(selected_cards):
                return {"error": "出牌不合法"}

            if not self.can_beat(self.map_cards_to_action(selected_cards, M, self.config.active_level),
                                 self.map_cards_to_action(self.last_play, M, self.config.active_level)):
                return {"error": "不能压过上家"}

            for card in selected_cards:
                player.hand.remove(card)
                player.played_cards.append(card)

            self.last_play = selected_cards
            self.last_player = self.current_player
            self.recent_actions[self.current_player] = selected_cards
            if not player.hand:
                self.ranking.append(self.current_player)
                if len(self.ranking) <= 2:
                    self.jiefeng = True
            self.pass_count = 0
            if self.is_free_turn:
                self.is_free_turn = False

        self.current_player = (self.current_player + 1) % 4
        self.maybe_reset_turn()
        return {"success": True, "game_over": self.check_game_over()}

    def step(self):
        """推进一步（仅用于非用户玩家），返回字典说明状态"""
        if self.is_game_over:
            return {"game_over": True}

        if self.current_player == self.user_player:
            self.actor_play(self.players[self.user_player])
            self.maybe_reset_turn()
            return {"waiting_for_user": True}

        # 处理 AI 或其他自动玩家的出牌
        self.play_turn()
        self.maybe_reset_turn()

        # 如果刚好出完最后一张牌并结束
        if self.is_game_over:
            return {"game_over": True}

        # 如果下一个轮到用户，告诉前端等待
        if self.current_player == self.user_player:
            return {"waiting_for_user": True}

        # 否则仍轮到 AI，下次前端可继续调用 step
        return {"next_step_needed": True}

    def get_player_statuses(self):
        """
        返回每位玩家的状态，用于前端显示：
        [
            {'id': 1, 'hand_size': 15, 'last_play': ['红桃3', '黑桃3']},
            ...
        ]
        """
        result = []
        for i, player in enumerate(self.players):
            result.append({
                "id": i + 1,
                "hand_size": len(player.hand),
                "last_play": player.last_played_cards
            })
        return result

    def get_game_state(self):
        """获取游戏的完整可视状态字典，供前端展示"""
        return {
            "user_hand": self.players[self.user_player].hand,
            "last_play": self.last_play,
            "current_player": self.current_player,
            "history": self.history,
            "ai_suggestions": self.get_ai_suggestions(),
            "ranking": self.ranking,
            "is_game_over": self.is_game_over,
            "level_rank": self.config.active_level,
            "recent_actions": self.recent_actions
        }
