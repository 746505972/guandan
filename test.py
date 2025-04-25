# 2025/4/19 21:20
import numpy as np
import json
from get_actions import enumerate_colorful_actions, CARD_RANKS, SUITS,RANKS
import random
from collections import Counter,defaultdict
with open("doudizhu_actions.json", "r", encoding="utf-8") as f:
    M = json.load(f)
action_dim = len(M)

# 构建动作映射字典
M_id_dict = {a['id']: a for a in M}
def map_cards_to_action(cards, M, level_rank):
    """
    从实际出过的牌中（带花色），判断其结构动作（含同花顺识别）。
    """
    point_count = defaultdict(int)
    suits = set()

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

def can_beat(curr_action, prev_action):
    """
    判断结构动作 curr_action 是否能压过 prev_action
    """
    # 如果没人出牌，当前动作永远可以出
    if prev_action["type"] == "None":
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




print(map_cards_to_action(['梅花9'],M,6))
print(map_cards_to_action([ '梅花2' ,'红桃3', '黑桃4', '梅花5', '方块A'],M,6))
print(can_beat(map_cards_to_action(['梅花2' ,'红桃3', '黑桃4', '梅花5','方块A'],M,6),
               map_cards_to_action(['梅花9'],M,6)))