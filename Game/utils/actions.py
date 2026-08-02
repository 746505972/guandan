"""动作空间：加载、枚举、映射"""
import json
from collections import defaultdict
from itertools import combinations, product

from Game.constants import SUITS, RANKS, CARD_RANKS, DATA_DIR

# 模块级缓存
_actions_cache = None
_actions_id_dict = None


def load_actions():
    """加载动作空间 JSON，返回 (M, M_id_dict)"""
    global _actions_cache, _actions_id_dict
    if _actions_cache is not None:
        return _actions_cache, _actions_id_dict
    path = DATA_DIR / "actions.json"

    with open(path, "r", encoding="utf-8") as f:
        _actions_cache = json.load(f)
    _actions_id_dict = {a['id']: a for a in _actions_cache}
    return _actions_cache, _actions_id_dict


def get_action_dim():
    M, _ = load_actions()
    return len(M)


def find_entry_by_id(data, target_id):
    """返回匹配 id 的 JSON 对象"""
    for entry in data:
        if entry.get("id") == target_id:
            return entry
    return None


# ============ 手牌解析与组合枚举 ============

def parse_hand(hand):
    """将手牌转换为点数到花色牌组的映射"""
    point_to_cards = defaultdict(list)
    for card in hand:
        for rank in RANKS + ['小王', '大王']:
            if rank in card:
                point = CARD_RANKS[rank]
                point_to_cards[point].append(card)
                break
    return point_to_cards


def parse_hand_with_level(hand, level_rank: int):
    """将手牌转换为 point_to_cards 映射（级牌映射为15点）"""
    point_to_cards = defaultdict(list)
    for card in hand:
        for rank in RANKS + ['小王', '大王']:
            if rank in card:
                raw_point = CARD_RANKS[rank]
                logic_point = 15 if raw_point == level_rank else raw_point
                point_to_cards[logic_point].append(card)
                break
    return point_to_cards


def find_combinations(points, point_to_cards):
    """递归回溯找出所有不重复使用牌的组合"""
    results = []

    card_pool = []
    card_to_idx = {}
    idx = 0
    for p, cards in point_to_cards.items():
        for c in cards:
            card_pool.append((p, c))
            card_to_idx.setdefault(p, []).append(idx)
            idx += 1

    def backtrack(index, path, used_idx):
        if index == len(points):
            results.append([card_pool[i][1] for i in path])
            return
        p = points[index]
        available = [i for i in card_to_idx.get(p, []) if i not in used_idx]
        if not available:
            return
        for i in available:
            used_idx.add(i)
            path.append(i)
            backtrack(index + 1, path, used_idx)
            path.pop()
            used_idx.remove(i)

    backtrack(0, [], set())
    return results


def group_points(points):
    counts = defaultdict(int)
    for p in points:
        counts[p] += 1
    return dict(counts)


def match_structured_action(points, point_to_cards):
    """匹配结构化牌型（三带二、连对、钢板）"""
    grouped = group_points(points)
    group_by_count = defaultdict(list)
    for pt, cnt in grouped.items():
        group_by_count[cnt].append(pt)

    all_combos = []

    # 三带二：3+2
    if set(grouped.values()) == {3, 2} and len(grouped) == 2:
        triples = group_by_count[3]
        pairs = group_by_count[2]
        for triple_point in triples:
            for triple_cards in combinations(point_to_cards.get(triple_point, []), 3):
                for pair_point in pairs:
                    if pair_point == triple_point:
                        continue
                    for pair_cards in combinations(point_to_cards.get(pair_point, []), 2):
                        all_combos.append(list(triple_cards) + list(pair_cards))

    # 连对：3个连续点数，每个2张
    elif all(cnt == 2 for cnt in grouped.values()) and len(grouped) >= 3:
        seq = sorted(grouped.keys())
        if all(seq[i + 1] - seq[i] == 1 for i in range(len(seq) - 1)):
            pair_options = []
            for pt in seq:
                pair_options.append(list(combinations(point_to_cards.get(pt, []), 2)))
            for pairs in product(*pair_options):
                combo = [card for pair in pairs for card in pair]
                all_combos.append(combo)

    # 钢板：2个连续点数，每个3张
    elif all(cnt == 3 for cnt in grouped.values()) and len(grouped) == 2:
        seq = sorted(grouped.keys())
        if seq[1] - seq[0] == 1:
            triple_options = []
            for pt in seq:
                triple_options.append(list(combinations(point_to_cards.get(pt, []), 3)))
            for triples in product(*triple_options):
                combo = [card for trip in triples for card in trip]
                all_combos.append(combo)

    return all_combos


def enumerate_colorful_actions(action, hand, level_rank: int):
    """枚举动作的所有合法带花色出牌组合"""
    point_to_cards = parse_hand_with_level(hand, level_rank)
    raw_combos = []

    structured_combos = match_structured_action(action['points'], point_to_cards)
    raw_combos.extend(structured_combos)

    if not structured_combos:
        raw_combos = find_combinations(action['points'], point_to_cards)

    if action['type'] == 'flush_rocket':
        filtered_combos = []
        for combo in raw_combos:
            suits = [card[:2] for card in combo]
            if all(s == suits[0] for s in suits):
                filtered_combos.append(combo)
        raw_combos = filtered_combos

    seen = set()
    unique_combos = []
    for combo in raw_combos:
        key = frozenset(combo)
        if key not in seen:
            seen.add(key)
            unique_combos.append(combo)
    return unique_combos


# ============ 牌面到结构动作映射 ============

def map_cards_to_action(cards, M, level_rank):
    """从实际出过的牌（带花色）判断其结构动作"""
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
        for s in SUITS:
            if card.startswith(s):
                suits.add(s)
                break

    logic_points = []
    for pt, count in sorted(point_count.items()):
        logic_points.extend([pt] * count)

    # 同花顺检测
    if len(cards) == 5 and len(point_count) == 5:
        sorted_points = sorted(point_count.keys())
        if all(sorted_points[i + 1] - sorted_points[i] == 1 for i in range(4)):
            if len(suits) == 1:
                for action in M:
                    if action['type'] == 'flush_rocket' and sorted(action['points']) == sorted_points:
                        return action

    # 普通结构匹配
    for action in M:
        if sorted(action['points']) == sorted(logic_points):
            return action

    return None


# ============ 结构动作压制判断 ============

def can_beat_action(curr_action, prev_action):
    """判断结构动作 curr_action 是否能压过 prev_action"""
    if prev_action["type"] == "None":
        if curr_action["type"] == "None":
            return False
        else:
            return True

    curr_type = curr_action["type"]
    prev_type = prev_action["type"]

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

    if is_curr_bomb and not is_prev_bomb:
        return True
    if not is_curr_bomb and is_prev_bomb:
        return False

    if is_curr_bomb and is_prev_bomb:
        if bomb_power[curr_type] > bomb_power[prev_type]:
            return True
        elif bomb_power[curr_type] < bomb_power[prev_type]:
            return False
        else:
            return curr_action["logic_point"] > prev_action["logic_point"]

    if curr_type != prev_type:
        return False

    return curr_action["logic_point"] > prev_action["logic_point"]
