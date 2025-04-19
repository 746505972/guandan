# 2025/4/17 10:22
from collections import defaultdict
from itertools import combinations, product
# 定义花色和点数
SUITS = ['黑桃', '红桃', '梅花', '方块']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
CARD_RANKS = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    '小王': 16, '大王': 17
}
RANK_STR = {v: k for k, v in CARD_RANKS.items()}

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

def find_combinations(points, point_to_cards):
    """递归回溯地找出所有不重复使用牌的组合"""
    results = []

    def backtrack(index, path, used_cards):
        if index == len(points):
            results.append(path[:])
            return
        p = points[index]
        available = [c for c in point_to_cards.get(p, []) if c not in used_cards]
        for card in set(available):
            used_cards.add(card)
            path.append(card)
            backtrack(index + 1, path, used_cards)
            path.pop()
            used_cards.remove(card)

    backtrack(0, [], set())
    return results

def parse_hand_with_level(hand, level_rank: int):
    """
    将手牌转换为 point_to_cards 映射：
    - 如果是级牌（点数 == level_rank），在逻辑上映射为 15 点
    """
    point_to_cards = defaultdict(list)
    for card in hand:
        for rank in RANKS + ['小王', '大王']:
            if rank in card:
                raw_point = CARD_RANKS[rank]
                logic_point = 15 if raw_point == level_rank else raw_point
                point_to_cards[logic_point].append(card)
                break
    return point_to_cards

def group_points(points):
    counts = defaultdict(int)
    for p in points:
        counts[p] += 1
    return dict(counts)

def match_structured_action(points, point_to_cards):
    grouped = group_points(points)
    group_by_count = defaultdict(list)
    for pt, cnt in grouped.items():
        group_by_count[cnt].append(pt)

    all_combos = []

    # ✅ 三带二：3+2
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

    # ✅ 连对：3个连续点数，每个2张（如 334455）
    elif all(cnt == 2 for cnt in grouped.values()) and len(grouped) >= 3:
        seq = sorted(grouped.keys())
        if all(seq[i+1] - seq[i] == 1 for i in range(len(seq) - 1)):
            pair_options = []
            for pt in seq:
                pair_options.append(list(combinations(point_to_cards.get(pt, []), 2)))
            for pairs in product(*pair_options):
                combo = [card for pair in pairs for card in pair]
                all_combos.append(combo)

    # ✅ 钢板：2个连续点数，每个3张（如 555666）
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
    point_to_cards = parse_hand_with_level(hand, level_rank)
    raw_combos = []

    # 特判结构型牌型（三带二）
    structured_combos = match_structured_action(action['points'], point_to_cards)
    raw_combos.extend(structured_combos)

    # 普通动作（顺子/对子等）
    if not structured_combos:
        raw_combos = find_combinations(action['points'], point_to_cards)

    # 去重
    seen = set()
    unique_combos = []
    for combo in raw_combos:
        key = frozenset(combo)
        if key not in seen:
            seen.add(key)
            unique_combos.append(combo)
    return unique_combos

'''
# 示例测试
hand1 = ['红桃2', '红桃2','黑桃3', '红桃3', '黑桃4', '红桃4', '红桃4', '黑桃4','黑桃5', '红桃5']
action1 = {'type': 'pair_chain', 'points': [3,3,4,4,5,5]}
print(enumerate_colorful_actions(action1, hand1, level_rank=10))
'''