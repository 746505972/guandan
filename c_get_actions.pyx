# 2025/4/17 10:22
from collections import defaultdict

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

def enumerate_colorful_actions(action, hand, level_rank: int):
    point_to_cards = parse_hand_with_level(hand, level_rank)
    raw_combos = find_combinations(action['points'], point_to_cards)

    # 去重（不关心顺序的重复组合）
    seen = set()
    unique_combos = []
    for combo in raw_combos:
        key = frozenset(combo)
        if key not in seen:
            seen.add(key)
            unique_combos.append(combo)
    return unique_combos


# 示例测试
sample_hand = ['红桃3', '红桃10', '黑桃10','梅花10', '梅花7', '红桃7']
sample_action = {'type': 'pair', 'points': [15, 15]}
print(enumerate_colorful_actions(sample_action, sample_hand, level_rank=3))
