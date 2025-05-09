# 2025/4/17 10:22
from collections import defaultdict
from itertools import combinations, product
import numpy as np
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
    """递归回溯地找出所有不重复使用牌的组合（通过索引区分同名牌）"""
    results = []

    card_pool = []  # (point, card) 的列表
    card_to_idx = {}  # point ➔ 对应的全局 idx 列表
    idx = 0
    for p, cards in point_to_cards.items():
        for c in cards:
            card_pool.append((p, c))
            card_to_idx.setdefault(p, []).append(idx)
            idx += 1  # 全局编号递增

    def backtrack(index, path, used_idx):
        if index == len(points):
            results.append([card_pool[i][1] for i in path])  # 只保留牌面
            return
        p = points[index]
        available = [i for i in card_to_idx.get(p, []) if i not in used_idx]
        if not available:
            return  # ❌ 找不到需要的牌直接剪枝
        for i in available:
            used_idx.add(i)
            path.append(i)
            backtrack(index + 1, path, used_idx)
            path.pop()
            used_idx.remove(i)

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

    if action['type'] == 'flush_rocket':
        filtered_combos = []
        for combo in raw_combos:
            suits = [card[:2] for card in combo]
            if all(s == suits[0] for s in suits):  # 花色必须完全一致
                filtered_combos.append(combo)
        raw_combos = filtered_combos

    # 去重
    seen = set()
    unique_combos = []
    for combo in raw_combos:
        key = frozenset(combo)
        if key not in seen:
            seen.add(key)
            unique_combos.append(combo)
    return unique_combos

def build_card_index_map():
    index_map = {}
    idx = 0
    for copy in range(2):
        for suit in ['黑桃', '红桃', '梅花', '方块']:
            for rank in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']:
                card = f"{suit}{rank}"
                if card not in index_map:
                    index_map[card] = []
                index_map[card].append(idx)
                idx += 1
    for copy in range(2):
        index_map.setdefault('小王', []).append(idx)
        idx += 1
        index_map.setdefault('大王', []).append(idx)
        idx += 1
    return index_map

def encode_hand_108(hand):
    card_map = build_card_index_map()
    obs = np.zeros(108)
    if hand == ['Pass'] or hand == ['None'] or not hand:
        hand = []

    card_count = {}
    for card in hand:
        card_count[card] = card_count.get(card, 0) + 1

    for card, count in card_count.items():
        indices = card_map.get(card, [])
        for i in range(min(count, len(indices))):
            obs[indices[i]] = 1.0
    return obs

'''
# 示例测试
hand1 = ['红桃2', '红桃2','黑桃3', '红桃3', '黑桃4', '红桃4', '红桃4', '黑桃4','黑桃5', '红桃5','大王','小王','大王']
action1 = {"type":"single","points":[16],"logic_point":16,"id":16}

obs = np.zeros(108)
# 1️⃣ 当前玩家手牌 (108)
obs[:108] = encode_hand_108(hand1)
print(obs)
#mask = torch.tensor(self.get_valid_action_mask(player.hand, M, self.active_level, self.last_play)).unsqueeze(0)
'''

def test_cases():
    level_rank = 9
    hand = ['大王', '大王', '方块8', '黑桃A', '方块K', '方块K', '黑桃K', '梅花Q', '黑桃Q', '方块J', '梅花10', '梅花10', '梅花10', '方块9', '红桃9', '方块7', '梅花7', '黑桃6', '方块6', '红桃6', '红桃5', '方块5', '梅花4', '方块4', '方块3', '红桃2', '方块2']
    a={"type":"straight","points":[10,11,12,13,14],"logic_point":10,"a_as":"high","id":339}
    print(enumerate_colorful_actions(a, hand, level_rank=8))
    # ---------------- 连对（aabbcc）测试 ----------------
    # 测试连对 - 同花色
    hand = ['红桃7', '红桃7', '红桃8', '红桃8', '红桃9', '红桃9']
    a = {"type": "pair_chain", "points": [7, 7, 8, 8, 9, 9], "logic_point": 7, "id": 5}
    print(enumerate_colorful_actions(a, hand, level_rank))

    # 测试连对 - 异花色
    hand = ['红桃7', '黑桃7', '红桃8', '黑桃8', '红桃9', '黑桃9']
    a = {"type": "pair_chain", "points": [7, 7, 8, 8, 9, 9], "logic_point": 7, "id": 6}
    print(enumerate_colorful_actions(a, hand, level_rank))


    # ---------------- 四大天王（四王炸）测试 ----------------
    # 四大天王 - 正确
    hand = ['小王', '小王', '大王', '大王']
    a = {"type": "joker_bomb", "points": [16, 16, 17, 17], "logic_point": 16, "id": 9}
    print(enumerate_colorful_actions(a, hand, level_rank))




