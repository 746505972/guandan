import json
from itertools import combinations, groupby

# 定义点数及其大小
RANK_ORDER = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# 对应点数（原始点数）
CARD_RANKS = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13,
              'A': 14, '小王': 16, '大王': 17}

# 构建所有牌点（两副牌，每种牌点有4张，王有2张）
def all_ranks():
    ranks = []
    for r in RANK_ORDER:
        ranks.extend([CARD_RANKS[r]] * 8)  # 两副牌，每点数8张
    ranks.extend([CARD_RANKS['小王']] * 2)
    ranks.extend([CARD_RANKS['大王']] * 2)
    return ranks

def generate_singles(level_rank=15):
    singles = [{'type': 'single', 'points': [pt], 'logic_point': pt} for pt in range(2, 15)]
    # 万能牌：级牌打出，视为点数 15
    singles.append({'type': 'single', 'points': [15], 'suit': '非红桃','logic_point': 15})
    # 红桃级牌单独打出（非万能，视为普通点数）
    singles.append({'type': 'single', 'points': [level_rank], 'suit': '红桃', 'logic_point': level_rank})
    singles.append({'type': 'single', 'points': [16], 'logic_point': 16})
    singles.append({'type': 'single', 'points': [17], 'logic_point': 17})
    return singles

def generate_pairs():
    pairs = [{'type': 'pair', 'points': [pt]*2, 'logic_point': pt} for pt in range(2, 15)]
    pairs.append({'type': 'pair', 'points': [15]*2, 'suit': '非红桃','logic_point': 15})
    pairs.append({'type': 'pair', 'points': [15] * 2, 'suit': '红桃', 'logic_point': 15})
    pairs.append({'type': 'pair', 'points': [16] * 2,'logic_point': 16})
    pairs.append({'type': 'pair', 'points': [17] * 2,'logic_point': 17})
    return pairs

def generate_triples():
    return [{'type': 'triple', 'points': [pt]*3, 'logic_point': pt} for pt in range(2, 16)]


# 顺子（5张，考虑A=1或A=14）
def generate_straights():
    straights = [{
            'type': 'straight',
            'points': [2,3,4,5,14],
            'logic_point': 1,
            'a_as': 'low'
        }]
    for i in range(len(RANK_ORDER) - 4):
        ranks = RANK_ORDER[i:i+5]
        points = [CARD_RANKS[r] for r in ranks]
        straights.append({
            'type': 'straight',
            'points': points,
            'logic_point': points[0],
            'a_as': 'high'
        })
    return straights

# 连对（三连）
def generate_pairs_chain():
    chains = [{
            'type': 'pair_chain',
            'points': [2,2,3,3,14,14],
            'logic_point': 1,
            'a_as': 'low'
        }]
    for i in range(len(RANK_ORDER) - 2):
        ranks = RANK_ORDER[i:i+3]
        points = sum([[CARD_RANKS[r]] * 2 for r in ranks], [])
        chains.append({
            'type': 'pair_chain',
            'points': points,
            'logic_point': points[0],
            'a_as': 'high'
        })
    return chains

# 钢板（三连对）：两个相连的三同张（共6张）
def generate_gangban():
    gangban = [{
            'type': 'gangban',
            'points': [2,2,2,14,14,14],
            'logic_point': 1,
            'a_as': 'low'
        }]
    for i in range(len(RANK_ORDER) - 1):
        ranks = RANK_ORDER[i:i+2]
        points = sum([[CARD_RANKS[r]] * 3 for r in ranks], [])
        gangban.append({
            'type': 'gangban',
            'points': points,
            'logic_point': points[0],
            'a_as': 'high'
        })
    return gangban

# 炸弹（4~8张）
def generate_bombs():
    points = range(2, 16)
    bombs = []
    for count in range(4, 9):
        for pt in points:
            bombs.append({
                'type': f'{count}_bomb',
                'points': [pt] * count,
                'logic_point': pt
            })
    # 天王炸（4王，最大）
    bombs.append({
        'type': 'joker_bomb',
        'points': [16, 16, 17, 17]  # 小王小王大王大王
        # 不加 logic_point，永远最大
    })
    return bombs

# 三带二（主牌为三张，同点数；副牌为任意对子）
def generate_three_with_pair():
    actions = []
    for triple in range(2, 16):  # 主牌（不能是王）
        for pair in range(2, 18):  # 副牌（对子可以为 2 ~ 大王）
            if triple != pair:
                actions.append({
                    'type': 'three_with_pair',
                    'points': [triple] * 3 + [pair] * 2,
                    'logic_point': triple
                })
    return actions

# 同花顺（5张顺子 + 花色相同，但这里只保留点数结构；判断花色留给出牌时）
def generate_flush_rockets():
    flushes = []
    flushes.append({
        'type': 'flush_rocket',
        'points': [2, 3, 4, 5, 14],
        'logic_point': 1,
        'a_as': 'low'
    })
    for i in range(len(RANK_ORDER) - 4):
        ranks = RANK_ORDER[i:i+5]
        points = [CARD_RANKS[r] for r in ranks]
        flushes.append({
            'type': 'flush_rocket',
            'points': points,
            'logic_point': points[0],
            'a_as': 'high'
        })
    return flushes


# 逢人配（不枚举，战时生成）
def generate_wildcards_placeholder():
    return [{'type': 'wildcard_note', 'desc': '红桃级牌可替代其他牌参与组合，不预先枚举'}]

# 主函数
if __name__ == "__main__":
    total = [{"type":"pass","points":[],"logic_point":0},{"type":"None","points":[],"logic_point":0}]
    total += generate_singles()
    total += generate_pairs()
    total += generate_triples()
    total += generate_bombs()
    total += generate_three_with_pair()
    total += generate_straights()
    total += generate_pairs_chain()
    total += generate_gangban()
    total += generate_flush_rockets()
    #total += generate_wildcards_placeholder()
    for idx, action in enumerate(total):
        action['id'] = idx
    print(f"共生成动作：{len(total)} 个")
    # 标准 JSON 文件
    with open("doudizhu_actions.json", "w", encoding="utf-8") as f:
        json.dump(total, f, ensure_ascii=False, separators=(',', ':'))

    # 调试用的逐行 JSONL 文件
    with open("actions.jsonl", "w", encoding="utf-8") as f:
        for entry in total:
            f.write(json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n')
