"""掼蛋游戏常量定义"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# 花色
SUITS = ['黑桃', '红桃', '梅花', '方块']

# 点数
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# 点数映射
# 点数15留给级牌
CARD_RANKS = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    '小王': 16, '大王': 17
}

# 反向映射
RANK_STR = {v: k for k, v in CARD_RANKS.items()}

# 牌型中文映射
type_to_chinese = {
    "None": "无",
    "Pass (不出)": "Pass",
    "single": "单张",
    "pair": "对子",
    "triple": "三张",
    "4_bomb": "4炸",
    "5_bomb": "5炸",
    "6_bomb": "6炸",
    "7_bomb": "7炸",
    "8_bomb": "8炸",
    "joker_bomb": "天王炸",
    "three_with_pair": "三带二",
    "straight": "顺子",
    "pair_chain": "连对",
    "gangban": "钢板",
    "flush_rocket": "同花顺"
}
