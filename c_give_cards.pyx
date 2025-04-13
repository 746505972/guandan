import random

# 定义牌型
SUITS = ['黑桃', '红桃', '梅花', '方块']
#SUITS = ['♠','♥','♦','♣']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# 定义牌的点数
CARD_RANKS = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    '小王': 16, '大王': 17
}

# 创建两副牌
def create_deck():
    deck = []
    for _ in range(2):  # 两副牌
        for suit in SUITS:
            for rank in RANKS:
                card = f"{suit}{rank}"
                deck.append(card)
        # 添加大小王
        deck.append('小王')
        deck.append('大王')
    return deck

# 洗牌
def shuffle_deck(deck):
    random.shuffle(deck)
    return deck
# 发牌
def deal_cards(deck):
    players = [[], [], [], []]  # 4个玩家
    for i in range(len(deck)):
        players[i % 4].append(deck[i])
    return players


