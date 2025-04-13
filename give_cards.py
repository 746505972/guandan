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


# 获取级牌
def get_level_card(current_round):
    # 级牌通常是当前局数的数字（如第2局的级牌是2）
    return str(current_round)

# 手牌排序（从大到小）
def sort_cards(cards, level_card=None):
    def get_card_value(card):
        if card == '大王':
            return (CARD_RANKS['大王'], 0)
        if card == '小王':
            return (CARD_RANKS['小王'], 0)
        rank = card[2:] if len(card) > 2 else card[2]  # 处理10以上的牌
        if level_card and rank == level_card:
            return (CARD_RANKS['A'] + 1, SUITS.index(card[:2]))  # 级牌仅小于大小王
        return (CARD_RANKS.get(rank, 0), SUITS.index(card[:2]))

    return sorted(cards, key=get_card_value, reverse=True)

# 打印玩家手牌
def print_players_cards(players, level_card=None):
    for i, player in enumerate(players):
        print(f"玩家{i+1}的牌：")
        sorted_cards = sort_cards(player, level_card)
        for card in sorted_cards:
            if level_card and level_card in card:
                print(f"*{card}* (级牌)", end=" ")
            else:
                print(card, end=" ")
        print()
        print()

if __name__ == "__main__":
    # 当前局数（假设为第2局）
    current_round = 2
    use_level_card = False  # 是否启用级牌规则
    level_card = get_level_card(current_round) if use_level_card else None

    # 创建牌堆
    deck = create_deck()

    # 洗牌
    shuffled_deck = shuffle_deck(deck)

    # 发牌
    players = deal_cards(shuffled_deck)

    # 打印每个玩家的牌
    print_players_cards(players, level_card)