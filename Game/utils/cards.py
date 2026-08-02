"""牌组操作：创建、洗牌、发牌、排序"""
import random
from Game.constants import SUITS, RANKS, CARD_RANKS


def create_deck():
    """创建两副牌"""
    deck = []
    for _ in range(2):
        for suit in SUITS:
            for rank in RANKS:
                card = f"{suit}{rank}"
                deck.append(card)
        deck.append('小王')
        deck.append('大王')
    return deck


def shuffle_deck(deck):
    """洗牌"""
    random.shuffle(deck)
    return deck


def deal_cards(deck):
    """发牌给4个玩家"""
    players = [[], [], [], []]
    for i in range(len(deck)):
        players[i % 4].append(deck[i])
    return players


def give_cards() -> list:
    """发牌给4个玩家，并返回手牌列表"""
    return deal_cards(shuffle_deck(create_deck()))
