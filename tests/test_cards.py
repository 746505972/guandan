"""give_cards 及相关牌组函数的单元测试"""
import random
import unittest
from collections import Counter
from unittest.mock import patch

from Game.constants import SUITS, RANKS
from Game.utils.cards import (
    create_deck,
    shuffle_deck,
    deal_cards,
    give_cards,
)


class TestCreateDeck(unittest.TestCase):
    """测试 create_deck"""

    def test_deck_size(self):
        """两副牌共 108 张"""
        deck = create_deck()
        self.assertEqual(len(deck), 108)

    def test_each_card_appears_twice(self):
        """每张普通牌出现 2 次，大小王各 2 张"""
        deck = create_deck()
        counts = Counter(deck)
        for suit in SUITS:
            for rank in RANKS:
                self.assertEqual(counts[f"{suit}{rank}"], 2)
        self.assertEqual(counts['小王'], 2)
        self.assertEqual(counts['大王'], 2)

    def test_unique_card_types(self):
        """牌的种类数 = 4花色×13点数 + 2王 = 54"""
        deck = create_deck()
        self.assertEqual(len(set(deck)), 54)


class TestShuffleDeck(unittest.TestCase):
    """测试 shuffle_deck"""

    def test_shuffle_preserves_elements(self):
        """洗牌前后牌的集合不变"""
        deck = create_deck()
        original = deck.copy()
        shuffled = shuffle_deck(deck)
        self.assertEqual(sorted(shuffled), sorted(original))

    def test_shuffle_returns_same_list(self):
        """shuffle_deck 原地修改并返回同一对象"""
        deck = create_deck()
        result = shuffle_deck(deck)
        self.assertIs(result, deck)

    def test_shuffle_changes_order(self):
        """以固定种子验证洗牌确实打乱了顺序（极低概率失败可忽略）"""
        deck = create_deck()
        original = deck.copy()
        random.seed(42)
        shuffle_deck(deck)
        # 108 张牌洗牌后与原顺序完全一致的概率几乎为 0
        self.assertNotEqual(deck, original)

    def test_shuffle_with_seed_reproducible(self):
        """相同种子产生相同洗牌结果"""
        deck1 = create_deck()
        deck2 = create_deck()
        random.seed(123)
        shuffle_deck(deck1)
        random.seed(123)
        shuffle_deck(deck2)
        self.assertEqual(deck1, deck2)


class TestDealCards(unittest.TestCase):
    """测试 deal_cards"""

    def test_four_players(self):
        """返回 4 个玩家的手牌"""
        deck = create_deck()
        players = deal_cards(deck)
        self.assertEqual(len(players), 4)

    def test_each_player_gets_27_cards(self):
        """108 张牌每人 27 张"""
        deck = create_deck()
        players = deal_cards(deck)
        for hand in players:
            self.assertEqual(len(hand), 27)

    def test_all_cards_distributed(self):
        """所有牌都被分配，无遗漏无重复"""
        deck = create_deck()
        players = deal_cards(deck)
        all_cards = [card for hand in players for card in hand]
        self.assertEqual(Counter(all_cards), Counter(deck))


    def test_empty_deck(self):
        """空牌组不发牌"""
        players = deal_cards([])
        self.assertEqual(players, [[], [], [], []])


class TestGiveCards(unittest.TestCase):
    """测试 give_cards 整合发牌流程"""

    def test_returns_list_of_four(self):
        """返回包含 4 个子列表的列表"""
        result = give_cards()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
        for hand in result:
            self.assertIsInstance(hand, list)

    def test_each_player_27_cards(self):
        """每个玩家获得 27 张牌"""
        result = give_cards()
        for i, hand in enumerate(result):
            self.assertEqual(len(hand), 27, f"玩家{i}手牌数量不正确")

    def test_total_108_cards(self):
        """总共 108 张牌"""
        result = give_cards()
        total = sum(len(hand) for hand in result)
        self.assertEqual(total, 108)

    def test_all_cards_valid(self):
        """所有发出的牌都是合法牌"""
        valid_cards = set(create_deck())
        result = give_cards()
        for hand in result:
            for card in hand:
                self.assertIn(card, valid_cards, f"非法牌: {card}")

    def test_no_missing_or_extra_cards(self):
        """无遗漏无重复：所有牌恰好出现预期次数"""
        result = give_cards()
        all_cards = [card for hand in result for card in hand]
        expected = Counter(create_deck())
        self.assertEqual(Counter(all_cards), expected)

    def test_randomness_across_calls(self):
        """多次调用返回不同的发牌结果（极低概率失败）"""
        results = [give_cards() for _ in range(5)]
        # 比较第一个玩家的手牌，5 次完全相同的概率极低
        first_hands = [tuple(sorted(r[0])) for r in results]
        self.assertGreater(len(set(first_hands)), 1)

    def test_with_mocked_shuffle(self):
        """mock 洗牌为确定性顺序，验证发牌结果"""
        deterministic_deck = create_deck()  # 不洗牌，直接用有序牌组
        with patch('Game.utils.cards.shuffle_deck', return_value=deterministic_deck):
            result = give_cards()
        # 第 0 张 → 玩家0，第 1 张 → 玩家1，...
        expected = deal_cards(deterministic_deck)
        self.assertEqual(result, expected)

    def test_with_mocked_deck(self):
        """mock 整副牌为简单序列，验证分配逻辑"""
        fake_deck = list(range(12))  # 12 张"牌"
        with patch('Game.utils.cards.create_deck', return_value=fake_deck), \
             patch('Game.utils.cards.shuffle_deck', side_effect=lambda d: d):
            result = give_cards()
        self.assertEqual(result[0], [0, 4, 8])
        self.assertEqual(result[1], [1, 5, 9])
        self.assertEqual(result[2], [2, 6, 10])
        self.assertEqual(result[3], [3, 7, 11])


if __name__ == '__main__':
    unittest.main()
