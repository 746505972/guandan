"""掼蛋游戏规则"""
from collections import Counter
from Game.constants import CARD_RANKS, RANKS

try:
    from c_rule import Rules as CRules
    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False


class Rules:
    """掼蛋规则判定"""

    def __init__(self, level_card=None):
        self.level_card = level_card
        self.CARD_RANKS = CARD_RANKS
        self.RANKS = RANKS

    def is_valid_play(self, cards):
        """判断出牌是否合法"""
        if not cards:
            return False
        length = len(cards)

        if length == 1:
            return True
        if length == 2:
            return self.is_pair(cards)
        if length == 3:
            return self.is_triple(cards)
        if length == 4:
            return self.is_king_bomb(cards) or self.is_bomb(cards)
        if length == 5:
            return self.is_straight(cards) or self.is_flush_straight(cards) or self.is_three_with_two(cards) or self.is_bomb(cards)
        if length == 6:
            return self.is_triple_pair(cards) or self.is_triple_consecutive(cards) or self.is_bomb(cards)
        if 6 < length <= 8:
            return self.is_bomb(cards)
        return False

    def is_pair(self, cards):
        """对子"""
        return len(cards) == 2 and self.get_rank(cards[0]) == self.get_rank(cards[1])

    def is_triple(self, cards):
        """三同张"""
        return len(cards) == 3 and len(set(self.get_rank(card) for card in cards)) == 1

    def is_three_with_two(self, cards):
        """三带二"""
        if len(cards) != 5:
            return False
        counts = Counter(self.get_rank(card) for card in cards)
        return 3 in counts.values() and 2 in counts.values()

    def is_triple_pair(self, cards):
        """连对（木板），如 556677"""
        if len(cards) != 6:
            return False
        ranks = [self.get_rank(card, as_one=False) for card in cards]
        ranks_as_one = [self.get_rank(card, as_one=True) for card in cards]
        counts = Counter(ranks)
        counts_as_one = Counter(ranks_as_one)
        pairs = sorted([rank for rank, count in counts.items() if count == 2])
        pairs_as_one = sorted([rank for rank, count in counts_as_one.items() if count == 2])
        return (len(pairs) == 3 and self._is_consecutive(pairs)) or \
            (len(pairs_as_one) == 3 and self._is_consecutive(pairs_as_one))

    def is_triple_consecutive(self, cards):
        """三同连张（钢板），如 555666"""
        if len(cards) != 6:
            return False
        ranks = [self.get_rank(card, as_one=False) for card in cards]
        ranks_as_one = [self.get_rank(card, as_one=True) for card in cards]
        counts = Counter(ranks)
        counts_as_one = Counter(ranks_as_one)
        triples = sorted([rank for rank, count in counts.items() if count == 3])
        triples_as_one = sorted([rank for rank, count in counts_as_one.items() if count == 3])
        return (len(triples) == 2 and self._is_consecutive(triples)) or \
            (len(triples_as_one) == 2 and self._is_consecutive(triples_as_one))

    def is_straight(self, cards):
        """顺子（5张，A可作为1或14）"""
        if len(cards) != 5:
            return False
        ranks = sorted(self.get_rank(card, as_one=False) for card in cards)
        ranks_as_one = sorted(self.get_rank(card, as_one=True) for card in cards)
        return self._is_consecutive(ranks) or self._is_consecutive(ranks_as_one)

    def is_flush_straight(self, cards):
        """同花顺（火箭）"""
        if len(cards) != 5:
            return False
        ranks = sorted(self.get_rank(card, as_one=False) for card in cards)
        ranks_as_one = sorted(self.get_rank(card, as_one=True) for card in cards)
        suits = {card[:2] for card in cards}
        return len(suits) == 1 and (self._is_consecutive(ranks) or self._is_consecutive(ranks_as_one))

    def is_bomb(self, cards):
        """炸弹（4张及以上相同牌）"""
        if len(cards) < 4:
            return False
        ranks = [self.get_rank(card) for card in cards]
        return len(set(ranks)) == 1

    def is_king_bomb(self, cards):
        """天王炸"""
        return sorted(cards) == ['大王', '大王', '小王', '小王']

    def get_rank(self, card, as_one=False):
        """获取牌的点数，支持 A=1"""
        if card in ['小王', '大王']:
            return CARD_RANKS[card]

        rank = card[2:] if len(card) > 2 else card[2]

        if rank == RANKS[self.level_card - 2]:
            return CARD_RANKS['A'] + 1

        if as_one and rank == 'A':
            return 1

        return CARD_RANKS.get(rank, 0)

    def _is_consecutive(self, ranks):
        """判断是否为连续数字序列"""
        return all(ranks[i] == ranks[i - 1] + 1 for i in range(1, len(ranks)))

    def can_beat(self, previous_play, current_play):
        """判断当前出牌是否能压过上家"""
        if not self.is_valid_play(current_play):
            return False
        if not previous_play:
            return True

        prev_type = self.get_play_type(previous_play)
        curr_type = self.get_play_type(current_play)

        bomb_order = ['天王炸', '8炸', '7炸', '6炸', '同花顺', '5炸', '4炸']

        if curr_type in bomb_order and prev_type not in bomb_order:
            return True
        if prev_type in bomb_order and curr_type in bomb_order:
            return bomb_order.index(curr_type) < bomb_order.index(prev_type)

        if prev_type != curr_type:
            return False

        return self.get_play_value(current_play) > self.get_play_value(previous_play)

    def get_play_type(self, cards):
        """获取牌型"""
        if self.is_king_bomb(cards):
            return '天王炸'
        if self.is_flush_straight(cards):
            return '同花顺'
        if self.is_bomb(cards):
            size = len(cards)
            if size == 4:
                return '4炸'
            elif size == 5:
                return '5炸'
            elif size == 6:
                return '6炸'
            elif size == 7:
                return '7炸'
            elif size == 8:
                return '8炸'
        if self.is_triple_consecutive(cards):
            return '钢板'
        if self.is_triple_pair(cards):
            return '木板'
        if self.is_three_with_two(cards):
            return '三带二'
        if self.is_triple(cards):
            return '三同张'
        if self.is_straight(cards):
            return '顺子'
        if self.is_pair(cards):
            return '对子'
        if len(cards) == 1:
            return '单牌'
        return '非法牌型'

    def get_play_value(self, cards):
        """获取牌点数"""
        ranks = [self.get_rank(card) for card in cards]
        return max(ranks)
