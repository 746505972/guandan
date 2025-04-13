from collections import Counter

class Rules:
    def __init__(self, level_card=None):
        self.level_card = level_card  # 级牌
        self.CARD_RANKS = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
            '小王': 16, '大王': 17
        }
        self.RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    def is_valid_play(self, cards):
        """判断出牌是否合法"""
        if not cards:
            return False
        length = len(cards)

        if length == 1:
            return True  # 单张
        if length == 2:
            return self.is_pair(cards)  # 对子
        if length == 3:
            return self.is_triple(cards)  # 三同张
        if length == 4:
            return self.is_king_bomb(cards) or self.is_bomb(cards)  # 天王炸 or 4 炸
        if length == 5:
            return self.is_straight(cards) or self.is_flush_straight(cards) or self.is_three_with_two(
                cards) or self.is_bomb(cards)  # 顺子 / 同花顺 / 三带二
        if length == 6:
            return self.is_triple_pair(cards) or self.is_triple_consecutive(cards) or self.is_bomb(cards)  # 连对（木板） / 钢板
        if 6 < length <= 8:
            return self.is_bomb(cards)
        return False  # 其他情况不合法

    def is_pair(self, cards):
        """对子"""
        return len(cards) == 2 and self.get_rank(cards[0]) == self.get_rank(cards[1])

    def is_triple(self, cards):
        """三同张（三不带）"""
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

        # 获取所有牌的点数（去掉花色）
        ranks = [self.get_rank(card, as_one=False) for card in cards]
        ranks_as_one = [self.get_rank(card, as_one=True) for card in cards]

        # 统计点数出现次数
        counts = Counter(ranks)
        counts_as_one = Counter(ranks_as_one)

        # 获取所有 **点数为 2 的对子**
        pairs = sorted([rank for rank, count in counts.items() if count == 2])
        pairs_as_one = sorted([rank for rank, count in counts_as_one.items() if count == 2])

        # 必须有 3 组对子，并且它们的点数是连续的
        return (len(pairs) == 3 and self._is_consecutive(pairs)) or \
            (len(pairs_as_one) == 3 and self._is_consecutive(pairs_as_one))

    def is_triple_consecutive(self, cards):
        """三同连张（钢板），如 555666"""
        if len(cards) != 6:
            return False

        # 获取所有牌的点数（去掉花色）
        ranks = [self.get_rank(card, as_one=False) for card in cards]
        ranks_as_one = [self.get_rank(card, as_one=True) for card in cards]

        # 统计点数出现次数
        counts = Counter(ranks)
        counts_as_one = Counter(ranks_as_one)

        # 获取所有 **点数为 3 的三同张**
        triples = sorted([rank for rank, count in counts.items() if count == 3])
        triples_as_one = sorted([rank for rank, count in counts_as_one.items() if count == 3])

        # 必须有 2 组三同张，并且它们的点数是连续的
        return (len(triples) == 2 and self._is_consecutive(triples)) or \
            (len(triples_as_one) == 2 and self._is_consecutive(triples_as_one))

    def is_straight(self, cards):
        """顺子（必须 5 张，A 可作为 1 或 14）"""
        if len(cards) != 5:
            return False

        # 获取所有牌的点数（去掉花色）
        ranks = sorted(self.get_rank(card, as_one=False) for card in cards)
        ranks_as_one = sorted(self.get_rank(card, as_one=True) for card in cards)

        # 检查 A=1 或 A=14 的情况
        return self._is_consecutive(ranks) or self._is_consecutive(ranks_as_one)

    def is_flush_straight(self, cards):
        """同花顺（火箭），如 ♠10JQKA"""
        if len(cards) != 5:
            return False

        # 获取所有牌的点数（去掉花色）
        ranks = sorted(self.get_rank(card, as_one=False) for card in cards)
        ranks_as_one = sorted(self.get_rank(card, as_one=True) for card in cards)

        # 获取所有牌的花色
        suits = {card[:2] for card in cards}

        # 需要 **同一花色** 且 **顺序正确**
        return len(suits) == 1 and (self._is_consecutive(ranks) or self._is_consecutive(ranks_as_one))

    def is_bomb(self, cards):
        """炸弹（5 张及以上的相同牌 or 4 张相同牌）"""
        if len(cards) < 4:
            return False
        ranks = [self.get_rank(card) for card in cards]
        return len(set(ranks)) == 1


    def is_king_bomb(self, cards):
        """四大天王（天王炸）"""
        return sorted(cards) == ['大王', '大王', '小王', '小王']

    def get_rank(self, card, as_one=False):
        """获取牌的点数，支持 A=1"""
        if card in ['小王', '大王']:
            return self.CARD_RANKS[card]

        rank = card[2:] if len(card) > 2 else card[2]  # 解析点数

        # **只检查当前局的级牌**
        if rank == self.RANKS[self.level_card - 2]:
            return self.CARD_RANKS['A'] + 1  # 级牌比 A 还大

        if as_one and rank == 'A':
            return 1  # A 作为 1

        return self.CARD_RANKS.get(rank, 0)

    def _is_consecutive(self, ranks):
        """判断是否为连续数字序列"""
        return all(ranks[i] == ranks[i - 1] + 1 for i in range(1, len(ranks)))

    def can_beat(self, previous_play, current_play):
        """判断当前出牌是否能压过上家"""
        if not self.is_valid_play(current_play):
            return False
        if not previous_play:
            return True  # 没人出牌，可以随便出

        prev_type = self.get_play_type(previous_play)
        curr_type = self.get_play_type(current_play)


        # **修正炸弹牌力顺序**
        bomb_order = ['天王炸', '8炸', '7炸', '6炸', '同花顺', '5炸', '4炸']

        # **炸弹能压制非炸弹**
        if curr_type in bomb_order and prev_type not in bomb_order:
            return True
        if prev_type in bomb_order and curr_type in bomb_order:
            return bomb_order.index(curr_type) < bomb_order.index(prev_type)

        # **牌型必须相同才能比较**
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