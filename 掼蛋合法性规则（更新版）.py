from collections import Counter

# 定义牌的点数
CARD_RANKS = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    '小王': 16, '大王': 17
}

class Rules:
    def __init__(self):
        # 两队级牌，初始都为2
        self.team_level_cards = {
            'team1': '2',
            'team2': '2'
        }
        # 跟踪是否打过A级
        self.passed_a = {
            'team1': False,
            'team2': False
        }
        # 记录首次达到A级的时间
        self.reached_a = {
            'team1': None,
            'team2': None
        }
        # 记录达到A级后的失败次数
        self.fail_count_after_a = {
            'team1': 0,
            'team2': 0
        }

    def reset_level(self, team):
        """将指定队伍的级牌重置为2"""
        self.team_level_cards[team] = '2'
        self.passed_a[team] = False
        self.reached_a[team] = None
        self.fail_count_after_a[team] = 0

    def is_valid_play(self, cards):
        """判断出牌是否合法"""
        if not cards:
            return False
        length = len(cards)

        if length == 1:
            return True  # 单张
        if length == 2:
            return self.is_pair(cards)  # 只保留对子
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

    def is_rocket(self, cards):
        """王炸"""
        return set(cards) == {'小王', '大王'}

    def is_king_bomb(self, cards):
        """四大天王（天王炸）"""
        return sorted(cards) == ['大王', '大王', '小王', '小王']

    def get_rank(self, card, as_one=False):
        """获取牌的点数，支持 A=1"""
        if card in ['小王', '大王']:
            return CARD_RANKS[card]

        rank = card[2:] if len(card) > 2 else card[2]

        if as_one and rank == 'A':
            return 1  # A 作为 1

        # 检查是否是任一队伍的级牌
        for team, level in self.team_level_cards.items():
            if level in rank:
                return CARD_RANKS['A'] + 1  # 级牌比 A 大

        return CARD_RANKS.get(rank, 0)

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

    def has_passed_a(self, team):
        """检查某队是否打过A级"""
        return self.passed_a[team]

    def update_level_card(self, finished_players):
        """根据游戏结果更新级牌
        finished_players: 按排名排序的玩家列表，0为头游，3为末游
        """
        # 确定获胜队伍
        head_player = finished_players[0]
        teammate = (head_player + 2) % 4  # 对家是同伴
        winning_team = 'team1' if head_player in [0, 2] else 'team2'
        
        # 获取同伴的排名
        teammate_rank = finished_players.index(teammate)
        
        # 根据同伴排名确定升级数
        if teammate_rank == 1:  # 二游
            upgrade = 3
        elif teammate_rank == 2:  # 三游
            upgrade = 2
        else:  # 末游
            upgrade = 1
            
        # 更新获胜队伍的级牌
        current_level = int(self.team_level_cards[winning_team])
        new_level = current_level + upgrade
        
        # 检查是否打过A级
        if current_level <= 14 and new_level > 14:
            self.passed_a[winning_team] = True
            
        if new_level > 14:  # 最大到A
            new_level = 14
        self.team_level_cards[winning_team] = str(new_level)




if __name__ == "__main__":
    current_round = 2
    rules = Rules(level_card=str(current_round))



    print(rules.is_valid_play(['黑桃6', '红桃2', '方块7', '梅花9', '黑桃7', '黑桃9']))
    '''
    prev_type = rules.get_play_type(['黑桃6', '红桃6', '方块6', '梅花6', '黑桃6'])  # 5炸
    curr_type = rules.get_play_type(['黑桃10', '黑桃J', '黑桃Q', '黑桃K', '黑桃A'])  # 同花顺

    bomb_order = ['天王炸', '8炸', '7炸', '6炸', '同花顺', '5炸', '4炸']
    print(f"5炸排名: {bomb_order.index(prev_type)}, 同花顺排名: {bomb_order.index(curr_type)}")
    print(rules.get_play_type(['黑桃10', '黑桃J', '黑桃Q', '黑桃K', '黑桃A']))  # ✅ True（同花顺）
    print(rules.get_play_type(['黑桃6', '红桃6', '方块6', '梅花6', '黑桃6']))
    print(rules.get_play_type(['红桃6', '方块6', '梅花6', '黑桃6', '红桃6', '黑桃6']))

    # ✅ 正确识别同花顺
    print(rules.get_play_type(['黑桃10', '黑桃J', '黑桃Q', '黑桃K', '黑桃A']))  # ✅ True（同花顺）
    print(rules.get_play_type(['黑桃6', '红桃6', '方块6', '梅花6', '黑桃6']))
    print(rules.get_play_type(['红桃6', '方块6', '梅花6', '黑桃6', '红桃6', '黑桃6']))

    print(rules.can_beat(['黑桃6', '红桃6', '方块6', '梅花6', '黑桃6'],
                         ['黑桃10', '黑桃J', '黑桃Q', '黑桃K', '黑桃A']))
    print(rules.is_flush_straight(['红桃A', '红桃2', '红桃3', '红桃4', '红桃5']))
    # ✅ 5 炸 vs 同花顺（同花顺应当更大）
 # ✅ True（同花顺 > 5炸）

    # ✅ 6 炸 vs 同花顺（6 炸应当更大）
    print(rules.can_beat(['黑桃10', '黑桃J', '黑桃Q', '黑桃K', '黑桃A'],
                         ['黑桃6', '红桃6', '方块6', '梅花6', '黑桃6', '红桃6']))  # ✅ True（6炸 > 同花顺）
    '''
