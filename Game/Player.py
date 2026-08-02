# 2026/8/1 19:35
# 管理玩家信息

class Player:
    def __init__(self,hand: list):
        """
        程序里的玩家是从0开始的，输出时会+1
        """
        self.hand = hand  # 手牌
        self.played_cards = []  # 记录已出的牌
        self.last_played_cards = []
