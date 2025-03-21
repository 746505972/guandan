# 2025/3/16 17:16
from give_cards import create_deck, shuffle_deck, deal_cards
from liu_rule import Rules
import random


class GuandanGame:
    def __init__(self, level_card=None, user_player=None):
        self.rules = Rules()  # 初始化Rules
        self.players = deal_cards(shuffle_deck(create_deck()))  # 调用 `give_cards.py`
        self.current_player = 0  # 当前出牌玩家
        self.last_play = None  # 记录上一手牌
        self.last_player = -1  # 记录上一手是谁出的
        self.pass_count = 0  # 记录连续 Pass 的次数
        self.user_player = user_player - 1 if user_player else None  # 转换为索引（0~3）
        self.finished_players = []  # 记录已出完牌的玩家

        # **手牌排序**
        for i in range(4):
            self.players[i] = self.sort_cards(self.players[i])

    def sort_cards(self, cards):
        """按牌的大小排序（从大到小）"""
        return sorted(cards, key=lambda card: self.rules.get_rank(card), reverse=True)

    def get_possible_moves(self, player_hand):
        """获取玩家所有可能的出牌"""
        possible_moves = []
        for size in [1, 2, 3, 4, 5, 6, 7, 8]:
            for i in range(len(player_hand) - size + 1):
                move = player_hand[i:i + size]
                if self.rules.is_valid_play(move):
                    possible_moves.append(move)
        return possible_moves

    def play_turn(self):
        """当前玩家尝试出牌"""
        # 跳过已出完牌的玩家
        while self.current_player in self.finished_players:
            self.current_player = (self.current_player + 1) % 4

        player_hand = self.players[self.current_player]

        if self.user_player == self.current_player:
            self.show_user_hand()

        # 计算剩余玩家数量
        remaining_players = 4 - len(self.finished_players)
        if self.pass_count >= remaining_players - 1:
            print(f"\n🆕 所有玩家Pass，本轮重置！玩家 {self.current_player + 1} 可以自由出牌。\n")
            self.last_play = None
            self.pass_count = 0

        if self.user_player == self.current_player:
            return self.user_play(player_hand)

        return self.ai_play(player_hand)

    def ai_play(self, player_hand):
        """AI 出牌逻辑（随机选择合法且能压过上家的出牌）"""

        possible_moves = self.get_possible_moves(player_hand)
        possible_moves = [move for move in possible_moves
                          if not self.last_play or self.rules.can_beat(self.last_play, move)]

        if not possible_moves:
            # 跳过已出完牌的玩家
            if self.current_player not in self.finished_players:
                print(f"玩家 {self.current_player + 1} Pass")
                self.pass_count += 1
        else:
            chosen_move = random.choice(possible_moves)
            self.last_play = chosen_move
            self.last_player = self.current_player
            for card in chosen_move:
                player_hand.remove(card)
            print(f"玩家 {self.current_player + 1} 出牌: {' '.join(chosen_move)}")

            if not player_hand:
                print(f"\n🎉 玩家 {self.current_player + 1} 出完所有牌！\n")
                self.finished_players.append(self.current_player)

                # 检查是否两个对家已出完
                if len(self.finished_players) >= 2:
                    # 检查是否是队友
                    teammate = (self.current_player + 2) % 4
                    if teammate in self.finished_players:
                        return True

                if len(self.finished_players) == 3:
                    return True
                self.check_jiefeng()
                return False

            self.pass_count = 0

        # 找到下一个未出完牌的玩家
        next_player = (self.current_player + 1) % 4
        while next_player in self.finished_players:
            next_player = (next_player + 1) % 4
        self.current_player = next_player
        return False

    def user_play(self, player_hand):
        """让用户手动选择出牌"""
        while True:
            self.show_user_hand()
            choice = input("\n请选择要出的牌（用空格分隔），或直接回车跳过： ").strip()

            if choice == "":  # **回车等同于 pass**
                if self.current_player not in self.finished_players:
                    print(f"玩家 {self.current_player + 1} Pass")
                    self.pass_count += 1
                break

            if choice.lower() == "pass":
                if self.current_player not in self.finished_players:
                    print(f"玩家 {self.current_player + 1} Pass")
                    self.pass_count += 1
                break

            selected_cards = choice.split()
            if all(card in player_hand for card in selected_cards) and self.rules.is_valid_play(selected_cards):
                if self.last_play is None or self.rules.can_beat(self.last_play, selected_cards):
                    for card in selected_cards:
                        player_hand.remove(card)
                    self.last_play = selected_cards
                    self.last_player = self.current_player
                    print(f"玩家 {self.current_player + 1} 出牌: {' '.join(selected_cards)}")

                    if not player_hand:
                        print(f"\n🎉 你出完所有牌！\n")
                        self.finished_players.append(self.current_player)
                        if len(self.finished_players) == 3:
                            return True
                        self.check_jiefeng()
                        return False

                    self.pass_count = 0
                    break
                else:
                    print("❌ 你出的牌不能压过上一手牌，请重新选择！")
            else:
                print("❌ 你的输入无效，请确保牌在你的手牌中并符合规则！")

        self.current_player = (self.current_player + 1) % 4
        return False

    def show_user_hand(self):
        """显示用户手牌（按排序后的顺序）"""
        sorted_hand = self.sort_cards(self.players[self.user_player])
        print("\n🃏 你的手牌：", " ".join(sorted_hand))
        if self.last_play:
            print(f"🃏 场上最新出牌：{' '.join(self.last_play)}\n")

    def check_jiefeng(self):
        """检查是否需要接风"""
        if self.last_player in self.finished_players:
            # 先让其他玩家尝试压牌
            for i in range(4):
                if i not in self.finished_players and i != self.current_player:
                    # 检查是否有牌能压过
                    if any(self.rules.can_beat(self.last_play, move)
                           for move in self.get_possible_moves(self.players[i])):
                        return

            # 如果所有玩家都pass，则由对家接风
            teammate = (self.last_player + 2) % 4
            if teammate not in self.finished_players:
                print(f"\n🔄 所有玩家Pass，玩家 {self.last_player + 1} 的队友 {teammate + 1} 接风！\n")
                self.current_player = teammate
                self.last_play = None  # 接风玩家可以自由出牌

    def play_game(self):
        """执行一整局游戏"""
        print("🎮 游戏开始！")
        while True:
            if self.play_turn():
                break

        # 确定剩余玩家
        remaining_players = [i for i in range(4) if i not in self.finished_players]
        # 按座位顺序排序
        remaining_players.sort()
        self.finished_players.extend(remaining_players)

        # 显示胜利排名
        print("\n🏆 游戏结束！最终排名：")
        ranks = ["头游", "二游", "三游", "末游"]
        for i, player in enumerate(self.finished_players):
            print(f"{ranks[i]}：玩家 {player + 1}")

        # 更新级牌
        old_levels = self.rules.team_level_cards.copy()
        self.rules.update_level_card(self.finished_players)

        # 显示两队级牌变化
        print("\n级牌更新：")
        for team, level in self.rules.team_level_cards.items():
            old_level = old_levels[team]
            if old_level != level:
                print(f"{team} 级牌从 {old_level} 升级到 {level}")
            else:
                print(f"{team} 级牌保持不变 ({level})")


if __name__ == "__main__":
    user_pos = int(input("请选择你的座位（1~4）："))
    game = GuandanGame(user_player=user_pos)
    game.play_game()
