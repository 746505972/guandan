# 2025/3/16 17:16
# 2025/3/21 新增级牌升级规则
from give_cards import create_deck, shuffle_deck, deal_cards
from rule import Rules
import random

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

class Player:
    def __init__(self, hand):
        self.hand = hand  # 手牌
        self.played_cards = []  # 记录已出的牌

class GuandanGame:
    def __init__(self, team_levels=None, user_player=None, active_level=2):
        # **两队各自的级牌**
        self.team_levels = team_levels if team_levels else {1: 2, 2: 2}

        # **本局级牌取决于上一局的胜者**
        self.active_level = active_level
        # 历史记录，记录最近 20 轮的出牌情况（每轮包含 4 个玩家的出牌）
        self.history = []
        # **只传当前局的有效级牌**
        self.rules = Rules(self.active_level)
        # self.players = deal_cards(shuffle_deck(create_deck()))
        self.players = [Player(hand) for hand in deal_cards(shuffle_deck(create_deck()))]# 发牌
        self.current_player = 0  # 当前出牌玩家
        self.last_play = None  # 记录上一手牌
        self.last_player = -1  # 记录上一手是谁出的
        self.pass_count = 0  # 记录连续 Pass 的次数
        self.user_player = user_player - 1 if user_player else None  # 转换为索引（0~3）
        self.ranking = []  # 存储出完牌的顺序
        self.recent_actions = {i: [] for i in range(4)}

        # **手牌排序**
        for player in self.players:
            player.hand = self.sort_cards(player.hand)
        '''
        for i in range(4):
            self.players[i] = self.sort_cards(self.players[i])
        '''

    def sort_cards(self, cards):
        """按牌的大小排序（从大到小）"""
        return sorted(cards, key=lambda card: self.rules.get_rank(card), reverse=True)

    def play_turn(self):
        """执行当前玩家的回合"""
        if self.current_player in self.ranking:
            self.current_player = (self.current_player + 1) % 4
            return False

        player = self.players[self.current_player]  # 获取当前玩家对象
        player_hand = player.hand  # 取出手牌

        # **计算当前仍有手牌的玩家数**
        active_players = sum(1 for p in self.players if len(p.hand) > 0)

        # **如果 Pass 的人 == "当前有手牌的玩家数 - 1"，就重置轮次**
        if self.pass_count >= (active_players - 1):
            print(f"\n🆕 {self.pass_count} 人 Pass，轮次重置！玩家 {self.current_player + 1} 可以自由出牌。\n")
            self.last_play = None  # ✅ 允许新的自由出牌
            self.pass_count = 0  # ✅ Pass 计数归零

        # **AI 或用户出牌**
        if self.user_player == self.current_player:
            result = self.user_play(player)
        else:
            result = self.ai_play(player)

        # **记录最近 5 轮历史**
        if self.current_player == 0:
            round_history = [self.recent_actions[i] for i in range(4)]
            self.history.append(round_history)
            if len(self.history) > 20:
                self.history.pop(0)

        return result

    def ai_play(self, player_hand):
        """AI 出牌逻辑（随机选择合法且能压过上家的出牌）"""

        # **如果玩家已经打完，就跳过**
        if self.current_player in self.ranking:
            self.current_player = (self.current_player + 1) % 4
            return False
        player_hand = player_hand.hand
        # **构造可选牌型**
        possible_moves = []
        for size in [1, 2, 3, 4, 5, 6, 7, 8]:
            for i in range(len(player_hand) - size + 1):
                move = player_hand[i:i + size]
                if self.rules.is_valid_play(move) and (not self.last_play or self.rules.can_beat(self.last_play, move)):
                    possible_moves.append(move)

        if not possible_moves:
            print(f"玩家 {self.current_player + 1} Pass")
            self.pass_count += 1
            self.recent_actions[self.current_player] = []
        else:
            chosen_move = random.choice(possible_moves)  # **随机选择一个合法的牌型**
            self.last_play = chosen_move
            self.last_player = self.current_player
            for card in chosen_move:
                player_hand.remove(card)
            print(f"玩家 {self.current_player + 1} 出牌: {' '.join(chosen_move)}")
            self.recent_actions[self.current_player] = list(chosen_move)
            if not player_hand:
                print(f"\n🎉 玩家 {self.current_player + 1} 出完所有牌！\n")
                self.ranking.append(self.current_player)

            self.pass_count = 0

        self.current_player = (self.current_player + 1) % 4
        return self.check_game_over()

    def user_play(self, player):
        """用户出牌逻辑"""
        while True:
            self.show_user_hand()  # 显示手牌
            choice = input("\n请选择要出的牌（用空格分隔），或直接回车跳过（PASS）： ").strip()

            # **用户选择 PASS**
            if choice == "":
                print(f"玩家 {self.current_player + 1} 选择 PASS")
                self.pass_count += 1
                self.recent_actions[self.current_player] = []  # ✅ 记录 PASS
                break

            if choice.lower() == "pass":
                print(f"玩家 {self.current_player + 1} 选择 PASS")
                self.pass_count += 1
                self.recent_actions[self.current_player] = []  # ✅ 记录 PASS
                break

            # **解析用户输入的牌**
            selected_cards = choice.split()

            # **检查牌是否在手牌中**
            if not all(card in player.hand for card in selected_cards):
                print("❌ 你的输入无效，请确保牌在你的手牌中！")
                continue  # 重新输入

            # **检查牌是否合法**
            if not self.rules.is_valid_play(selected_cards):
                print("❌ 你的出牌不符合规则，请重新选择！")
                continue  # 重新输入

            # **检查是否能压过上一手牌**
            if self.last_play is not None and not self.rules.can_beat(self.last_play, selected_cards):
                print("❌ 你的牌无法压过上一手牌，请重新选择！")
                continue  # 重新输入

            # **成功出牌**
            for card in selected_cards:
                player.hand.remove(card)  # 从手牌中移除
            self.last_play = selected_cards  # 记录这次出牌
            self.last_player = self.current_player  # 记录是谁出的
            self.recent_actions[self.current_player] = list(selected_cards)  # ✅ 记录出牌

            print(f"玩家 {self.current_player + 1} 出牌: {' '.join(selected_cards)}")

            # **如果手牌为空，玩家出完所有牌**
            if not player.hand:
                print(f"\n🎉 玩家 {self.current_player + 1} 出完所有牌！\n")
                self.ranking.append(self.current_player)

            # **出牌成功，Pass 计数归零**
            self.pass_count = 0
            break

        # **切换到下一个玩家**
        self.current_player = (self.current_player + 1) % 4

        return self.check_game_over()

    def check_game_over(self):
        """检查游戏是否结束"""
        team_1 = {0, 2}
        team_2 = {1, 3}

        # **如果有 2 个人出完牌，并且他们是同一队伍，游戏立即结束**
        if len(self.ranking) >= 2:
            first_player, second_player = self.ranking[0], self.ranking[1]
            if (first_player in team_1 and second_player in team_1) or (
                    first_player in team_2 and second_player in team_2):
                self.ranking.extend(i for i in range(4) if i not in self.ranking)  # 剩下的按出牌顺序补全
                self.update_level()
                return True

        # **如果 3 人出完了，自动补全最后一名，游戏结束**
        if len(self.ranking) == 3:
            self.ranking.append(next(i for i in range(4) if i not in self.ranking))  # 找出最后一个玩家
            self.update_level()
            return True

        return False

    def update_level(self):
        """升级级牌"""
        team_1 = {0, 2}
        first_player = self.ranking[0]
        winning_team = 1 if first_player in team_1 else 2

        upgrade_map = {0: 3, 1: 2, 2: 1}  # 头游 + (n 游) 对应的升级规则
        upgrade_amount = upgrade_map[self.ranking.index(self.ranking[1])]  # 根据第二名确定升级级数

        # 计算新级牌（最多升到 A）
        new_level = min(14, self.team_levels[winning_team] + upgrade_amount)
        self.team_levels[winning_team] = new_level

        print(f"\n🏆 {winning_team} 号队伍获胜，级牌升至 {RANKS[new_level - 2]}！")

        if self.team_levels[winning_team] == 14 and upgrade_amount >= 2:
            print("\n🎯 该队已到 A 级，并取得非一四名胜利，游戏结束！")
            return

        new_game = GuandanGame(team_levels=self.team_levels, active_level=new_level)
        new_game.play_game()
        '''
        # **提示玩家是否继续游戏**
        cont = input("是否继续下一局？(y/n): ").strip().lower()
        if cont == 'y' or cont == "":
            # **传入新局的级牌，只使用获胜队伍的级牌**
            new_game = GuandanGame(team_levels=self.team_levels, active_level=new_level)
            new_game.play_game()
        else:
            print("游戏结束！")
        '''

    def play_game(self):
        """执行一整局游戏"""
        level1 = RANKS[self.team_levels[1] - 2]  # 获取 1 号队伍的级牌字符
        level2 = RANKS[self.team_levels[2] - 2]  # 获取 2 号队伍的级牌字符
        print(f"\n🎮 游戏开始！当前级牌 - 1 号队：{level1}，2 号队：{level2}")

        while True:
            if self.play_turn():
                break
    def show_user_hand(self):
        """显示用户手牌（按排序后的顺序）"""
        sorted_hand = self.sort_cards(self.players[self.user_player])
        print("\n🃏 你的手牌：", " ".join(sorted_hand))
        if self.last_play:
            print(f"🃏 场上最新出牌：{' '.join(self.last_play)}\n")

if __name__ == "__main__":
    user_pos = int(input("请选择你的座位（1~4）："))
    game = GuandanGame(user_player=user_pos)
    game.play_game()

