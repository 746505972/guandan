import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from guandan_env import GuandanGame
from get_actions import enumerate_colorful_actions
import random
# 加载动作全集 M
with open("doudizhu_actions.json", "r", encoding="utf-8") as f:
    M = json.load(f)
action_dim = len(M)
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
# 构建动作映射字典
M_id_dict = {a['id']: a for a in M}
def find_entry_by_id(data, target_id):
    """返回匹配 id 的整个 JSON 对象"""
    for entry in data:
        if entry.get("id") == target_id:
            return entry
    return None

# Actor 网络定义
class ActorNet(nn.Module):
    def __init__(self, state_dim=3049, action_dim=action_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x, mask=None):
        logits = self.net(x)
        if mask is not None:
            logits = logits + (mask - 1) * 1e9
        return F.softmax(logits, dim=-1)

class CriticNet(nn.Module):
    def __init__(self, state_dim=3049, action_dim=1, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        value = self.net(x)
        return value


# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor = ActorNet().to(device)
critic = CriticNet().to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)
gamma=0.9

# 尝试加载已有模型
def load_latest_models(actor, critic, model_dir="models"):
    model_files = sorted(Path(model_dir).glob("actor_ep*.pth"))
    if model_files:
        latest_actor_path = str(model_files[-1])
        ep = int(latest_actor_path.split("_ep")[1].split(".pth")[0])
        latest_critic_path = f"{model_dir}/critic_ep{ep}.pth"

        actor.load_state_dict(torch.load(latest_actor_path))
        print(f"✅ 加载已有 actor 模型: {latest_actor_path}")

        if Path(latest_critic_path).exists():
            critic.load_state_dict(torch.load(latest_critic_path))
            print(f"✅ 加载已有 critic 模型: {latest_critic_path}")
        else:
            print(f"⚠️ 未找到 critic 模型: {latest_critic_path}")

        return ep
    return 0

# 调用加载函数
initial_ep = load_latest_models(actor, critic)

# 训练函数
def train_on_batch(batch, gamma=0.99, device= None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # 将数据组织为 tensor
    states = torch.tensor(np.array([s["state"] for s in batch]), dtype=torch.float32).to(device)
    actions = torch.tensor(np.array([s["action_id"] for s in batch]), dtype=torch.long).to(device)
    rewards = torch.tensor(np.array([s["reward"] for s in batch]), dtype=torch.float32).to(device)

    # 自动构造 next_states 和 dones
    next_states = torch.zeros_like(states)
    next_states[:-1] = states[1:]  # t+1 的 state
    next_states[-1] = 0.0          # 最后一条无下一状态
    dones = torch.zeros(len(batch), dtype=torch.bool, device=device)
    dones[-1] = True               # 最后一条为 done

    # === Critic 估值 ===
    values = critic(states).squeeze(-1)            # [batch]
    next_values = critic(next_states).squeeze(-1)  # [batch]
    next_values[dones] = 0.0

    # === Advantage ===
    advantages = rewards + gamma * next_values - values

    # === Critic Loss ===
    critic_loss = advantages.pow(2).mean()

    # === Actor Loss ===
    probs = actor(states)                          # [batch, action_dim]
    log_probs = torch.log(probs + 1e-8)
    chosen_log_probs = log_probs[range(len(batch)), actions]
    actor_loss = -(chosen_log_probs * advantages.detach()).mean()

    # === 更新参数 ===
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    return actor_loss.item(), critic_loss.item()

# 模拟训练流程
def run_training(episodes=1000):
    os.makedirs("models", exist_ok=True)
    for ep in range(initial_ep, initial_ep + episodes): # 从上次的ep继续
        game = GuandanGame(verbose=False)
        memory = []
        game.log(f"\n🎮 游戏开始！当前级牌：{RANKS[game.active_level - 2]}")

        while True:
            if game.is_game_over or len(game.history) > 200:  # 如果游戏结束，立即跳出循环
                break
            player = game.players[game.current_player]
            active_players = 4 - len(game.ranking)

            # **如果 Pass 的人 == "当前有手牌的玩家数 - 1"，就重置轮次**
            if game.pass_count >= (active_players - 1) and game.current_player not in game.ranking:
                if game.jiefeng:
                    first_player = game.ranking[-1]
                    teammate = 2 if first_player == 0 else 0 if first_player == 2 else 3 if first_player == 1 else 1
                    game.log(f"\n🆕 轮次重置！玩家 {teammate + 1} 接风。\n")
                    game.recent_actions[game.current_player] = []  # 记录空列表
                    game.current_player = (game.current_player + 1) % 4
                    game.last_play = None  # ✅ 允许新的自由出牌
                    game.pass_count = 0  # ✅ Pass 计数归零
                    game.is_free_turn = True
                    game.jiefeng = False
                else:
                    game.log(f"\n🆕 轮次重置！玩家 {game.current_player + 1} 可以自由出牌。\n")
                    game.last_play = None  # ✅ 允许新的自由出牌
                    game.pass_count = 0  # ✅ Pass 计数归零
                    game.is_free_turn = True

            if game.current_player == 0:
                # 1. 模型推理
                state = game._get_obs()
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                mask = torch.tensor(game.get_valid_action_mask(player.hand, M, game.active_level,game.last_play)).unsqueeze(0).to(device)
                mask1 = mask.squeeze(0)
                probs = actor(state_tensor, mask)
                action_id = torch.multinomial(probs, 1).item()
                action_struct = M_id_dict[action_id]
                # 2. 枚举所有合法出牌组合（带花色）
                combos = enumerate_colorful_actions(action_struct, player.hand, game.active_level)
                if combos:
                    chosen_move = random.choice(combos)
                    if not chosen_move:
                        game.log(f"玩家 {game.current_player + 1} Pass")
                        game.pass_count += 1
                        game.recent_actions[game.current_player] = ['Pass']  # 记录 Pass
                    else:
                        # 如果 chosen_move 不为空，继续进行正常的出牌逻辑
                        game.last_play = chosen_move
                        game.last_player = game.current_player
                        for card in chosen_move:
                            player.played_cards.append(card)
                            player.hand.remove(card)
                        game.log(f"玩家 {game.current_player + 1} 出牌: {' '.join(chosen_move)}")
                        game.recent_actions[game.current_player] = list(chosen_move)  # 记录出牌
                        game.jiefeng = False
                        if not player.hand:  # 玩家出完牌
                            game.log(f"\n🎉 玩家 {game.current_player + 1} 出完所有牌！\n")
                            game.ranking.append(game.current_player)
                            if len(game.ranking) <= 2:
                                game.jiefeng = True

                        game.pass_count = 0
                        if not player.hand:
                            game.pass_count -= 1

                        if game.is_free_turn:
                            game.is_free_turn = False
                else:
                    game.log(f"玩家 {game.current_player + 1} Pass")
                    game.pass_count += 1
                    game.recent_actions[game.current_player] = ['Pass']  # 记录 Pass

                # ============ 奖励函数 ============
                # 从id找到JSON出牌结构
                entry = find_entry_by_id(M,action_id)
                if action_id ==0 :# pass无奖励
                    reward = -0.05
                else:
                    if 120 <= action_id < 330:
                        # DONE: 三带中对子的判断
                        side_point = entry['points'][-1]  # 直接取副牌点数
                        alpha = 0.8  # 控制副牌惩罚强度（越大惩罚越重）
                        reward = len(entry['points']) / ((entry['logic_point'] ** 0.5) + alpha * (side_point ** 0.5))
                    else:
                        reward = float(len(entry['points']) / (entry['logic_point'] ** 0.5))
                    # 对高级出牌奖励修正
                    if 120 <= action_id < 330 : reward += reward * 2
                    if action_id == 119 : reward += 1

                # ========== 拆炸弹、连对惩罚 ==========
                mask2 = torch.tensor(game.get_valid_action_mask(player.hand, M, game.active_level, game.last_play)).squeeze(0).to(device)
                if 0 < action_id <= 48:# 出单牌、对子、三张
                    if (any(mask1[i] and not mask2[i] for i in range(49, 120)) or
                            any(mask1[i] and not mask2[i] for i in range(330, 375))):# 拆了炸、顺子、连对
                        reward -= 0.8
                elif 120 <= action_id < 330:# 出三带二，对子拆了炸
                    side_point = entry['points'][-1]
                    if 2 <= side_point <= 15:
                        # TODO: 判断拆顺子、连对
                        if any(mask1[side_point + 47 + 14 * i] and not mask2[side_point + 47 + 14 * i] for i in range(5)):
                            reward -= 0.8
                    else:
                        if mask1[119] and not mask2[119]:
                            reward -= 0.8
                elif 330 <= action_id < 375:
                    if any(mask1[i] and not mask2[i] for i in range(49, 120)):# 拆了炸
                        reward -= 0.8

                memory.append({"state": state, "action_id": action_id, "reward": reward})
                player.last_played_cards = game.recent_actions[game.current_player]
                game.current_player = (game.current_player + 1) % 4
            else:
                game.ai_play(player)  # 其他人用随机
            # **记录最近 5 轮历史**
            if game.current_player == 0:
                round_history = [game.recent_actions[i] for i in range(4)]
                game.history.append(round_history)
                game.recent_actions = [['None'], ['None'], ['None'], ['None']]

        final_reward = game.upgrade_amount*(1 if game.winning_team == 1 else -1)
        if memory:  # 确保 memory 不为空
            for i, s in enumerate(memory):
                s["reward"] += gamma ** (len(memory) - i - 1) * final_reward
            al,cl = train_on_batch(memory)
            if (ep + 1) % 20 == 0:
                print(f"Episode {ep + 1}, action_loss: {al:.4f},critic_loss: {cl:.4f}")


        if (ep + 1) % 200 == 0:
            torch.save(actor.state_dict(), f"models/actor_ep{ep + 1}.pth")
            torch.save(critic.state_dict(), f"models/critic_ep{ep + 1}.pth")

if __name__ == "__main__":
    run_training()
