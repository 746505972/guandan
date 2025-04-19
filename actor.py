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

# 初始化模型
actor = ActorNet()
optimizer = optim.Adam(actor.parameters(), lr=1e-4)

# 尝试加载已有模型
def load_latest_model(actor, model_dir="models"):
    model_files = sorted(Path(model_dir).glob("actor_ep*.pth"))
    if model_files:
        latest_model = str(model_files[-1])  # 取最新的模型
        actor.load_state_dict(torch.load(latest_model))
        print(f"✅ 加载已有模型: {latest_model}")
        return int(latest_model.split("_ep")[1].split(".pth")[0])  # 返回最后的ep数
    return 0
initial_ep = load_latest_model(actor)

# 训练函数
def train_on_batch(batch, device="cpu"):
    # 高效转换（无警告）
    states = torch.tensor(np.array([s["state"] for s in batch]), dtype=torch.float32).to(device)
    actions = torch.tensor(np.array([s["action_id"] for s in batch]), dtype=torch.long).to(device)
    rewards = torch.tensor(np.array([s["reward"] for s in batch]), dtype=torch.float32).to(device)

    probs = actor(states)
    log_probs = torch.log(probs + 1e-8)
    chosen_log_probs = log_probs[range(len(batch)), actions]
    loss = -(chosen_log_probs * rewards).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# 模拟训练流程
def run_training(episodes=1000):
    os.makedirs("models", exist_ok=True)
    for ep in range(initial_ep, initial_ep + episodes): # 从上次的ep继续
        game = GuandanGame(verbose=False)
        memory = []
        game.log(f"\n🎮 游戏开始！当前级牌：{RANKS[game.active_level - 2]}")

        while True:
            if game.is_game_over:  # 如果游戏结束，立即跳出循环
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
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                mask = torch.tensor(game.get_valid_action_mask(player.hand, M, game.active_level,game.last_play)).unsqueeze(0)
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

                reward = -len(player.hand)  # 越少越好
                memory.append({"state": state, "action_id": action_id, "reward": reward})
                game.current_player = (game.current_player + 1) % 4
            else:
                game.ai_play(player)  # 其他人用随机
            # **记录最近 5 轮历史**
            if game.current_player == 0:
                round_history = [game.recent_actions[i] for i in range(4)]
                game.history.append(round_history)
                game.recent_actions = [['None'], ['None'], ['None'], ['None']]

        if memory:  # 确保 memory 不为空
            loss = train_on_batch(memory)
            if (ep + 1) % 10 == 0:  # 每 10 局输出一次 loss
                print(f"Episode {ep + 1}, loss: {loss:.4f}")

        if (ep + 1) % 100 == 0:
            torch.save(actor.state_dict(), f"models/actor_ep{ep + 1}.pth")

if __name__ == "__main__":
    run_training()
