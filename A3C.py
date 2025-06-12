import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import threading
import multiprocessing as mp
from collections import deque
import random
import time
import signal # 用于进程清理

# 假设 guandan_env 和 get_actions 在同一个目录下或已添加到 PYTHONPATH
from guandan_env import GuandanGame
from get_actions import enumerate_colorful_actions # 确保此导入正确

# 加载完整的动作集合 M
# 确保 doudizhu_actions.json 文件存在
try:
    with open("doudizhu_actions.json", "r", encoding="utf-8") as f:
        M = json.load(f)
    action_dim = len(M)
    # 构建动作映射字典
    M_id_dict = {a['id']: a for a in M}
except FileNotFoundError:
    print("错误: doudizhu_actions.json 未找到。请确保文件存在。")
    M = []
    action_dim = 0 # 占位符
    M_id_dict = {}

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

def find_entry_by_id(data, target_id):
    """
    返回与给定 ID 匹配的整个 JSON 对象。
    """
    for entry in data:
        if entry.get("id") == target_id:
            return entry
    return None

# 网络定义
class ActorNet(nn.Module):
    def __init__(self, state_dim=3049, action_dim_param=None, hidden_dim=512, num_heads=8, num_layers=1):
        super().__init__()
        if action_dim_param is None:
            try:
                action_dim_param = action_dim 
            except NameError:
                raise ValueError("必须提供 action_dim_param 或定义全局 'action_dim'。")
        self.state_dim = state_dim
        self.action_dim_param = action_dim_param
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(state_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim_param) )
    
    def forward(self, x, mask=None, temperature=1.0): # 添加温度参数
        x = self.input_projection(x)
        x = x.unsqueeze(1)
        transformer_output = self.transformer_encoder(x)
        transformer_output = transformer_output.squeeze(1)
        logits = self.output_net(transformer_output)
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand_as(logits)
            # 对于布尔掩码，无效动作应设置为一个很大的负值
            logits = logits.masked_fill(~mask, -1e9) 
        return F.softmax(logits / temperature, dim=-1) # 应用温度参数

class CriticNet(nn.Module):
    def __init__(self, state_dim=3049, hidden_dim=512, num_heads=8, num_layers=1):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(state_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)  )
    
    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(1)
        transformer_output = self.transformer_encoder(x)
        transformer_output = transformer_output.squeeze(1)
        value = self.output_net(transformer_output)
        return value

def plot_loss(loss_values, filename="critic_loss.png"):
    """
    绘制并保存评论家网络的损失曲线。
    """
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, label='评论家损失')
        plt.xlabel('训练步数'); plt.ylabel('损失')
        plt.title('评论家损失曲线'); plt.legend(); plt.grid(True)
        plt.savefig(filename); plt.close()
    except ImportError:
        print(f"未找到 Matplotlib。无法绘制损失图。跳过 {filename} 的绘图。")
    except Exception as e:
        print(f"绘制损失图时出错: {e}")

class GlobalActorNet(ActorNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = mp.Lock() 

class GlobalCriticNet(CriticNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = mp.Lock()
        
if action_dim > 0:
    global_actor = GlobalActorNet(action_dim_param=action_dim).to("cpu")
    global_critic = GlobalCriticNet().to("cpu")
    global_actor.share_memory(); global_critic.share_memory()
    global_actor_optimizer = optim.Adam(global_actor.parameters(), lr=1e-5) 
    global_critic_optimizer = optim.Adam(global_critic.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.StepLR(global_actor_optimizer, step_size=100, gamma=0.95)
else:
    global_actor = None; global_critic = None
    global_actor_optimizer = None; global_critic_optimizer = None; scheduler = None
    print("严重错误: 由于 action_dim 为 0，全局网络未初始化。")

gamma = 0.9

def load_latest_models(model_dir="models"):
    """
    从指定目录加载最新模型。
    """
    if global_actor is None or global_critic is None:
        print("跳过模型加载，因为全局网络未初始化。")
        return 0
    model_files = sorted(Path(model_dir).glob("actor_ep*.pth"))
    if model_files:
        latest_actor_path = str(model_files[-1])
        try:
            ep_str = latest_actor_path.split("_ep")[1].split(".pth")[0]
            ep = int(ep_str)
            latest_critic_path = Path(model_dir) / f"critic_ep{ep}.pth"
            global_actor.load_state_dict(torch.load(latest_actor_path, map_location=torch.device('cpu')))
            if latest_critic_path.exists():
                global_critic.load_state_dict(torch.load(str(latest_critic_path), map_location=torch.device('cpu')))
            print(f"已从第 {ep} 集加载模型")
            return ep
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return 0
    return 0

initial_ep = load_latest_models()

class ReplayBuffer:
    """
    简单的经验回放缓冲区。
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """
        向缓冲区添加一个经验元组。
        """
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """
        从缓冲区随机采样一批数据。
        """
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        """
        返回缓冲区中存储的经验数量。
        """
        return len(self.buffer)

class Worker(mp.Process):
    """
    A3C 算法中的工作进程，负责与环境交互、收集经验并更新全局网络。
    """
    def __init__(self, worker_id, g_actor, g_critic, g_actor_opt, g_critic_opt, log_q=None, max_eps=1000, log_action_probability=0.05):
        super().__init__()
        self.worker_id = worker_id
        self.log_queue = log_q
        self.global_actor = g_actor; self.global_critic = g_critic
        self.global_actor_optimizer = g_actor_opt; self.global_critic_optimizer = g_critic_opt
        self.gamma = gamma; self.device = "cpu"
        self.critic_losses = []; self.max_episodes = max_eps
        self.current_episode = 0  # 添加回合计数器
        self.log_action_probability = log_action_probability
        self.M = M; self.M_id_dict = M_id_dict; self.action_dim = action_dim
        self.best_reward = -float('inf') # 用于保存最佳模型

        if self.action_dim == 0:
            self.log(f"错误: 工作器 {self.worker_id} 的 action_dim 为 0。中止初始化。")
            return
        self.local_actors = [ActorNet(action_dim_param=self.action_dim).to(self.device) for _ in range(4)]
        self.local_critics = [CriticNet().to(self.device) for _ in range(4)]
        if self.global_actor and self.global_critic: self.sync_with_global()
        else: self.log("全局网络不可用，跳过初始同步。")

        self.replay_buffer = ReplayBuffer(capacity=5000) # 初始化经验回放缓冲区
        self.sync_counter = 0
        self.sync_frequency = 5 # 每 5 个回合同步一次

    def log(self, message):
        """
        将日志消息发送到日志队列。
        """
        log_message = f"工作器 {self.worker_id} (PID {os.getpid()}): {message}"
        if self.log_queue:
            try: self.log_queue.put(log_message)
            except Exception as e: print(f"写入日志队列时出错: {e}。回退到打印: {log_message}")
        else: print(log_message)

    def sync_with_global(self):
        """
        将本地 Actor 和 Critic 网络的参数与全局网络同步。
        """
        if not self.global_actor or not self.global_critic:
            self.log("无法同步，全局 Actor 或 Critic 为 None。")
            return
        try:
            with self.global_actor._lock, self.global_critic._lock:
                global_actor_state = self.global_actor.state_dict()
                # 使用 state_dict 的深拷贝以避免多进程问题
                global_actor_state = {k: v.clone() for k, v in global_actor_state.items()}
                for actor in self.local_actors: actor.load_state_dict(global_actor_state)
                
                global_critic_state = self.global_critic.state_dict()
                # 使用 state_dict 的深拷贝
                global_critic_state = {k: v.clone() for k, v in global_critic_state.items()}
                for critic in self.local_critics: critic.load_state_dict(global_critic_state)
        except Exception as e: self.log(f"sync_with_global 时出错: {e}")

    def _compute_returns_and_advantages(self, memory, next_state_val=0.0, last_done=False):
        """
        使用广义优势估计 (GAE) 计算 n 步回报和优势函数。
        将访问方式从字典改为元组索引
        """
        # 修改数据访问方式，使用元组索引而不是字典键
        states = torch.tensor([s[0] for s in memory], dtype=torch.float32).to(self.device)
        rewards = torch.tensor([s[2] for s in memory], dtype=torch.float32).to(self.device)
        dones = torch.tensor([s[4] for s in memory], dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            values = self.local_critics[0](states).squeeze(-1)
        
        # 其余代码保持不变
        gae = 0
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        lambda_param = 0.95
        
        next_value_bootstrap = next_state_val if not last_done else 0.0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_val = next_value_bootstrap
            else:
                next_val = values[i+1] * (1 - dones[i+1].float())
                
            delta = rewards[i] + self.gamma * next_val - values[i]
            gae = delta + self.gamma * lambda_param * gae * (1 - dones[i].float())
            advantages[i] = gae
            returns[i] = advantages[i] + values[i]
        
        return returns, advantages

    def train_on_batch(self, batch, episode):
        """
        在给定经验批次上训练本地 Actor 和 Critic 网络并更新全局网络。
        """
        if not batch: self.log("批次为空，跳过训练。"); return 0.0, 0.0
        self.current_episode = episode  # 更新当前回合
        if not batch: self.log("批次为空，跳过训练。"); return 0.0, 0.0
        if not self.global_actor or not self.global_critic: self.log("全局模型未初始化。跳过训练。"); return 0.0, 0.0

        # 已更正: 通过索引而不是字符串键访问元素
        states_np = np.array([s[0] for s in batch])
        actions_np = np.array([s[1] for s in batch])
        rewards_np = np.array([s[2] for s in batch])
        next_states_np = np.array([s[3] for s in batch]) # 用于 n 步回报/GAE
        dones_list = [s[4] for s in batch]
        

        states = torch.tensor(states_np, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions_np, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards_np, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones_list, dtype=torch.bool, device=self.device)
        next_states = torch.tensor(next_states_np, dtype=torch.float32).to(self.device)

        current_critic = self.local_critics[0]; current_actor = self.local_actors[0]   

        # 计算 GAE 引导的 next_state_val
        next_state_val = 0.0
        if not dones[-1]: # 如果批次中的最后一个状态不是终止状态
            with torch.no_grad():
                next_state_val = current_critic(next_states[-1].unsqueeze(0)).item()

        returns, advantages = self._compute_returns_and_advantages(batch, next_state_val, bool(dones[-1].item()))
        
        values = current_critic(states).squeeze(-1) # 重新计算用于损失计算的值
        critic_loss = advantages.pow(2).mean() # 使用 GAE 优势计算评论家损失

        probs = current_actor(states)
        log_probs = torch.log(probs.clamp(min=1e-8))
        chosen_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        actor_loss = -(chosen_log_probs * advantages.detach()).mean() 
        
        # 动态调整熵系数
        # 这里 ep 是当前回合数，self.max_episodes 是总回合数
        # 确保 ep 在 self.max_episodes 范围内，以避免除以零或负值
        current_ep_ratio = min(self.current_episode / self.max_episodes, 1.0) if self.max_episodes > 0 else 0.0
        entropy_coeff = max(0.01 * (1 - current_ep_ratio * 0.5), 0.001) # 从 0.01 降到 0.005，最小 0.001

        entropy = -(probs * log_probs).sum(dim=-1).mean()
        actor_loss = actor_loss - entropy_coeff * entropy

        # L2 正则化
        l2_reg = 1e-4
        l2_norm_actor = sum(p.pow(2.0).sum() for p in current_actor.parameters())
        l2_norm_critic = sum(p.pow(2.0).sum() for p in current_critic.parameters())
        actor_loss += l2_reg * l2_norm_actor
        critic_loss += l2_reg * l2_norm_critic

        with self.global_actor._lock, self.global_critic._lock:
            self.global_actor_optimizer.zero_grad()
            self.global_critic_optimizer.zero_grad()
            
            actor_loss.backward(retain_graph=True)
            critic_loss.backward()
            
            # 直接设置全局梯度，此步骤不会在工作器之间累积梯度
            for lp, gp in zip(current_actor.parameters(), self.global_actor.parameters()):
                if lp.grad is not None: 
                    gp.grad = lp.grad.clone()
            for lp, gp in zip(current_critic.parameters(), self.global_critic.parameters()):
                if lp.grad is not None: 
                    gp.grad = lp.grad.clone()
            
            self.global_actor_optimizer.step()
            self.global_critic_optimizer.step()

            # 学习率调度器步骤（仅工作器 0）
            if self.worker_id == 0 and scheduler is not None:
                scheduler.step()

        # 更新后将本地模型与全局模型同步
        self.sync_with_global()
        cl_val = critic_loss.item(); al_val = actor_loss.item()
        self.critic_losses.append(cl_val)
        
        if len(self.critic_losses) % 100 == 0 and self.worker_id == 0 and len(self.critic_losses) % 1000 == 0:
            plot_loss(self.critic_losses, filename=f"critic_loss_w{self.worker_id}_e{len(self.critic_losses)}.png")
        
        if self.worker_id == 0:
            with torch.no_grad():
                mean_adv = advantages.mean().item(); mean_val = values.mean().item(); mean_rew = rewards.mean().item() 
            self.log(f"批次训练摘要 (W0): MA:{mean_adv:.2f}, MV:{mean_val:.2f}, MR:{mean_rew:.2f}, AL:{al_val:.3f}, CL:{cl_val:.3f}")
        return al_val, cl_val

    def evaluate_agent(self, num_eval_episodes):
        """
        在不探索的情况下评估代理的性能。
        """
        if self.worker_id != 0: # 只有主工作器执行评估
            return

        self.log(f"--- 开始评估 (运行 {num_eval_episodes} 回合) ---")
        self.local_actors[0].eval() # 将 Actor 网络设置为评估模式
        eval_rewards = []
        for _ in range(num_eval_episodes):
            game = GuandanGame(verbose=False)
            episode_reward = 0
            game_steps = 0
            max_game_steps = 200

            while not game.is_game_over and game_steps < max_game_steps:
                current_player_id = game.current_player
                player = game.players[current_player_id]

                if current_player_id in [0, 2]: # 代理动作
                    state_t = game._get_obs()
                    valid_action_mask_np = game.get_valid_action_mask(player.hand, self.M, game.active_level, game.last_play)
                    mask_tensor = torch.tensor(valid_action_mask_np, dtype=torch.bool).to(self.device)

                    with torch.no_grad():
                        # 在评估模式下，可以将温度参数设置为一个非常小的值，或者直接使用 argmax
                        # 这里我们使用 argmax 进行确定性选择
                        probs = self.local_actors[current_player_id](torch.tensor(state_t, dtype=torch.float32).unsqueeze(0).to(self.device), mask_tensor, temperature=0.1) # 低温度
                        
                        # 确保有有效动作可供选择，否则选择过牌
                        if mask_tensor.sum().item() == 0:
                            action_id = 0 # 强制过牌
                        else:
                            action_id = int(torch.argmax(probs).item()) # 确保 action_id 是整数
                            # 正确索引: mask_tensor 在这里是一个 1D 张量
                            if not mask_tensor[action_id]: # 再次检查是否有效
                                valid_indices = np.where(valid_action_mask_np == 1)[0]
                                action_id = int(np.random.choice(valid_indices)) if len(valid_indices) > 0 else 0 # 确保 action_id 是整数


                    action_struct = self.M_id_dict.get(action_id)
                    chosen_move = []
                    if action_struct:
                        combos = enumerate_colorful_actions(action_struct, player.hand, game.active_level)
                        if combos: chosen_move = random.choice(combos) # 在评估期间也可以随机选择一个有效组合

                    if not chosen_move or action_id == 0:
                        game.pass_count += 1
                        game.recent_actions[current_player_id] = ['过牌']
                        actual_played_cards = ['过牌']
                    else:
                        game.last_play = chosen_move; game.last_player = current_player_id
                        temp_hand_for_check = list(player.hand)
                        valid_move_in_hand = True
                        for card_in_move_check in chosen_move:
                            if card_in_move_check in temp_hand_for_check: temp_hand_for_check.remove(card_in_move_check)
                            else: valid_move_in_hand = False; break
                        if valid_move_in_hand:
                            player.hand = temp_hand_for_check
                            for card_in_move in chosen_move: player.played_cards.append(card_in_move)
                            game.recent_actions[current_player_id] = list(chosen_move)
                            game.jiefeng = False; game.is_free_turn = False
                            if not player.hand:
                                game.ranking.append(current_player_id)
                                if len(game.ranking) <= 2: game.jiefeng = True
                            game.pass_count = 0
                            actual_played_cards = chosen_move
                        else:
                            game.pass_count += 1
                            game.recent_actions[current_player_id] = ['过牌']; actual_played_cards = ['过牌']
                    
                    # 在评估期间不计算中间奖励，只关注最终的输赢
                    # episode_reward += current_step_reward # 如果需要，累积中间奖励

                    game.current_player = (current_player_id + 1) % 4
                else: # 对手动作
                    if current_player_id not in game.ranking:
                        game.ai_play(player)
                    else:
                        game.recent_actions[current_player_id] = ['已完成']
                        game.current_player = (current_player_id + 1) % 4
                
                game_steps += 1
                game.check_game_over() # 检查游戏是否结束

            # 计算最终奖励
            final_eval_reward = 0
            if game.is_game_over and game.winning_team is not None:
                if game.winning_team == 1: # 代理团队获胜
                    final_eval_reward = game.upgrade_amount
                else: # 代理团队失败
                    final_eval_reward = -game.upgrade_amount
            eval_rewards.append(final_eval_reward)
        
        avg_eval_reward = np.mean(eval_rewards)
        self.log(f"--- 评估结束: 平均奖励 = {avg_eval_reward:.2f} ---")
        self.local_actors[0].train() # 将 Actor 网络设置回训练模式

    def run(self):
        """
        工作进程的主运行循环。
        """
        if not self.global_actor or not self.global_critic or self.action_dim == 0:
            self.log(f"工作器 {self.worker_id} 无法启动，因为模型或 action_dim 问题。")
            return
        self.log(f"开始训练，最大回合数={self.max_episodes}，日志概率={self.log_action_probability:.2f}")
        
        for ep in range(initial_ep, self.max_episodes):
            # 每 N 个回合将本地模型与全局模型同步
            if self.sync_counter == 0: # __init__ 中已完成初始同步
                self.sync_with_global()
            self.sync_counter = (self.sync_counter + 1) % self.sync_frequency

            game = GuandanGame(verbose=False) 
            episode_memory = []
            game_steps = 0
            max_game_steps = 200 
            log_this_episode = random.random() < self.log_action_probability
            
            if log_this_episode and self.worker_id == 0:
                 self.log(f"--- 回合 {ep+1} (工作器 {self.worker_id}): 详细日志已启用 ---")

            while not game.is_game_over and game_steps < max_game_steps :
                # current_player_id 在此步骤开始时确定，以查看谁应该行动
                current_player_id = game.current_player 
                player = game.players[current_player_id]
                
                active_players_count = 4 - len(game.ranking)
                is_player_still_in_game = current_player_id not in game.ranking

                # 回合重置逻辑 (接风或正常过牌回合)
                if game.pass_count >= (active_players_count - 1) and is_player_still_in_game and active_players_count > 0:
                    original_player_for_reset_check = current_player_id
                    if log_this_episode and self.worker_id == 0:
                        self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps}: 检测到 P{original_player_for_reset_check+1} 的回合重置条件。接风:{game.jiefeng}, 过牌计数:{game.pass_count}, 活跃玩家:{active_players_count}, 排名:{game.ranking}")
                    
                    if game.jiefeng: 
                        if game.ranking: 
                            last_to_finish = game.ranking[-1]
                            teammate = (last_to_finish + 2) % 4 
                            if teammate not in game.ranking: 
                                game.current_player = teammate 
                                if log_this_episode and self.worker_id == 0: self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps}: 接风！P{last_to_finish+1} 的队友 P{teammate+1} 可以行动。")
                            else: 
                                game.current_player = original_player_for_reset_check 
                                if log_this_episode and self.worker_id == 0: self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps}: 接风失败 (队友 P{teammate+1} 已完成)。P{original_player_for_reset_check+1} 正常自由出牌。")
                        else: 
                             game.current_player = original_player_for_reset_check
                             if log_this_episode and self.worker_id == 0: self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps}: 接风期间排名为空 (异常！)。P{original_player_for_reset_check+1} 正常自由出牌。")
                        game.jiefeng = False 
                    else: 
                        game.current_player = original_player_for_reset_check 
                        if log_this_episode and self.worker_id == 0: self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps}: P{original_player_for_reset_check+1} 的正常过牌回合重置。过牌计数:{game.pass_count}")
                    
                    game.last_play = None; game.pass_count = 0; game.is_free_turn = True
                    # 更新 current_player_id 和 player 以反映重置后的实际行动者
                    current_player_id = game.current_player
                    player = game.players[current_player_id]
                    if log_this_episode and self.worker_id == 0: self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps}: 回合重置后，实际行动玩家 P{current_player_id+1}。自由出牌:{game.is_free_turn}, 上一手牌:{game.last_play}")

                acted_player_id = current_player_id # 记录在此回合中实际行动的玩家

                # 代理 (0 或 2) 动作
                if current_player_id in [0, 2]: 
                    state_t = game._get_obs() 
                    valid_action_mask_np = game.get_valid_action_mask( player.hand, self.M, game.active_level, game.last_play )
                    mask_tensor = torch.tensor(valid_action_mask_np, dtype=torch.bool).to(self.device)
                    actor_net_idx = current_player_id

                    # 动态调整温度参数
                    temperature = max(1.0 - ep / self.max_episodes * 0.8, 0.1) # 从 1.0 降到 0.1

                    with torch.no_grad():
                        probs = self.local_actors[actor_net_idx](torch.tensor(state_t, dtype=torch.float32).unsqueeze(0).to(self.device), mask_tensor, temperature=temperature)
                    
                    action_id = 0 
                    if torch.isnan(probs).any() or torch.isinf(probs).any():
                        self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps} P{current_player_id+1}: 检测到 NaN/Inf 概率。强制随机有效动作选择。")
                        valid_indices = np.where(valid_action_mask_np == 1)[0]
                        action_id = int(np.random.choice(valid_indices)) if len(valid_indices) > 0 else 0 # 确保 action_id 是整数
                    elif mask_tensor.sum().item() == 0:
                        self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps} P{current_player_id+1}: 没有有效动作 (掩码全部为 F)。强制过牌。")
                        action_id = 0 # 强制过牌
                    else:
                        try: 
                            action_id = int(torch.multinomial(probs.clamp(min=1e-8), 1).item()) # 确保 action_id 是整数
                            # 正确索引: mask_tensor 在这里是一个 1D 张量
                            if not mask_tensor[action_id]: # 再次检查是否有效
                                self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps} P{current_player_id+1}: 采样到无效动作 {action_id}。强制随机有效动作。")
                                valid_indices = np.where(valid_action_mask_np == 1)[0]
                                action_id = int(np.random.choice(valid_indices)) if len(valid_indices) > 0 else 0 # 确保 action_id 是整数
                        except RuntimeError as e:
                            self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps} P{current_player_id+1}: 多项式采样错误:{e}。概率和:{probs.sum().item()}, 概率:{probs.tolist()}。随机有效动作。")
                            valid_indices = np.where(valid_action_mask_np == 1)[0]
                            action_id = int(np.random.choice(valid_indices)) if len(valid_indices) > 0 else 0 # 确保 action_id 是整数
                    
                    action_struct = self.M_id_dict.get(action_id)
                    chosen_move = []
                    if action_struct: 
                        combos = enumerate_colorful_actions(action_struct, player.hand, game.active_level)
                        if combos: chosen_move = random.choice(combos) 
                        elif action_id != 0: 
                            if self.worker_id == 0: # 只有关键工作器记录此详细日志
                                self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps} P{current_player_id+1}: 打算出牌 (ID:{action_id}, 结构:{action_struct})，但未找到组合。手牌:{len(player.hand)}, 级别:{game.active_level}。将过牌。")
                    
                    actual_played_cards = [] # 初始化 actual_played_cards

                    if not chosen_move or action_id == 0 : 
                        # 此块处理代理过牌 (自愿或强制)
                        game.pass_count += 1
                        game.recent_actions[current_player_id] = ['过牌']
                        actual_played_cards = ['过牌']
                        if log_this_episode and self.worker_id == 0: self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps} P{current_player_id+1}: 过牌。上一手牌:{game.last_play}, 手牌:{len(player.hand)}")
                    else: 
                        # 此块处理代理尝试出牌
                        game.last_play = chosen_move; game.last_player = current_player_id
                        temp_hand_for_check = list(player.hand) 
                        valid_move_in_hand = True
                        for card_in_move_check in chosen_move:
                            if card_in_move_check in temp_hand_for_check: temp_hand_for_check.remove(card_in_move_check)
                            else: valid_move_in_hand = False; break
                        if valid_move_in_hand:
                            player.hand = temp_hand_for_check 
                            for card_in_move in chosen_move: player.played_cards.append(card_in_move)
                            game.recent_actions[current_player_id] = list(chosen_move)
                            game.jiefeng = False; game.is_free_turn = False 
                            if not player.hand: 
                                game.ranking.append(current_player_id)
                                if len(game.ranking) <= 2: game.jiefeng = True 
                            game.pass_count = 0
                            actual_played_cards = chosen_move 
                            if log_this_episode and self.worker_id == 0: self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps} P{current_player_id+1}: 出牌 {actual_played_cards}。手牌:{len(player.hand)}")
                        else: 
                            # 这是一个严重错误: 代理尝试出牌但手牌中没有。强制过牌。
                            self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps} P{current_player_id+1}: 严重错误！手牌 {player.hand} 中没有牌 {chosen_move}。强制过牌。")
                            game.pass_count += 1
                            game.recent_actions[current_player_id] = ['过牌']; actual_played_cards = ['过牌']

                    player.last_played_cards = game.recent_actions[current_player_id]
                    is_done_after_agent = game.check_game_over() 
                    
                    # game.current_player 尚未改变，因此 _get_obs() 观察当前行动代理行动后的状态
                    state_t_plus_1 = game._get_obs() 
                    
                    # --- 中间奖励计算 ---
                    reward_t = 0.0 # 在此块开始时初始化 reward_t
                    if action_id == 0: # 代理选择过牌
                        if game.is_free_turn:
                            reward_t = -1 # 自由出牌时过牌的惩罚更重
                            if log_this_episode and self.worker_id == 0:
                                self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps} P{current_player_id+1}: 选择自由过牌。奖励: {reward_t}")
                        else:
                            reward_t = -1 
                            if log_this_episode and self.worker_id == 0:
                                self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps} P{current_player_id+1}: 选择过牌 (非自由出牌)。奖励: {reward_t}")
                    elif actual_played_cards != ['过牌']: # 代理成功出牌
                        entry = find_entry_by_id(self.M, action_id) # action_id 应该是选定的那个
                        if entry and 'points' in entry and 'logic_point' in entry and entry['logic_point'] > 0:
                            base_reward = len(entry['points']) / entry['logic_point']
                            reward_t = torch.sigmoid(torch.tensor([base_reward])).item() # 奖励在 0-1 之间
                            
                            # 根据剩余手牌大小调整奖励 (进度奖励)
                            # 假设初始手牌大小为 54 (两副牌)
                            cards_left_ratio = len(player.hand) / 54.0 
                            progress_reward = 0.1 * (1 - cards_left_ratio) # 牌越少奖励越高
                            reward_t += progress_reward

                            # 对特殊牌型 (炸弹、火箭) 给予额外奖励
                            if 121 <= action_id <= 364: 
                                reward_t += 0.5 
                            else: 
                                reward_t -= 0.1


                            if log_this_episode and self.worker_id == 0:
                                self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps} P{current_player_id+1}: 成功出牌 (ID:{action_id})。奖励: {reward_t:.2f}")
                        else:
                            reward_t = -0.01 # 出牌但结构信息不完整，轻微惩罚
                            if log_this_episode and self.worker_id == 0:
                                self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps} P{current_player_id+1}: 出牌但结构信息不完整 (ID:{action_id})。奖励: {reward_t:.2f}")
                    elif action_id != 0 and actual_played_cards == ['过牌']:
                        # 代理打算出牌 (action_id != 0)，但实际操作是过牌 (actual_played_cards == ['过牌'])
                        # 这意味着代理选择了无效动作，应该受到惩罚。
                        reward_t = -3 # 选择无效动作的显著惩罚
                        self.log(f"惩罚事件: 玩家 {current_player_id+1} 打算出牌 (action_id: {action_id}) 但被迫过牌 (无效动作)。奖励: {reward_t}")
                    else: # 其他情况，例如，无效的 action_id 但不知何故到达这里
                        reward_t = -0.1 # 未知情况，施加惩罚
                        if log_this_episode and self.worker_id == 0:
                            self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps} P{current_player_id+1}: 未知情况/无效 action_id。奖励: {reward_t:.2f}")

                    episode_memory.append((
                        state_t,
                        action_id,
                        reward_t,
                        state_t_plus_1,
                        is_done_after_agent
                    ))
                    
                    if is_done_after_agent and log_this_episode and self.worker_id == 0: self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps} P{current_player_id+1}: 动作后游戏结束。")
                    
                    # 代理动作后，轮到下一个玩家
                    game.current_player = (current_player_id + 1) % 4

                # 对手 (1 或 3) 动作
                else: 
                    if current_player_id not in game.ranking: 
                        if log_this_episode and self.worker_id == 0: self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps}: 对手 P{current_player_id+1} (AI) 动作开始。手牌:{len(player.hand)}, 上一手牌:{game.last_play}, 自由出牌:{game.is_free_turn}")
                        game.ai_play(player) # game.ai_play 内部更新 game.current_player
                        if log_this_episode and self.worker_id == 0: self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps}: 对手 P{current_player_id+1} (AI) 出牌: {game.recent_actions[current_player_id]}。手牌:{len(player.hand)}")
                    else: 
                        if log_this_episode and self.worker_id == 0: self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps}: 对手 P{current_player_id+1} (AI) 已完成，跳过。")
                        game.recent_actions[current_player_id] = ['已完成']
                        # 如果 AI 已完成，仍然需要逻辑上轮换到下一个玩家
                        game.current_player = (current_player_id + 1) % 4
                
                game_steps += 1
                # game.is_game_over 将在 game.check_game_over() (由代理调用) 或 game.ai_play() (由 AI 调用) 中更新
                if game.is_game_over:
                    self.log(f"W{self.worker_id} Ep{ep+1} St{game_steps}: 游戏结束 (在 P{acted_player_id+1} 的动作之后)。排名:{game.ranking}")
                    break
            
            if game_steps >= max_game_steps and not game.is_game_over:
                self.log(f"W{self.worker_id} Ep{ep+1}: 达到最大步数 ({max_game_steps})。结束回合。")

            final_scalar_reward = 0 

            reward_multiplier = 0.5
            if game.is_game_over and game.winning_team is not None: 
                if game.winning_team == 1: # 假设团队 1 包含玩家 0 和 2 (代理)
                    final_scalar_reward = game.upgrade_amount * reward_multiplier
                else: # 代理团队失败
                    final_scalar_reward = -game.upgrade_amount * reward_multiplier
            
            # 添加完成牌的奖励 (如果适用且未被最终奖励覆盖)
            # 这是一种通用的“进度”奖励，不同于游戏输赢
            if current_player_id in [0, 2] and not player.hand and not game.is_game_over:
                # 如果代理在游戏中完成牌，但游戏尚未结束 (例如，队友仍在玩)
                final_scalar_reward += 0.3 # 提前完成牌的奖励
                self.log(f"W{self.worker_id} Ep{ep+1}: 代理 P{current_player_id+1} 提前完成牌。额外奖励 +2.0。")
            elif current_player_id in [0, 2] and len(player.hand) <= 5 and not game.is_game_over:
                # 鼓励减少手牌大小
                final_scalar_reward += 0.2
                self.log(f"W{self.worker_id} Ep{ep+1}: 代理 P{current_player_id+1} 剩余牌很少。额外奖励 +0.1。")

            # 在条件块之前初始化 al 和 cl
            al = 0.0
            cl = 0.0

            if episode_memory: 
                # 创建新的元组来更新最后一个经验
                last_experience = episode_memory[-1]
                updated_experience = (
                    last_experience[0],  # state
                    last_experience[1],  # action_id
                    last_experience[2] + final_scalar_reward,  # reward + final_scalar_reward
                    last_experience[3],  # next_state
                    True  # done
                )
                episode_memory[-1] = updated_experience

                # 将回合经验推送到回放缓冲区
                for transition in episode_memory:
                    self.replay_buffer.push(
                        transition[0],  # state
                        transition[1],  # action_id
                        transition[2],  # reward
                        transition[3],  # next_state
                        transition[4]   # done
                    )

                if len(self.replay_buffer) >= 128: # 最小批次大小
                    minibatch = self.replay_buffer.sample(min(128, len(self.replay_buffer)))
                    al, cl = self.train_on_batch(minibatch, ep) 
                    minibatch = self.replay_buffer.sample(min(128, len(self.replay_buffer)))
                    al, cl = self.train_on_batch(minibatch, ep) 
                else:
                    self.log(f"W{self.worker_id} Ep{ep+1}: 回放缓冲区太小 ({len(self.replay_buffer)})。跳过批次训练。")

                self.log(f"W{self.worker_id} Ep{ep+1} 摘要: AL={al:.4f}, CL={cl:.4f}, 最终奖励={final_scalar_reward}, 获胜队伍:{game.winning_team}, 步数:{game_steps}, 内存大小:{len(episode_memory)}, 缓冲区大小:{len(self.replay_buffer)}")
            else:
                self.log(f"W{self.worker_id} Ep{ep+1} 摘要: 无训练数据。最终奖励={final_scalar_reward}, 获胜队伍:{game.winning_team}, 步数:{game_steps}")
            
            if log_this_episode and self.worker_id == 0:
                self.log(f"--- 回合 {ep+1} (工作器 {self.worker_id}): 详细日志结束。最终排名:{game.ranking}, 获胜队伍:{game.winning_team}, 升级数量:{game.upgrade_amount} ---")

            # 最佳模型保存逻辑
            if self.worker_id == 0:
                current_avg_reward = final_scalar_reward # 使用最终奖励作为回合性能的代理
                if current_avg_reward > self.best_reward:
                    self.best_reward = current_avg_reward
                    self.log(f"保存最佳模型 @ 回合 {ep + 1} (W0), 奖励: {current_avg_reward:.2f}")
                    if self.global_actor and self.global_critic:
                        with self.global_actor._lock, self.global_critic._lock:
                            os.makedirs("models", exist_ok=True)
                            torch.save(self.global_actor.state_dict(), f"models/actor_best.pth")
                            torch.save(self.global_critic.state_dict(), f"models/critic_best.pth")
                
                if (ep + 1) % 50 == 0: # 也定期保存
                    self.log(f"保存模型 @ 回合 {ep + 1} (W0)")
                    if self.global_actor and self.global_critic:
                        with self.global_actor._lock, self.global_critic._lock:
                            os.makedirs("models", exist_ok=True)
                            torch.save(self.global_actor.state_dict(), f"models/actor_ep{ep + 1}.pth")
                            torch.save(self.global_critic.state_dict(), f"models/critic_ep{ep + 1}.pth")
            
            # 每 N 个回合添加评估阶段 (无探索，仅用于评估)
            if (ep + 1) % 100 == 0 and self.worker_id == 0:
                self.evaluate_agent(5) # 运行 5 个评估回合

log_queue = mp.Queue()

def logger_thread_func(q):
    """
    日志线程函数，从队列获取消息并打印。
    """
    while True:
        try:
            message = q.get()
            if message == "STOP": print("日志线程停止。"); break
            print(message)
        except EOFError: print("日志队列 EOF，日志线程退出。"); break
        except Exception as e: print(f"日志线程错误: {e}"); break

def run_global_training(num_workers=2, total_episodes_per_worker=1500, log_action_probability=0.05): 
    """
    启动全局训练过程，包括创建工作进程和日志线程。
    """
    if not global_actor or not global_critic: print("无法开始训练: 全局网络未初始化。"); return
    os.makedirs("models", exist_ok=True)
    logger_proc = threading.Thread(target=logger_thread_func, args=(log_queue,)); logger_proc.daemon = True; logger_proc.start()
    log_queue.put("日志线程已启动。")
    workers = []
    for i in range(num_workers):
        workers.append(Worker(i, global_actor, global_critic, global_actor_optimizer, global_critic_optimizer, log_queue, total_episodes_per_worker, log_action_probability))
    for w in workers: w.start()
    log_queue.put(f"已启动 {num_workers} 个工作进程。")
    try:
        for w in workers: w.join() 
    except KeyboardInterrupt:
        log_queue.put("捕获到 KeyboardInterrupt: 正在停止工作进程...")
        for w in workers:
            if w.is_alive(): w.terminate(); w.join(timeout=5)
            if w.is_alive(): log_queue.put(f"工作器 {w.worker_id} 未能干净地终止。")
    finally:
        log_queue.put("所有工作进程已完成或终止。")
        log_queue.put("正在保存最终模型...")
        if global_actor and global_critic: 
            try:
                with global_actor._lock, global_critic._lock: 
                    torch.save(global_actor.state_dict(), "models/actor_final.pth")
                    torch.save(global_critic.state_dict(), "models/critic_final.pth")
                log_queue.put("最终模型已保存。")
            except Exception as e: log_queue.put(f"保存最终模型时出错: {e}")
        log_queue.put("STOP") 
        if logger_proc.is_alive(): logger_proc.join(timeout=2) 
        try:
            while not log_queue.empty(): log_queue.get_nowait()
        except Exception: pass 
        log_queue.close()

if __name__ == "__main__":
    if not M: print("M (动作集) 为空。退出。请检查 'doudizhu_actions.json'。")
    elif global_actor is None or global_critic is None: print("全局网络初始化失败。退出。")
    else:
        if os.name == 'nt': mp.set_start_method('spawn', force=True)
        run_global_training(num_workers=4, total_episodes_per_worker=1500, log_action_probability=0.05) 
        print("训练脚本已完成。")
