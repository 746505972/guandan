"""手牌编码函数"""
import numpy as np

def build_card_index_map():
    """构建牌面到索引的映射"""
    index_map = {}
    idx = 0
    for copy in range(2):
        for suit in ['黑桃', '红桃', '梅花', '方块']:
            for rank in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']:
                card = f"{suit}{rank}"
                if card not in index_map:
                    index_map[card] = []
                index_map[card].append(idx)
                idx += 1
    for copy in range(2):
        index_map.setdefault('小王', []).append(idx)
        idx += 1
        index_map.setdefault('大王', []).append(idx)
        idx += 1
    return index_map


def encode_hand_108(hand):
    """将手牌编码为108维向量"""
    card_map = build_card_index_map()
    obs = np.zeros(108)
    if hand == ['Pass'] or hand == ['None'] or not hand:
        hand = []

    card_count = {}
    for card in hand:
        card_count[card] = card_count.get(card, 0) + 1

    for card, count in card_count.items():
        indices = card_map.get(card, [])
        for i in range(min(count, len(indices))):
            obs[indices[i]] = 1.0
    return obs


def find_entry_by_id(data, target_id):
    """返回匹配 id 的 JSON 对象"""
    for entry in data:
        if entry.get("id") == target_id:
            return entry
    return None


def plot_loss(loss_values, filename="critic_loss.png"):
    """绘制损失曲线"""
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, label='Critic Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Critic Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
    except ImportError:
        print(f"未找到 Matplotlib，跳过 {filename} 的绘图。")
    except Exception as e:
        print(f"绘制损失图时出错: {e}")
