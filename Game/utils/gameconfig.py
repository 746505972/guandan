# 2026/8/1 20:57
import random
from dataclasses import dataclass
from typing import Literal


@dataclass
class GameConfig:
    player_0: Literal['Human', 'AI', 'Script'] = 'Script'
    player_1: Literal['Human', 'AI', 'Script'] = 'Script'
    player_2: Literal['Human', 'AI', 'Script'] = 'Script'
    player_3: Literal['Human', 'AI', 'Script'] = 'Script'
    active_level: int = random.choice(range(2, 15))  # 级牌
    verbose: bool = True  # 是否输出文本信息
    print_history: bool = False  # 是否打印历史记录
    test: bool = False  # 是否为测试模式
    model: str = 'base.pth'
    sug_len: int = 3 # 建议个数

