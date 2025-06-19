<template>
  <div class="solo">
    <div class="header" style="width: 100%; margin-bottom: 1.5rem;">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="flex: 1;"></div> <!-- 左侧占位 -->
        <h2 style="text-align: center; flex: 1;">🤖 AI 掼蛋对战演示</h2>
        <div style="display: flex; align-items: center; gap: 0.5rem; justify-content: flex-end; flex: 1;">
          <a href="https://github.com/746505972/guandan" target="_blank" class="github-link">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="20">
            <span>项目仓库</span>
          </a>
          <div class="badge">ver. 1.4.1</div>
        </div>
      </div>
    </div>

    <div v-if="!gameData">加载中...</div>
    
    <div v-else class="game-container">
      <!-- 两列布局 - 主区域和侧边栏 -->
      <div class="main-content">
        <!-- 玩家状态 -->
        <div class="player-status-container">
          <div v-for="i in 4" :key="i" class="player-card" 
               :style="{ backgroundColor: i-1 === gameData.last_player ? '#ffe9b3' : '#f5f5f5' }"
               :class="{ 'current-player': i-1 === gameData.current_player }">
            <div class="player-name">
              玩家 {{ i }}{{ i-1 === gameData.user_player ? ' 🧑‍💻' : '' }}
              <span v-if="gameData.ranking?.includes(i-1)" class="player-rank">
                {{ getRankText(gameData.ranking.indexOf(i-1)) }}
              </span>
            </div>
            <div class="player-info">
              <div :style="{ color: getHandColor(i-1) }">手牌：{{ getHandSize(i-1) }} 张</div>
              <div>出牌：{{ getLastPlay(i-1) }}</div>
            </div>
          </div>
        </div>

        <!-- AI建议与上次出牌 -->
        <div class="ai-suggestion-container">
          <div class="suggestion-section" :class="{ 'two-columns': shouldUseTwoColumns }">
            <h3>🤖 AI建议：</h3>
            <ul>
              <li v-for="(sug, i) in gameData.ai_suggestions" :key="i">{{ sug }}</li>
            </ul>
          </div>
          <div class="last-play-section">
            <h3>📦 上次出牌</h3>
            <div>类型：<strong>{{ gameData.last_play_type }}</strong></div>
            <div class="last-play-cards">
              {{ gameData.last_play }}
            </div>
          </div>
        </div>

        <!-- 游戏结束显示 -->
        <div v-if="gameData.is_game_over" class="game-over-container">
          <h3>🎉 游戏结束！</h3>
          <p><strong>最终排名：</strong></p>
          <ul>
            <li v-for="(p, i) in gameData.ranking" :key="i">
              {{ ['头游', '二游', '三游', '末游'][i] }}：玩家 {{ p + 1 }}
            </li>
          </ul>
        </div>

        <!-- 玩家行动区域 -->
        <div v-if="!gameData.is_game_over" class="player-action-container">
          <h3>🕹️ 出牌</h3>
          
          <!-- 手牌选择 -->
          <div class="hand-cards-container">
            <div class="hand-cards" ref="handCards">
              <div
                v-for="(card, index) in gameData.user_hand"
                :key="index"
                @click="toggleSelect(index)"
                :class="['card', { 'selected-card': selected.includes(index) }]"
                :style="{
                  left: `${index * 30}px`, 
                  zIndex: index,
                  color: getCardColor(card)
                }"
              >
                <div class="card-content">
                  <template v-if="isJoker(card)">
                    <div class="joker-vertical-text">J<br>O<br>K<br>E<br>R</div>
                    <div class="joker-center">🃟</div>
                    <div class="joker-vertical-text bottom-right">J<br>O<br>K<br>E<br>R</div>
                  </template>
                  <template v-else>
                    <div class="card-corner top-left">
                      <div>{{ getCardRankDisplay(card) }}</div>
                      <div>{{ getCardSuitSymbol(card) }}</div>
                    </div>
                    <div class="card-center">
                      {{ getCardCenterDisplay(card) }}
                    </div>
                    <div class="card-corner bottom-right">
                      <div>{{ getCardRankDisplay(card) }}</div>
                      <div>{{ getCardSuitSymbol(card) }}</div>
                    </div>
                  </template>
                </div>
              </div>
            </div>
          </div>

          <!-- 已选牌显示 -->
          <div class="selected-cards-display">
            <strong>已选择：</strong>
            <span v-if="selectedCards.length > 0" class="has-selection">
              {{ selectedCards.map(card => convertCardDisplay(card)).join('、') }}
            </span>
            <span v-else-if="gameData.is_free_turn" class="no-selection">
              自由回合
            </span>
            <span v-else class="no-selection">
              尚未选择任何牌
            </span>
          </div>

          <!-- 操作按钮 -->
          <div v-if="gameData.current_player === gameData.user_player" class="action-buttons">
            <button @click="clearSelection" class="secondary-btn">🗑️ 清空选择</button>
            <button @click="pass" :disabled="gameData.is_free_turn" class="secondary-btn">👟 PASS</button>
            <button @click="submitMove" class="primary-btn">✔️ 确认出牌</button>
            <button @click="autoPlay" class="secondary-btn">🤖 自动</button>
          </div>
        </div>
      </div>

      <!-- 侧边栏 -->
      <div class="sidebar">
        <!-- 操作按钮网格 -->
        <div class="sidebar-grid">
          <button @click="newGame" class="sidebar-btn">🔄 新一局</button>
          <button @click="goBack" class="sidebar-btn">🔙 返回设置</button>
        </div>
        


        <!-- 当前状态 -->
        <div class="current-status">
          <div class="status-item">
            <strong>当前级牌：</strong>
            <span class="level-card">{{ gameData.active_level }}</span>
          </div>
          <div class="status-divider"></div>
          <div class="status-item">
            <strong>当前轮到：</strong>
            <span class="current-player">玩家 {{ gameData.current_player + 1 }}</span>
          </div>
        </div>

        <div class="card-tracker">
          <div class="card-tracker-grid">
            <!-- 王牌 -->
            <div class="card-tracker-item">
              <div class="card-tracker-label">大王:</div>
              <div class="card-tracker-count">{{ 2-remainingCards['大王'] || 0 }}</div>
            </div>
            <div class="card-tracker-item">
              <div class="card-tracker-label">小王:</div>
              <div class="card-tracker-count">{{ 2-remainingCards['小王'] || 0 }}</div>
            </div>
            
            <!-- 普通牌 -->
            <div v-for="(value, index) in ['2','3','4','5','6','7','8','9','10','J','Q','K','A']" :key="index" class="card-tracker-item">
              <div class="card-tracker-label">{{ value }}:</div>
              <div class="card-tracker-count">{{ 8-remainingCards[value] || 0 }}</div>
            </div>
          </div>
        </div>
        <!-- 出牌历史 -->
        <div class="play-history">
          <h3>📝 出牌历史</h3>
          <textarea :value="formattedHistory" readonly></textarea>
        </div>

        <!-- 调试信息 -->
        <details class="debug-info">
          <summary>调试区</summary>
          <div class="debug-content">
            <code>is_free_turn: {{ gameData.is_free_turn }}</code>
            <code>pass_count: {{ gameData.pass_count }}</code>
            <code>jiefeng: {{ gameData.jiefeng }}</code>
            <code>{{ gameData.model_path }}</code>
            <!-- 其他调试信息可以根据需要添加 -->
          </div>
        </details>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { onMounted, ref, computed, watch } from 'vue'
import { useGlobalStore } from '../stores'
import { api } from '../utils/axios'
import { useRouter } from 'vue-router'

const store = useGlobalStore()
const router = useRouter()
const gameData = ref<any>(null)
const selected = ref<number[]>([])
const isAutoPlaying = ref(false)
//const aiAdvice = ref<string>(''); // 新增：存储AI建议

const refreshState = async () => {
  if (!store.userId) {
    console.error('userId 为空，无法获取游戏状态');
    return;
  }
  try {
    const res = await api.get(`/solo_state/${store.userId}`, {
      headers: {'ngrok-skip-browser-warning': 'true'}
    });
    gameData.value = res.data;
    // 调用工作流获取建议
    //await fetchAIAdvice();
    console.log('刷新后的游戏状态:', gameData.value);
    
    // 无论是否轮到玩家都尝试自动推进
    if (!gameData.value.is_game_over) {
      autoAdvanceGame();
    }
  } catch (e) {
    console.error('获取游戏状态失败', e);
  }
};

// 自动推进游戏
const autoAdvanceGame = async () => {
  if (isAutoPlaying.value || !gameData.value) return;
  
  isAutoPlaying.value = true;
  try {
    // 添加最大尝试次数防止死循环
    let maxAttempts = 20;
    let attempts = 0;
    
    while (
      !gameData.value.is_game_over && 
      gameData.value.current_player !== gameData.value.user_player && 
      attempts < maxAttempts
    ) {
      attempts++;
      
      // 先获取最新状态
      const state = await api.get(`/solo_state/${store.userId}`, {
        headers: { 'ngrok-skip-browser-warning': 'true' }
      });
      gameData.value = state.data;
      
      // 如果还是AI回合才执行自动出牌
      if (!gameData.value.is_game_over && 
          gameData.value.current_player !== gameData.value.user_player) {
        await api.post('/solo_autoplay', { user_id: store.userId }, {
          headers: { 'ngrok-skip-browser-warning': 'true' }
        });
        
        // 再次获取更新后的状态
        const newState = await api.get(`/solo_state/${store.userId}`, {
          headers: { 'ngrok-skip-browser-warning': 'true' }
        });
        gameData.value = newState.data;
      }
      
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  } catch (e) {
    console.error('自动推进出错:', e);
  } finally {
    isAutoPlaying.value = false;
  }
};

// const fetchAIAdvice = async () => {
//   try {
//     const response = await api.post('/get_ai_advice', {
//       user_id: store.userId,
//       game_state: gameData.value
//     }, {
//       headers: {'ngrok-skip-browser-warning': 'true'}
//     });
//     aiAdvice.value = response.data.advice || '暂无建议';
//   } catch (e) {
//     console.error('获取AI建议失败', e);
//     aiAdvice.value = '建议获取失败';
//   }
// };

// 计算剩余牌数
const remainingCards = computed(() => {
  if (!gameData.value?.hand) return {};
  
  const cards: Record<string, number> = {};
  
  // 初始化牌库 (2副牌)
  // 王牌
  cards['大王'] = 2;
  cards['小王'] = 2;
  
  const normalCards = ['2','3','4','5','6','7','8','9','10','J','Q','K','A'];
  normalCards.forEach(card => {
    cards[card] = 8;
  });

  // 统计所有玩家手牌
  const allPlayerCards: string[] = [];
  gameData.value.hand.forEach((playerHand: string[]) => {
    allPlayerCards.push(...playerHand);
  });

  // 从总牌库中扣除玩家手牌
  allPlayerCards.forEach((card: string) => {
    let cardKey;
    if (card === '大王' || card === '小王') {
      cardKey = card;
    } else {
      // 去掉花色前缀，例如"黑桃3" -> "3"
      cardKey = card.slice(2);
    }
    
    if (cards[cardKey] !== undefined && cards[cardKey] > 0) {
      cards[cardKey]--;
    } else {
      console.warn(`无效或重复扣除的牌: ${card}`);
    }
  });
  
  return cards;
});


// 监听游戏数据变化，检查是否需要自动推进
watch(() => gameData.value, (newVal) => {
  if (newVal && !newVal.is_game_over && 
      newVal.current_player !== newVal.user_player) {
    autoAdvanceGame();
  }
});

// 其他已有的方法保持不变...
const getHandColor = (i: number) => 
  gameData.value?.hand_size?.[i] < 3 ? 'red' : 'black'

const getHandSize = (i: number) => 
  gameData.value?.hand_size?.[i] ?? gameData.value?.hand?.[i]?.length ?? "unknown"

const getLastPlay = (i: number) => 
  gameData.value?.last_plays?.[i]?.join(' ') ?? gameData.value?.last_play_history?.[i] ?? 'unknown'

const getRankText = (rankIndex: number) => {
  const ranks = ["🏅头游", "🥈二游", "🥉三游", "🛑末游"];
  return ranks[rankIndex];
}

const toggleSelect = (idx: number) => {
  if (selected.value.includes(idx)) {
    selected.value = selected.value.filter(i => i !== idx)
  } else {
    selected.value.push(idx)
  }
}

const clearSelection = () => {
  selected.value = [];
}

const selectedCards = computed(() => 
  selected.value.map(i => gameData.value.hand[gameData.value.user_player][i])
)

const formattedHistory = computed(() => {
  if (!gameData.value.history) return '';
  return gameData.value.history.map((round: any[], i: number) => {
    const roundNumber = gameData.value.history.length - i;
    return `第${roundNumber}轮: ` + round.map(p => p ? p.join(' ') : 'Pass').join(' | ');
  }).join('\n');
})

const convertCardDisplay = (cardStr: string) => {
  const suitSymbols = {'黑桃': '♠️', '红桃': '♥️', '梅花': '♣️', '方块': '♦️'};
  if (cardStr === '大王') return '大王🃏';
  if (cardStr === '小王') return '小王🃟';
  for (const [cnSuit, symbol] of Object.entries(suitSymbols)) {
    if (cardStr.startsWith(cnSuit)) {
      return cardStr.replace(cnSuit, symbol);
    }
  }
  return cardStr;
}
const getCardRankDisplay = (cardStr: string) => {
  if (cardStr === '大王') return 'JOKER';
  if (cardStr === '小王') return 'JOKER';
  return cardStr.slice(2); // 去掉花色前缀，例如"黑桃3" -> "3"
}

const getCardSuitSymbol = (cardStr: string) => {
  const suitSymbols = {'黑桃': '♠', '红桃': '♥', '梅花': '♣', '方块': '♦'};
  if (cardStr === '大王' || cardStr === '小王') return '🃏';
  for (const [cnSuit, symbol] of Object.entries(suitSymbols)) {
    if (cardStr.startsWith(cnSuit)) {
      return symbol;
    }
  }
  return '';
}

const getCardCenterDisplay = (cardStr: string) => {
  if (cardStr === '大王') return '大王';
  if (cardStr === '小王') return '小王';
  return getCardSuitSymbol(cardStr);
}

const getCardColor = (cardStr: string) => {
  if (cardStr === '大王') return 'red';
  if (cardStr === '小王') return 'black';
  if (isLevelCard(cardStr)) {
    return '#ffb700';
  }
  if (cardStr.startsWith('红桃') || cardStr.startsWith('方块')) return 'red';
  return 'black';
}

const isJoker = (cardStr: string) => {
  return cardStr === '大王' || cardStr === '小王';
}

const isLevelCard = (cardStr: string) => {
  return gameData.value.active_level && cardStr.includes(gameData.value.active_level);
}

const submitMove = async () => {
  try {
    const response = await api.post('/solo_play_card', {
      user_id: store.userId,
      cards: selectedCards.value
    }, {
      headers: {'ngrok-skip-browser-warning': 'true'}
    });
    selected.value = [];
    await refreshState(); // 确保等待状态刷新完成
    // 如果返回error，前端提示
    if (response.data.error) {
      alert('出牌失败，' + response.data.error);
    }
  } catch (e) {
    console.error('出牌失败', e);
  }
}

const pass = async () => {
  try {
    await api.post('/solo_play_card', {
      user_id: store.userId,
      cards: []
    }, {
      headers: {'ngrok-skip-browser-warning': 'true'}
    });
    selected.value = [];
    await refreshState(); // 确保等待状态刷新完成
  } catch (e) {
    console.error('PASS失败', e);
  }
}

const autoPlay = async () => {
  selected.value = [];
  await api.post('/solo_autoplay', { user_id: store.userId },{headers: {'ngrok-skip-browser-warning': 'true'}})
  refreshState()
}

const newGame = async () => {
  selected.value = [];
  await api.post('/solo_new_game', { user_id: store.userId , model: store.selectedModel ,position: store.joinedSeat},
  {headers: {'ngrok-skip-browser-warning': 'true',
    'Content-Type': 'application/json'
  }})
  refreshState()
}

const goBack = () => {
  console.log('返回设置页面');
  router.push('/')
}
// 是否使用双列布局（AI建议超过3条时）
const shouldUseTwoColumns = computed(() => {
  return gameData.value?.ai_suggestions?.length > 3;
});
onMounted(refreshState)
</script>

<style scoped>
.solo {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  padding: 1rem clamp(1rem, 5%, 3rem); /* 最小1rem，最大3rem，5%视口宽度 */
}

.game-container {
  display: flex;
  width: 100%;
  gap: 1.5rem;
}

.main-content {
  flex: 3;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.sidebar {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

/* 玩家状态 */
.player-status-container {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin-bottom: 0rem;
}

.player-card {
  padding: 1rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.player-name {
  font-weight: bold;
  font-size: 1.1rem;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.player-rank {
  color: green;
  font-size: 0.9rem;
}

.player-info {
  font-size: 0.9rem;
}

/* AI建议区域 */
.ai-suggestion-container {
  display: flex;
  gap: 20px;
  margin-top: 5px;
  background-color: #e6f2ff;  /* 浅蓝色背景 */
  padding: 20px;
  border-radius: 10px;
}

/* 2:1的比例布局 */
.suggestion-section {
  flex: 2;  /* 2份宽度 */
  background: rgba(255, 255, 255, 0.8); /* 浅白色半透明背景 */
  padding: 15px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.last-play-section {
  flex: 1;  /* 1份宽度 */
  background: rgba(255, 255, 255, 0.8); /* 浅白色半透明背景 */
  padding: 15px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.suggestion-section h3,
.last-play-section h3 {
  margin-top: 0;
  color: #333;
  border-bottom: 1px solid #eee;
  padding-bottom: 8px;
}

.suggestion-section ul {
  padding-left: 20px;
  margin: 10px 0 0;
}

.suggestion-section li {
  margin-bottom: 8px;
  line-height: 1.4;
}

/* 双列布局样式 */
.suggestion-section.two-columns ul {
  column-count: 2;
  column-gap: 20px;
}

.last-play-cards {
  margin-top: 10px;
  padding: 10px;
  background: #fff;
  border-radius: 4px;
  border: 1px solid #ddd;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .ai-suggestion-container {
    flex-direction: column;
  }
  
  .suggestion-section,
  .last-play-section {
    flex: none;
    width: 100%;
  }
  
  .suggestion-section.two-columns ul {
    column-count: 1;
  }
}

.last-play-section {
  flex: 1;
}

.last-play-section h3 {
  margin-top: 0;
  margin-bottom: 0.75rem;
  font-size: 1.1rem;
}

.last-play-cards {
  background: #f8f9fa;
  padding: 0.5rem;
  border-radius: 6px;
  margin-top: 0.5rem;
}

/* 手牌区域 */
.hand-cards-container {
  position: relative;
  height: 200px;
  margin-bottom: 1.5rem;
}

.hand-cards {
  position: relative;
  height: 100%;
}

.hand-section {
  margin-bottom: 5%; /* 移除默认底部边距 */
}

.hand-cards-container {
  height: 120px; /* 固定高度避免过大空白 */
}

.card {
  position: absolute;
  width: 80px;
  height: 120px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  cursor: pointer;
  transition: all 0.3s ease;
  transform-origin: bottom center;
}

.card:hover {
  transform: translateY(-10px) scale(1.05);
  /* 移除 z-index 修改 */
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
}

.card.selected-card {
  transform: translateY(-40px) !important;
  box-shadow: 0 0 15px rgba(95, 95, 95, 0.5);
}

.card-content {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 8px;
}

.card-corner {
  font-size: 12px;
  font-weight: bold;
  text-align: center;
}

.card-corner.top-left {
  align-self: flex-start;
}

.card-corner.bottom-right {
  position: absolute;
  right: 4px;
  bottom: 4px;
  transform: rotate(180deg);
  font-size: 12px;
  line-height: 1.2;
  text-align: center;
}

.card-center {
  font-size: 36px;
  text-align: center;
  margin: auto;
}
/* Joker牌专用样式 */
.joker-vertical-text {
  position: absolute;
  font-size: 12px;
  font-weight: bold;
  line-height: 1.2;
  letter-spacing: 1px;
}

.joker-vertical-text.bottom-right {
  right: 8px;
  bottom: 8px;
  transform: rotate(180deg);
}

.joker-center {
  position: absolute;
  font-size: 48px;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}

/* 响应式调整 */
@media (max-width: 768px) {
  .joker-center {
    font-size: 36px;
  }
}

/* 响应式调整 */
@media (max-width: 768px) {
  .card {
    width: 60px;
    height: 90px;
  }
  
  .card-center {
    font-size: 24px;
  }
}

/* 已选牌显示 */
.selected-cards-display {
  border: 1px solid #e6e6e6;
  padding: 0.75rem;
  border-radius: 5px;
  background-color: #f9f9f9;
  margin-bottom: 1rem;
}

.has-selection {
  color: #2e7d32;
  font-weight: bold;
}

.no-selection {
  color: gray;
  font-weight: bold;
}

/* 操作按钮 */
.action-buttons {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.action-buttons button {
  padding: 0.5rem 0.75rem;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.9rem;
  flex: 1;
  min-width: 120px;
}

.primary-btn {
  background-color: #4CAF50;
  color: white;
  border: none;
}

.secondary-btn {
  background-color: #f5f5f5;
  border: 1px solid #ccc;
}

.action-buttons button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* 侧边栏样式 */
.sidebar-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.5rem;
}

.sidebar-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 0.5rem;
  padding-top: 0.5rem;
  border-top: 1px solid #eee;
}

.sidebar-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.sidebar-btn {
  padding: 0.5rem;
  border-radius: 6px;
  background-color: #f5f5f5;
  border: 1px solid #ccc;
  cursor: pointer;
  font-size: 0.9rem;
  flex: 1;
  min-width: 120px;
}

.github-link {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  text-decoration: none;
  color: #333;
  font-size: 0.9rem;
}

.badge {
  background-color: #E85889;
  color: white;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.8rem;
}

.current-status {
  display: flex;
  align-items: center;
  gap: 1.25rem;
  margin: 0.1rem 0;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.level-card {
  background-color: red;
  color: white;
  padding: 0.25rem 0.5rem;
  border-radius: 6px;
  font-weight: bold;
  font-size: 1.1rem;
}

.current-player {
  color: orange;
  font-weight: bold;
  font-size: 1rem;
}

.status-divider {
  width: 1px;
  height: 1.5rem;
  background-color: #ccc;
}

.play-history textarea {
  width: 100%;
  height: 350px;
  resize: none;
  padding: 0.5rem;
  border-radius: 6px;
  border: 1px solid #ccc;
  font-family: monospace;
  font-size: 0.8rem;
}

/* 调试信息 */
.debug-info {
  margin-top: 1rem;
}

.debug-info summary {
  font-weight: bold;
  font-size: 0.9rem;
  cursor: pointer;
}

.debug-content {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.5rem;
  font-size: 0.8rem;
}

.debug-content code {
  background-color: #f5f5f5;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-family: monospace;
}

/* 游戏结束样式 */
.game-over-container {
  background-color: #e8f5e9;
  padding: 1.5rem;
  border-radius: 8px;
  margin-top: 1rem;
}

.game-over-container h3 {
  margin-top: 0;
  color: #2e7d32;
}

/* 当前玩家发光效果 */
.player-card.current-player {
  position: relative;
  box-shadow: 0 0 10px 3px rgba(255, 215, 0, 0.7);
  animation: pulse-glow 1.5s infinite alternate;
  z-index: 1;
}

@keyframes pulse-glow {
  0% {
    box-shadow: 0 0 5px 2px rgba(0, 215, 0, 0.055);
  }
  100% {
    box-shadow: 0 0 15px 5px rgba(0, 215, 0, 0.9);
  }
}

@media (max-width: 992px) {
  .game-container {
    flex-direction: column;
  }
  
  .player-status-container {
    grid-template-columns: repeat(2, 1fr);
  }
}

.card-tracker {
  background-color: #f8f9fa;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 1rem;
}

.card-tracker h3 {
  margin-top: 0;
  margin-bottom: 0.75rem;
}

.card-tracker-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 0.5rem;
}

.card-tracker-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-tracker-label {
  font-weight: bold;
}

.card-tracker-count {
  background-color: #e9ecef;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  min-width: 2rem;
  text-align: center;
}

@media (max-width: 768px) {
  .card-tracker-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}
</style>