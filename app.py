import time

import streamlit as st
from test import GuandanGame
import random

# 初始化游戏
if "game" not in st.session_state:
    st.session_state.game = GuandanGame(user_player=1, verbose=False, print_history=False)
game: GuandanGame = st.session_state.game  # 类型提示
st.set_page_config(
    page_title="🤖 AI 掼蛋对战演示",  # 浏览器标签页标题
    layout="wide"  # 可选宽布局
)

# 显示排名
if game.is_game_over:
    st.success("🎉 游戏结束！")
    st.markdown("**最终排名：**")
    ranks = ["头游", "二游", "三游", "末游"]
    for i, p in enumerate(game.ranking):
        st.markdown(f"- {ranks[i]}：玩家 {p + 1}")
# 分成两列：主区域（70%） 和 侧边栏区域（30%）
main_col, history_col = st.columns([3, 1])


with main_col:
    # 当前状态
    state = game.get_game_state()
    user_hand = state["user_hand"]
    last_play = state["last_play"]
    ai_suggestions = game.get_ai_suggestions()
    recent_actions = state["recent_actions"]

    cols = st.columns(4)
    statuses = game.get_player_statuses()
    for i in range(4):
        with cols[i]:
            status = statuses[i]
            st.markdown(f"**玩家 {status['id']}**")

            if i in game.ranking:
                rank_index = game.ranking.index(i)
                ranks = ["🏅头游", "🥈二游", "🥉三游", "🛑末游"]
                st.markdown(f":green[已出完]（{ranks[rank_index]}）")
            else:
                st.markdown(f"手牌: **{status['hand_size']}** 张")
                st.markdown("出牌：" + " ".join(status['last_play']))

    # 显示 AI 建议
    sug_html = "<div style='background-color:#e3f2fd; padding:15px; border-radius:8px;'>"
    sug_html += "<strong>🤖 AI 建议：</strong><br>"
    for sug in ai_suggestions:
        sug_html += f"• {sug}<br>"
    sug_html += "</div>"
    st.markdown(sug_html, unsafe_allow_html=True)

    # 玩家行动
    if not game.is_game_over and game.current_player == game.user_player:
        st.markdown(f"****🕹️ 出牌****")

        # 初始化用户已选牌
        if "selected_indices" not in st.session_state:
            st.session_state.selected_indices = []

        # 显示手牌按钮
        hand_cols = st.columns(min(10, len(user_hand)),gap='small')  # 每行最多8张牌
        for idx, card in enumerate(user_hand):
            col = hand_cols[idx % len(hand_cols)]
            with col:
                # 判断当前索引是否在选中列表中
                is_selected = idx in st.session_state.selected_indices
                if st.button(
                        f"{card}" if is_selected else card,
                        key=f"card_btn_{idx}",
                        type="primary" if is_selected else "secondary",
                        use_container_width=True
                ):
                    if is_selected:
                        # 通过索引精准移除（避免重复牌问题）
                        st.session_state.selected_indices.remove(idx)
                    else:
                        st.session_state.selected_indices.append(idx)
                    st.rerun()

        # 从索引转换实际牌面显示
        selected_cards = [user_hand[i] for i in sorted(st.session_state.selected_indices)]

        # 显示已选牌
        if selected_cards:
            st.markdown(
                f"""<div style='border:1px solid #e6e6e6; padding:10px; border-radius:5px; 
                background-color:#f9f9f9; margin-bottom:15px;'>
                <strong>已选择：</strong> <span style='color:#2e7d32; font-weight:bold'>
                {'、'.join(selected_cards)}</span></div>""",
                unsafe_allow_html=True
            )
        else:
            if game.is_free_turn:
                st.markdown(
                    f"""<div style='border:1px solid #e6e6e6; padding:10px; border-radius:5px; 
                    background-color:#f9f9f9; margin-bottom:15px;'>
                    <strong>已选择：</strong> <span style='color:gray; font-weight:bold'>
                    {'自由回合'}</span></div>""",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"""<div style='border:1px solid #e6e6e6; padding:10px; border-radius:5px; 
                                    background-color:#f9f9f9; margin-bottom:15px;'>
                                    <strong>已选择：</strong> <span style='color:gray; font-weight:bold'>
                                    {'尚未选择任何牌'}</span></div>""",
                    unsafe_allow_html=True)

        # 操作按钮组
        btn_col1, btn_col2,btn_col3 = st.columns([1,1,1])
        with btn_col1:
            if st.button("清空选择", use_container_width=True):
                st.session_state.selected_indices = []
                st.rerun()
        with btn_col2:
            if st.button(
                    "PASS",
                    use_container_width=True,
                    disabled=game.is_free_turn
            ):
                # 通过索引获取实际牌组
                move = []
                result = game.submit_user_move(move)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state.selected_indices = []  # 清空选择
                    st.rerun()
        with btn_col3:
            if st.button(
                    "确认出牌",
                    type="primary",
                    use_container_width=True
            ):
                # 通过索引获取实际牌组
                move = [user_hand[i] for i in st.session_state.selected_indices]
                result = game.submit_user_move(move)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state.selected_indices = []  # 清空选择
                    st.rerun()

    else:
        # 非用户轮次，自动推进
        if not game.is_game_over:
            while True:
                step_result = game.step()
                if step_result.get("waiting_for_user") or step_result.get("game_over"):
                    break
            st.rerun()



with history_col:
    col1, col2 = st.columns([1,1])
    # 开启新一局按钮
    with col1:
        if st.button("🔄新一局"):
            # 重置游戏状态
            st.session_state.game = GuandanGame(user_player=1, verbose=False, print_history=False)
            st.session_state.selected_indices = []
            st.rerun()
    # GitHub 链接（带图标）
    with col2:
        github_html = """
        <a href="https://github.com/746505972/guandan" target="_blank" style="text-decoration: none;">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" 
                 width="20" style="vertical-align: middle; margin-right: 6px;">
            <span style="font-size: 12px; vertical-align: middle;">查看项目仓库</span>
        </a>
        """
        st.markdown(github_html, unsafe_allow_html=True)
        st.markdown('![Static Badge](https://img.shields.io/badge/ver.-1.0.0(Beta)-orange)')
    # 显示级牌
    st.markdown(f"""
    <div style="display: flex; gap: 20px; align-items: center; margin-bottom: 15px;">
        <div><strong>当前级牌：</strong><span style="color: red;">{game.active_level}</span></div>
        <div style="width: 1px; height: 20px; background-color: #ccc;"></div>
        <div><strong>当前轮到：</strong>玩家 <span style="color: orange;">{game.current_player + 1}</span></div>
    </div>
    """, unsafe_allow_html=True)
    # 显示游戏历史
    history_lines = []
    for i, round in enumerate(reversed(state["history"])):
        round_number = len(state["history"]) - i
        line = f"第{round_number}轮: " + " | ".join([" ".join(p) if p else "Pass" for p in round])
        history_lines.append(line)
    history_text = "\n".join(history_lines)
    st.text_area("📝 出牌历史", value=history_text, height=350, disabled=True)
    st.markdown(f"""
    <div style="display: flex; gap: 0px; align-items: center; margin-bottom: 0px;">
        <div><span style="color: #000000;">调试区</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"`is_free_turn:{game.is_free_turn}`,`pass_count:{game.pass_count}`,`jiefeng:{game.jiefeng}`,"
                f"`last_play:{game.last_play}`,`last_player:{game.last_player}`")

