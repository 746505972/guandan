import streamlit as st
from test import GuandanGame,M
import os

def convert_card_display(card_str):
    """修改手牌显示"""
    suit_symbols = {'黑桃': '♠️', '红桃': '♥️', '梅花': '♣️', '方块': '♦️'}
    if card_str in ['大王']:
        return '大王🃏'
    if card_str in ['小王']:
        return '小王🃟'
    for cn_suit, symbol in suit_symbols.items():
        if card_str.startswith(cn_suit):
            return card_str.replace(cn_suit, symbol)
    return card_str

st.set_page_config(
    page_title="🤖 AI 掼蛋对战演示",  # 浏览器标签页标题
    layout="wide"  # 可选宽布局
)

# 初始化页面状态
if "page" not in st.session_state:
    st.session_state.page = "setup"

# ============ 页面一：设置界面 ============
if st.session_state.page == "setup":
    st.title("🧠 设置你的 AI 掼蛋对战")
    st.markdown("### 请在下方选择要使用的模型和你的位置：")
    st.markdown(">推荐使用`show`开头版本")

    model_dir = "models"
    available_models = [f for f in os.listdir(model_dir) if f.endswith(".pth") and (f.startswith("a") or f.startswith("s"))]

    default_model = "show2.pth"
    if default_model in available_models:
        available_models.remove(default_model)
        available_models.insert(0, default_model)

    selected_model = st.selectbox("请选择模型：", available_models, key="model_select")
    selected_position = st.selectbox("你的位置（玩家号）：", [1, 2, 3, 4], index=0, key="position_select")

    if st.button("✅ 确认设置并开始游戏"):
        st.session_state.selected_model = selected_model
        st.session_state.selected_position = selected_position
        selected_model_path = os.path.join(model_dir, selected_model)
        st.session_state.game = GuandanGame(
            user_player=selected_position,
            verbose=False,model_path=selected_model_path)
        st.session_state.selected_indices = []
        st.session_state.page = "main"
        st.rerun()

    if st.button("联机大厅",
                 disabled=False):
        st.session_state.page = "multi_setup"
        st.rerun()

# ============ 页面二：主界面（游戏） ============
elif st.session_state.page == "main":
    game: GuandanGame = st.session_state.game  # 类型提示

    # 显示排名
    if game.is_game_over:
        st.success("🎉 游戏结束！")
        st.markdown("**最终排名：**")
        ranks = ["头游", "二游", "三游", "末游"]
        for i, p in enumerate(game.ranking):
            st.markdown(f"- {ranks[i]}：玩家 {p + 1}")
    # 分成两列：主区域（70%） 和 侧边栏区域（30%）
    main_col, history_col = st.columns([3, 1])


    with (main_col):
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
                is_self = (i == game.user_player)
                is_last = (i == game.last_player)
                hand_count = status['hand_size']
                last_play = " ".join(status['last_play']) if status['last_play'] else "Pass"

                # 样式：背景 + 字体颜色
                bg_color = "#fffae6" if is_last else "#f5f5f5"
                hand_color = "red" if hand_count < 4 else "#000"

                # 玩家名称
                player_label = f"玩家 {status['id']}" + ("🧑‍💻" if is_self else "")

                # 已出完的显示
                if i in game.ranking:
                    rank_index = game.ranking.index(i)
                    ranks = ["🏅头游", "🥈二游", "🥉三游", "🛑末游"]
                    content_html = f"""
                    <div style="background-color: {bg_color}; padding: 12px; border-radius: 8px; text-align: center;">
                        <div style="font-weight: bold; font-size: 18px;">{player_label}</div>
                        <div style="color: green;">已出完（{ranks[rank_index]}）</div>
                    </div>
                    """
                else:
                    content_html = f"""
                    <div style="background-color: {bg_color}; padding: 12px; border-radius: 8px; text-align: center;">
                        <div style="font-weight: bold; font-size: 18px;">{player_label}</div>
                        <div style="color: {hand_color}; font-size: 16px;">手牌：{hand_count} 张</div>
                        <div>出牌：{last_play}</div>
                    </div>
                    """

                st.markdown(content_html, unsafe_allow_html=True)

        # AI建议与上轮出牌类型
        last_play_type = game.map_cards_to_action(game.last_play,M,game.active_level)["type"]
        last_play_str = "、".join(game.last_play) if game.last_play else "无"

        ai_html = f"""
        <div style="background-color: #e3f2fd; border-radius: 10px; padding: 20px; margin-top: 20px; display: flex; justify-content: space-between; gap: 20px;">
            <div style="flex: 3;">
                <div style="font-weight: bold; font-size: 16px; margin-bottom: 8px;">🤖 AI 建议：</div>
                <div style="line-height: 1.8;">
        """
        for sug in ai_suggestions:
            ai_html += f"""<div style="margin-top: 6px; background: #f8f9fa; padding: 6px 8px; 
            border-radius: 6px; color: #333; font-size: 14px;">{sug}</div>"""

        ai_html += """
                </div>
            </div>
            <div style="flex: 1;">
                <div style="font-weight: bold; font-size: 16px; margin-bottom: 8px;">📦 上次出牌</div>
                <div>类型：<strong>{}</strong></div>
                <div style="margin-top: 6px; background: #f8f9fa; padding: 6px 8px; border-radius: 6px; color: #333; font-size: 14px;">
                    {}
                </div>
            </div>
        </div>
        """.format(last_play_type, last_play_str)

        st.markdown(ai_html, unsafe_allow_html=True)


        # 玩家行动
        if not game.is_game_over and game.current_player == game.user_player:
            if game.user_player not in game.ranking:
                st.markdown(f"****🕹️ 出牌****")

                # 初始化用户已选牌
                if "selected_indices" not in st.session_state:
                    st.session_state.selected_indices = []

                # 显示手牌按钮
                hand_cols = st.columns(min(9, len(user_hand)),gap='small')  # 每行最多8张牌
                for idx, card in enumerate(user_hand):
                    col = hand_cols[idx % len(hand_cols)]
                    with col:
                        is_selected = idx in st.session_state.selected_indices
                        card_display = convert_card_display(card)
                        if st.button(
                                f"{card_display}" if is_selected else card_display,
                                key=f"card_btn_{idx}",
                                type="primary" if is_selected else "secondary",
                                use_container_width=True
                        ):
                            if is_selected:
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
                btn_col1, btn_col2,btn_col3,btn_col4 = st.columns([1,1,1,1])
                with btn_col1:
                    if st.button("🗑️清空选择", use_container_width=True):
                        st.session_state.selected_indices = []
                        st.rerun()
                with btn_col2:
                    if st.button(
                            "👟PASS",
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
                            "✔️确认出牌",
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
                with btn_col4:
                    if st.button(
                            "🤖自动",
                            use_container_width=True
                    ):
                        step_result = game.step()
                        st.session_state.selected_indices = []
                        st.rerun()
            else:
                if not game.is_game_over:
                    while True:
                        step_result = game.step()
                        if step_result.get("waiting_for_user") or step_result.get("game_over"):
                            break
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
        col1, col2,col3 = st.columns([1,1,1])
        # 开启新一局按钮
        with col1:
            if st.button("🔄新一局"):
                # 重置游戏状态
                selected_model_path = os.path.join("models", st.session_state.selected_model)
                st.session_state.game = GuandanGame(
                    user_player=st.session_state.selected_position,
                    verbose=False,model_path=selected_model_path)
                st.session_state.selected_indices = []
                st.rerun()
        with col2:
            if st.button("返回设置"):
                st.session_state.page = "setup"
                st.rerun()
        # GitHub 链接（带图标）
        with col3:
            github_html = """
            <a href="https://github.com/746505972/guandan" target="_blank" style="text-decoration: none;">
                <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" 
                     width="20" style="vertical-align: middle; margin-right: 6px;">
                <span style="font-size: 12px; vertical-align: middle;">查看项目仓库</span>
            </a>
            """
            st.markdown(github_html, unsafe_allow_html=True)
            st.markdown('![Static Badge](https://img.shields.io/badge/ver.-1.2.3-E85889)')
        # 显示级牌
        st.markdown(f"""
            <div style="display: flex; gap: 20px; align-items: center; margin-bottom: 15px;">
                <div>
                    <strong>当前级牌：</strong>
                    <span style="background-color: red; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 20px;">
                        {game.active_level}
                    </span>
                </div>
                <div style="width: 1px; height: 20px; background-color: #ccc;"></div>
                <div>
                    <strong>当前轮到：</strong>玩家 
                    <span style="color: orange; font-weight: bold; font-size: 16px;">
                        {game.current_player + 1}
                    </span>
                </div>
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
        <details style="margin-bottom: 5px;">
        <summary style="font-weight: bold; font-size: 14px; color: #000;">调试区</summary>
        <div style="margin-top: 5px; font-size: 16px; display: flex; flex-wrap: wrap; gap: 5px;">
            <code>is_free_turn: {game.is_free_turn}</code>
            <code>pass_count: {game.pass_count}</code>
            <code>jiefeng: {game.jiefeng}</code>
            <code>{game.model_path}</code>
            <code>1:{game.players[0].hand}</code>
            <code>{game.players[0].last_played_cards}</code>
            <code>2:{game.players[1].hand}</code>
            <code>{game.players[1].last_played_cards}</code>
            <code>3:{game.players[2].hand}</code>
            <code>{game.players[2].last_played_cards}</code>
            <code>4:{game.players[3].hand}</code>
            <code>{game.players[3].last_played_cards}</code>
        </div></details>""", unsafe_allow_html=True)
# ============ 页面三：多人设置 ============
elif st.session_state.page == "multi_setup":
    st.title("🕹️ 掼蛋联机大厅")

    model_dir = "models"
    available_models = [f for f in os.listdir(model_dir) if f.endswith(".pth") and (f.startswith("a") or f.startswith("s"))]

    # 房间号输入
    room_id = st.text_input("请输入房间号（任意字符串）", value=st.session_state.get("room_id", "room-001"))
    st.session_state.room_id = room_id

    # 初始化房间配置（本地模拟；正式应从后端加载）
    if "player_configs" not in st.session_state:
        st.session_state.player_configs = [{
            "model": available_models[0] if available_models else None,
            "selected": False
        } for _ in range(4)]

    if "joined_index" not in st.session_state:
        st.session_state.joined_index = None  # 尚未加入任何位置

    cols = st.columns(4)
    for i in range(4):
        with cols[i]:
            st.markdown(f"### 玩家 {i + 1}")

            if st.session_state.player_configs[i]["selected"]:
                if st.session_state.joined_index == i:
                    st.success("✅ 你已加入该座位")

                    if st.button("➖ 离开", key=f"swap_{i}"):
                        st.session_state.player_configs[i]["selected"] = False
                        st.session_state.joined_index = None
                        # TODO: 请求后端取消该玩家占用座位
                        st.rerun()

                else:
                    st.warning("已被占用")
            else:
                # AI 模型选择
                selected_model = st.selectbox("模型", available_models,
                                              index=available_models.index(
                                                  st.session_state.player_configs[i]["model"]),
                                              key=f"model_{i}")
                st.session_state.player_configs[i]["model"] = selected_model

                if st.session_state.joined_index is None:
                    if st.button("➕ 加入", key=f"join_{i}"):
                        st.session_state.player_configs[i]["selected"] = True
                        st.session_state.joined_index = i

                        # ✅ TODO: 请求后端 `/join_room`，上传 room_id、seat、model
                        # requests.post("http://后端地址/join_room", json={...})
                        st.rerun()
                else:
                    st.button("➕ 加入", key=f"join_{i}", disabled=True)

    st.markdown("---")

    # 如果当前用户是第一个加入的 → 房主
    if st.session_state.joined_index == 0:
        if st.button("🚀 开始游戏（房主）"):
            # ✅ TODO: 请求后端 /start_game 并跳转页面4
            st.success("游戏即将开始！（你是房主）")
            # st.session_state.page = "game"
            # st.rerun()
    else:
        st.markdown("等待房主开始游戏...")

    if st.button("🔙 离开房间"):
        # 清空状态
        if st.session_state.joined_index is not None:
            st.session_state.player_configs[st.session_state.joined_index]["selected"] = False
        st.session_state.joined_index = None
        st.session_state.page = "setup"
        st.rerun()





