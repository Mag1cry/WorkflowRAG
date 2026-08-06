"""上下文加载：从 LangGraph Thread 中提取并截取最近 k 轮对话。

new_k 定义：
    "最近 k 轮 User → AIResponse"。
    不是 checkpoint 数量，不是 message 数量。
"""

from langgraph.checkpoint.base import BaseCheckpointSaver


def load_context(
    checkpointer: BaseCheckpointSaver,
    thread_id: str,
    new_k: int = 3,
) -> list:
    """从 LangGraph Thread 加载最近 k 轮对话。

    Args:
        checkpointer: LangGraph Checkpointer 实例。
        thread_id: LangGraph Thread ID。
        new_k: 保留最近 k 轮（User → AIResponse 对）。

    Returns:
        List[BaseMessage] — 最近 k 轮的所有消息。
    """
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    # 获取最新 checkpoint 状态
    result = checkpointer.get_tuple(config)
    if result is None:
        return []

    channel_values = result.checkpoint.get("channel_values", {})
    messages = list(channel_values.get("messages", []))
    if not messages:
        return []

    # 按 User → AIResponse 分组为"轮"
    rounds = _group_into_rounds(messages)

    # 保留最新 k 轮
    recent_rounds = rounds[-new_k:] if len(rounds) > new_k else rounds

    # 展平为消息列表
    result_list: list = []
    for r in recent_rounds:
        result_list.extend(r)

    return result_list


def _group_into_rounds(messages: list) -> list[list]:
    """将消息列表按 User → AIResponse 分组。

    每遇到一个 HumanMessage 开始新的一轮。
    一轮 = [HumanMessage, AIMessage, AIMessage, ...]
    """
    from langchain_core.messages import HumanMessage

    rounds: list[list] = []
    current_round: list = []

    for msg in messages:
        if isinstance(msg, HumanMessage) and current_round:
            rounds.append(current_round)
            current_round = []
        current_round.append(msg)

    if current_round:
        rounds.append(current_round)

    return rounds
