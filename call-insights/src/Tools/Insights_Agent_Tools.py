from typing import Optional, Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command

@tool
def update_state_Insights_Agent(
    processing_status: Optional[str] = None,
    processing_complete: Optional[bool] = None,
    topic: Optional[str] = None,
    key_points: Optional[str] = None,
    action_items: Optional[str] = None,
    tool_call_id: Annotated[Optional[str], InjectedToolCallId] = None
) -> str:
    """
    Update the state of the agent
    Args:
        processing_status: Status of the processing
        processing_complete: Whether processing is complete
        topic: Topic of the conversation
        key_points: Key points of the conversation
        action_items: Action items of the conversation
    Returns:
        Confirmation message that the state has been updated.
    """
    update_dict = {}
    if processing_status is not None:
        update_dict['processing_status'] = processing_status
    if processing_complete is not None:
        update_dict['processing_complete'] = processing_complete
    if topic is not None:
        update_dict['topic'] = topic
    if key_points is not None:
        update_dict['key_points'] = key_points
    if action_items is not None:
        update_dict['action_items'] = action_items

    update_dict["messages"] = [
        ToolMessage(
            content="Insights state updated successfully.",
            tool_call_id=tool_call_id or "insights_state_updated",
        )
    ]
    # Return a Command to update the state
    return Command(update=update_dict)

