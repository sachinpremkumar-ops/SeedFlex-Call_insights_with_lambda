# from typing import Optional, Annotated
# from langchain_core.tools import tool, InjectedToolCallId
# from langchain_core.messages import ToolMessage
# from langgraph.types import Command


<<<<<<< HEAD
@tool
def update_state_Topic_Classification_Agent(
    processing_status: Optional[str] = None,
    processing_complete: Optional[bool] = None,
    topic: Optional[str] = None,
    tool_call_id: Annotated[Optional[str], InjectedToolCallId] = None
) -> str:
    """
    Update the state of the agent
    Args:
        processing_status: Status of the processing
        processing_complete: Whether processing is complete
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
=======
# @tool
# def classify_conversation(conversation: str, tool_call_id: Annotated[Optional[str], InjectedToolCallId] = None):
#     """
#     Classify the conversation into a topic.
#     Args:
#         conversation: The conversation text to classify
#     Returns:
#         The classified topic
#     """
#     # This is a placeholder - the actual classification should be done by the LLM
#     # The tool exists to prevent "tool not found" errors
#     update_dict = {
#         "messages": [
#             ToolMessage(
#                 content="Conversation classification completed. Use update_state_Topic_Classification_Agent to update the state.",
#                 tool_call_id=tool_call_id or "classify_conversation",
#             )
#         ]
#     }
#     return Command(update=update_dict)


# @tool
# def update_state_Topic_Classification_Agent(
#     processing_status: Optional[str] = None,
#     processing_complete: Optional[bool] = None,
#     topic: Optional[str] = None,
#     tool_call_id: Annotated[Optional[str], InjectedToolCallId] = None
# ) -> str:
#     """
#     Update the state of the agent
#     Args:
#         processing_status: Status of the processing
#         processing_complete: Whether processing is complete
#     Returns:
#         Confirmation message that the state has been updated.
#     """
#     update_dict = {}
#     if processing_status is not None:
#         update_dict['processing_status'] = processing_status
#     if processing_complete is not None:
#         update_dict['processing_complete'] = processing_complete
#     if topic is not None:
#         update_dict['topic'] = topic
>>>>>>> bff7368 (Lambda integration, with filtered messages)

#     update_dict["messages"] = [
#         ToolMessage(
#             content="State updated successfully.",
#             tool_call_id=tool_call_id or "topic_classification_state_updated",
#         )
#     ]
#     # Return a Command to update the state
#     return Command(update=update_dict)