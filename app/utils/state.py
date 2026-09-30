from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

from utils.models import PartialTripRequest, TripRequest


class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    partial_request: PartialTripRequest | None   # accumulates across turns
    missing_fields: list[str]
    problems: list[str]                          # validation issues to show the user
    trip_request: TripRequest | None             # set only when complete and valid