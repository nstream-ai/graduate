"""Slices: the unit a certificate is issued for (node x call type x tool x position).

Position matters: on real traffic, error rates fall with position in the session,
so early and deep calls are sliced apart. Buckets are coarse to keep slices large.
"""

from __future__ import annotations

from dataclasses import dataclass

from graduate.core.canonical import Action, action_of
from graduate.core.schema import Call, Session

POSITION_BUCKETS = ((0, 0, "1"), (1, 2, "2-3"), (3, 9, "4-10"), (10, 29, "11-30"))


def position_bucket(index: int) -> str:
    for lo, hi, name in POSITION_BUCKETS:
        if lo <= index <= hi:
            return name
    return "31+"


@dataclass(frozen=True, order=True)
class Slice:
    node: str
    call_type: str  # "tool" | "parallel_tool" | "text"
    tool: str  # tool name(s) for tool calls, "-" for text
    position: str

    def __str__(self) -> str:
        return f"{self.node}/{self.call_type}/{self.tool}/{self.position}"


def action_type_and_tool(action: Action) -> tuple[str, str]:
    if action.kind == "text":
        return "text", "-"
    names = sorted(set(action.tool_names))
    return ("parallel_tool" if len(action.tools) > 1 else "tool"), "+".join(names)


def call_type_and_tool(session: Session, call: Call) -> tuple[str, str]:
    return action_type_and_tool(action_of(session.call_tool_calls(call)))


def routing_slice(node: str, proposed: Action, index: int, *, by_tool: bool = True) -> Slice:
    """Slice used for routing and certification: keyed by the *candidate's proposed* action.

    A router cannot see the teacher's answer before deciding, so the certified population must
    be defined by what is observable: the node, the position and what the candidate proposes
    ("the student wants to call get_order_status at step 2"). Slicing by the teacher's action
    would select on an unobservable and mis-state what the router does.
    """
    call_type, tool = action_type_and_tool(proposed)
    return Slice(node, call_type, tool if by_tool else "*", position_bucket(index))


def slice_of(
    session: Session, call: Call, *, by_tool: bool = True, by_position: bool = True
) -> Slice:
    """Slice of a logged call by the action it contains (for describing traffic and training
    data). For certification use `routing_slice` on the candidate's proposed action."""
    call_type, tool = call_type_and_tool(session, call)
    return Slice(
        node=session.agent_node or "agent",
        call_type=call_type,
        tool=tool if by_tool else "*",
        position=position_bucket(call.index) if by_position else "*",
    )
