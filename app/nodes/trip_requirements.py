from datetime import date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END
from langgraph.types import Command
from pydantic import ValidationError

from utils.models import PartialTripRequest, TripRequest
from utils.state import TravelState

load_dotenv()  

# ---------- config ----------
TIMEZONE = ZoneInfo("Europe/London")   # later: take this from the user's profile
MAX_CONTEXT_MESSAGES = 4               # extractor only needs the last question + answer

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    reasoning_effort="low",  # remove this line if your langchain-groq version rejects it
)
extractor = llm.with_structured_output(PartialTripRequest)

SYSTEM_PROMPT = """You extract trip-planning details from a conversation.
Today's date is {today}.
Already known from earlier turns: {known}

Rules:
- Fill a field only if the user clearly stated it. Never guess or assume defaults; leave it null.
- Only extract details from the user's messages, never from the assistant's messages.
- Convert relative dates ("next Friday") to ISO dates using today's date.
- If the user says they have no dates yet or are flexible, set dates_undecided to true.
- If the user corrects earlier information, return the corrected value.
- Short replies (like "5" or "two") answer the assistant's last question; read them in that context.
- If the user starts planning a different trip, extract only what they state about the new trip.
"""

REQUIRED = ("origin", "destination", "duration_days", "travelers")

QUESTIONS = {
    "origin": "Where will you be travelling from?",
    "destination": "Where would you like to go?",
    "duration_days": "How many days are you planning for?",
    "travelers": "How many travelers will there be?",
    "departure_date": "What's your departure date? (Or tell me if you're still flexible.)",
}

def today_local() -> date:
    return datetime.now(TIMEZONE).date()

def merge_request(known: PartialTripRequest, extracted: PartialTripRequest) -> PartialTripRequest:
    """Apply newly extracted values over known ones, keeping the date fields consistent."""
    changes = extracted.model_dump(exclude_defaults=True)
    merged = known.model_copy(update=changes)

    trip_shape_changed = "departure_date" in changes or "duration_days" in changes
    if trip_shape_changed and "return_date" not in changes and merged.duration_days is not None:
        # departure or length changed: the old return date is stale, recompute it
        merged = merged.model_copy(update={"return_date": None})
    elif "return_date" in changes and "duration_days" not in changes and merged.departure_date:
        # return date changed: the old length is stale, recompute it
        merged = merged.model_copy(update={"duration_days": None})
    return merged

def fill_derived(p: PartialTripRequest) -> PartialTripRequest:
    """Derive what's implied by the other fields instead of asking the user."""
    dep, ret, n = p.departure_date, p.return_date, p.duration_days
    if dep and ret and n is None:
        n = (ret - dep).days
    if dep and n and ret is None:
        ret = dep + timedelta(days=n)
    return p.model_copy(update={"duration_days": n, "return_date": ret})

def find_missing(p: PartialTripRequest) -> list[str]:
    missing = [f for f in REQUIRED if getattr(p, f) is None]
    if p.departure_date is None and not p.dates_undecided:
        missing.append("departure_date")
    return missing

def format_errors(exc: ValidationError) -> list[str]:
    out = []
    for e in exc.errors():
        msg = e["msg"].removeprefix("Value error, ")
        out.append(f"{e['loc'][0]}: {msg}" if e["loc"] else msg)
    return out

# Node
def extract_trip(
    state: TravelState,
) -> Command[Literal["ask_follow_up", "finalize", "__end__"]]:
    today = today_local()
    known = state.get("partial_request") or PartialTripRequest()

    system = SystemMessage(SYSTEM_PROMPT.format(
        today=today.isoformat(),
        known=known.model_dump_json(exclude_defaults=True),
    ))
    recent = state["messages"][-MAX_CONTEXT_MESSAGES:]

    try:
        extracted = extractor.invoke([system, *recent])
    except OutputParserException:
        extracted = None

    if extracted is None:
        return Command(
            update={"messages": [AIMessage(
                "Sorry, I didn't catch that. Could you rephrase your trip details?"
            )]},
            goto=END,
        )

    merged = fill_derived(merge_request(known, extracted))

    problems: list[str] = []
    if merged.departure_date and merged.departure_date < today:
        problems.append(f"{merged.departure_date:%d %b %Y} is in the past.")
        merged = merged.model_copy(update={"departure_date": None, "return_date": None})

    missing = find_missing(merged)

    trip = None
    if not missing:
        try:
            trip = TripRequest(**merged.model_dump(exclude={"dates_undecided"}))
        except ValidationError as exc:
            problems += format_errors(exc)

    complete = trip is not None and not problems
    return Command(
        update={
            "partial_request": merged,
            "missing_fields": missing,
            "problems": problems,
            "trip_request": trip if complete else None,
        },
        goto="finalize" if complete else "ask_follow_up",
    )


def ask_follow_up(state: TravelState) -> dict:
    p = state["partial_request"]
    problems = state.get("problems", [])
    missing = state.get("missing_fields", [])

    parts = []
    if problems:
        parts.append("A couple of things don't add up:\n" +
                     "\n".join(f"- {x}" for x in problems))
    if missing:
        intro = (f"Great, {p.destination} it is! " if p.destination
                 else "Happy to help plan your trip! ")
        parts.append(intro + "I just need a few more details:\n" +
                     "\n".join(f"- {QUESTIONS[f]}" for f in missing))
    elif problems:
        parts.append("Could you confirm the correct details?")
    return {"messages": [AIMessage("\n\n".join(parts))]}


def finalize(state: TravelState) -> dict:
    trip = state["trip_request"]
    text = f"Got it! Here's your trip request:\n```json\n{trip.model_dump_json(indent=2)}\n```"
    return {
        "messages": [AIMessage(text)],
        "partial_request": None,   # the next message starts a fresh request
        "missing_fields": [],
        "problems": [],
    }