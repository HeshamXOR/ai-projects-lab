"""Dialog state machine: turn-taking and slot-driven follow-ups.

An explicit finite state machine drives the conversation. States:

- ``IDLE``        — no active conversation.
- ``LISTENING``   — ready to accept a new user turn.
- ``AWAITING_SLOT`` — a required slot is missing; the system asked a follow-up.
- ``CONFIRMING``  — all slots filled; system asks the user to confirm.
- ``RESPONDING``  — system produces the final response and acts.
- ``DONE``        — conversation closed (e.g. after goodbye).

Transitions are driven by ``(intent, slots)`` for each incoming turn. The
``DialogManager`` tracks per-session context (current intent + accumulated
slots) so a follow-up turn supplying only the missing value completes the frame.

Implemented as a method-dispatch FSM with an explicit transition table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .slots import DEFAULT_SCHEMAS, IntentSchema, fill_slots


class State(str, Enum):
    """Dialog states."""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    AWAITING_SLOT = "AWAITING_SLOT"
    CONFIRMING = "CONFIRMING"
    RESPONDING = "RESPONDING"
    DONE = "DONE"


# Human-readable follow-up prompts per slot.
_SLOT_PROMPTS: Dict[str, str] = {
    "destination": "Where would you like to fly to?",
    "origin": "Where are you departing from?",
    "date": "What date should I use?",
    "time": "What time would you like?",
    "city": "Which city?",
    "number": "How many?",
}

# Response templates per intent, formatted with the filled slots.
_RESPONSE_TEMPLATES: Dict[str, str] = {
    "greet": "Hello! How can I help you today?",
    "goodbye": "Goodbye! Have a great day.",
    "book_flight": "Booking a flight to {destination} on {date}.",
    "check_weather": "Here is the weather for {city}.",
    "set_alarm": "Alarm set for {time}.",
}

# Intents that confirm before acting (others respond directly).
_NEEDS_CONFIRMATION = {"book_flight"}


@dataclass
class DialogResult:
    """One turn's outcome: the new state, a system utterance, and slot status."""

    state: State
    response: str
    intent: Optional[str]
    slots: Dict[str, object] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)


@dataclass
class _Context:
    """Mutable per-session conversation context."""

    intent: Optional[str] = None
    slots: Dict[str, object] = field(default_factory=dict)
    pending_confirmation: bool = False


class DialogManager:
    """Slot-driven dialog FSM.

    Call :meth:`handle` once per user turn with the classified ``intent`` and the
    raw ``text`` (used to extract slots). The manager updates its state and
    returns a :class:`DialogResult` describing what to say back.
    """

    # Explicit transition table documenting which states a turn may move into.
    TRANSITIONS: Dict[State, List[State]] = {
        State.IDLE: [State.LISTENING],
        State.LISTENING: [State.AWAITING_SLOT, State.CONFIRMING, State.RESPONDING, State.DONE],
        State.AWAITING_SLOT: [State.AWAITING_SLOT, State.CONFIRMING, State.RESPONDING],
        State.CONFIRMING: [State.RESPONDING, State.LISTENING, State.AWAITING_SLOT],
        State.RESPONDING: [State.LISTENING, State.DONE],
        State.DONE: [State.LISTENING],
    }

    def __init__(self, schemas: Optional[Dict[str, IntentSchema]] = None) -> None:
        self.schemas = schemas or DEFAULT_SCHEMAS
        self.state: State = State.IDLE
        self.ctx = _Context()

    # -- transition helpers -------------------------------------------------

    def _can_transition(self, target: State) -> bool:
        """Whether moving from the current state to ``target`` is allowed."""
        return target in self.TRANSITIONS.get(self.state, [])

    def _goto(self, target: State) -> None:
        """Move to ``target`` if the transition table permits it."""
        if not self._can_transition(target):
            raise ValueError(f"Illegal transition {self.state} -> {target}")
        self.state = target

    def reset(self) -> None:
        """Clear context and return to IDLE."""
        self.state = State.IDLE
        self.ctx = _Context()

    # -- main entry point ---------------------------------------------------

    def handle(self, intent: str, text: str) -> DialogResult:
        """Process one user turn and return the system's response."""
        if self.state in (State.IDLE, State.DONE):
            self._goto(State.LISTENING)

        # If we were confirming, interpret yes/no.
        if self.ctx.pending_confirmation:
            return self._handle_confirmation(text)

        # Continue an in-progress frame (AWAITING_SLOT) with the same intent.
        if self.state == State.AWAITING_SLOT and self.ctx.intent:
            active_intent = self.ctx.intent
        else:
            active_intent = intent
            self.ctx.intent = intent
            self.ctx.slots = {}

        if active_intent == "goodbye":
            self.reset()
            self.state = State.DONE
            return DialogResult(State.DONE, _RESPONSE_TEMPLATES["goodbye"], "goodbye")

        result = fill_slots(
            active_intent, text, schemas=self.schemas, known=self.ctx.slots
        )
        self.ctx.slots = result.slots

        if result.missing:
            self._goto_from_listening(State.AWAITING_SLOT)
            slot = result.missing[0]
            prompt = _SLOT_PROMPTS.get(slot, f"Please provide the {slot}.")
            return DialogResult(
                State.AWAITING_SLOT, prompt, active_intent,
                slots=dict(result.slots), missing=list(result.missing),
            )

        # All required slots filled.
        if active_intent in _NEEDS_CONFIRMATION:
            self._goto_from_listening(State.CONFIRMING)
            self.ctx.pending_confirmation = True
            summary = _RESPONSE_TEMPLATES[active_intent].format(**result.slots)
            return DialogResult(
                State.CONFIRMING, f"{summary} Shall I confirm? (yes/no)",
                active_intent, slots=dict(result.slots),
            )

        return self._respond(active_intent, result.slots)

    # -- internal state moves ----------------------------------------------

    def _goto_from_listening(self, target: State) -> None:
        """Bridge LISTENING/AWAITING_SLOT into ``target`` honoring the table."""
        if self.state == State.LISTENING:
            self._goto(target)
        elif self.state == State.AWAITING_SLOT and self._can_transition(target):
            self._goto(target)
        elif self.state != target:
            # already where we need transitions to flow from
            self.state = target

    def _handle_confirmation(self, text: str) -> DialogResult:
        """Resolve a yes/no answer while in CONFIRMING."""
        intent = self.ctx.intent or ""
        slots = dict(self.ctx.slots)
        affirmative = bool(
            re.search(r"\b(yes|yeah|yep|sure|confirm|ok|okay)\b", text.lower())
        )
        self.ctx.pending_confirmation = False
        if affirmative:
            return self._respond(intent, slots)
        # negative: drop the frame, go back to listening
        self.state = State.CONFIRMING
        self._goto(State.LISTENING)
        self.ctx = _Context()
        return DialogResult(State.LISTENING, "Okay, cancelled. What else?", intent)

    def _respond(self, intent: str, slots: Dict[str, object]) -> DialogResult:
        """Produce the final response and move RESPONDING -> LISTENING."""
        if self.state != State.RESPONDING:
            # allowed from LISTENING, AWAITING_SLOT, CONFIRMING
            if self._can_transition(State.RESPONDING):
                self._goto(State.RESPONDING)
            else:
                self.state = State.RESPONDING
        template = _RESPONSE_TEMPLATES.get(intent, "Done.")
        try:
            response = template.format(**slots)
        except KeyError:
            response = template
        result = DialogResult(
            State.RESPONDING, response, intent, slots=dict(slots)
        )
        # ready for the next turn
        self._goto(State.LISTENING)
        self.ctx = _Context()
        result.state = State.RESPONDING
        return result
