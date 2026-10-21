"""Local-only scheduling for one-to-one outreach follow-ups.

This module intentionally has no mail transport integration.  It turns the
project's outreach policy into deterministic state transitions so a private
tracker can decide what is due without sending or collecting messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class OutreachState(str, Enum):
    """States for a single outreach conversation."""

    DRAFT = "draft"
    WAITING = "waiting"
    FOLLOW_UP_DUE = "follow_up_due"
    FOLLOWED_UP = "followed_up"
    REPLIED_YES = "replied_yes"
    REPLIED_NO = "replied_no"


REPLY_STATES = frozenset({OutreachState.REPLIED_YES, OutreachState.REPLIED_NO})


def add_business_days(start: datetime, days: int) -> datetime:
    """Return ``start`` advanced by ``days`` Monday-to-Friday workdays."""
    if days < 0:
        raise ValueError("days must be non-negative")

    current = start
    remaining = days
    while remaining:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


@dataclass(frozen=True)
class OutreachRecord:
    """Immutable outreach tracking record; timestamps must share a timezone."""

    state: OutreachState = OutreachState.DRAFT
    first_sent_at: datetime | None = None
    follow_up_sent_at: datetime | None = None
    replied_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.state is OutreachState.DRAFT:
            if any((self.first_sent_at, self.follow_up_sent_at, self.replied_at)):
                raise ValueError("a draft cannot have outreach timestamps")
            return
        if self.first_sent_at is None:
            raise ValueError("non-draft records require first_sent_at")
        if self.state is OutreachState.FOLLOWED_UP and self.follow_up_sent_at is None:
            raise ValueError("followed_up records require follow_up_sent_at")
        if self.state in REPLY_STATES and self.replied_at is None:
            raise ValueError("reply states require replied_at")
        if self.state in {OutreachState.WAITING, OutreachState.FOLLOW_UP_DUE} and any(
            (self.follow_up_sent_at, self.replied_at)
        ):
            raise ValueError("waiting records cannot have a follow-up or reply timestamp")
        if self.state is OutreachState.FOLLOWED_UP and self.replied_at is not None:
            raise ValueError("followed_up records cannot have a reply timestamp")

    @property
    def follow_up_due_at(self) -> datetime | None:
        """Return the first time one follow-up is permitted, if applicable."""
        if self.first_sent_at is None:
            return None
        return add_business_days(self.first_sent_at, 5)

    def refresh(self, now: datetime) -> OutreachRecord:
        """Expose a due follow-up; never changes a terminal state."""
        if self.state is OutreachState.WAITING and self.follow_up_due_at is not None and now >= self.follow_up_due_at:
            return OutreachRecord(OutreachState.FOLLOW_UP_DUE, first_sent_at=self.first_sent_at)
        return self

    def send_first(self, now: datetime) -> OutreachRecord:
        """Record the first manual contact."""
        if self.state is not OutreachState.DRAFT:
            raise ValueError("first contact can only be sent from draft")
        return OutreachRecord(OutreachState.WAITING, first_sent_at=now)

    def send_follow_up(self, now: datetime) -> OutreachRecord:
        """Record the one allowed manual follow-up after its due date."""
        if self.state is OutreachState.FOLLOWED_UP:
            raise ValueError("the one allowed follow-up has already been sent")
        if self.state in {OutreachState.REPLIED_YES, OutreachState.REPLIED_NO}:
            raise ValueError("a terminal outreach record cannot receive a follow-up")
        due_record = self.refresh(now)
        if due_record.state is not OutreachState.FOLLOW_UP_DUE:
            raise ValueError("follow-up is not due or has already been sent")
        return OutreachRecord(
            OutreachState.FOLLOWED_UP,
            first_sent_at=due_record.first_sent_at,
            follow_up_sent_at=now,
        )

    def record_reply(self, *, accepted: bool, now: datetime) -> OutreachRecord:
        """Record a reply and stop any pending follow-up schedule."""
        if self.state is OutreachState.DRAFT:
            raise ValueError("a draft cannot receive a reply")
        if self.state in REPLY_STATES:
            raise ValueError("a terminal outreach record cannot receive another reply")
        return OutreachRecord(
            OutreachState.REPLIED_YES if accepted else OutreachState.REPLIED_NO,
            first_sent_at=self.first_sent_at,
            follow_up_sent_at=self.follow_up_sent_at,
            replied_at=now,
        )
