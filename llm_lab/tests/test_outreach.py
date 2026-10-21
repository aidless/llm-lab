"""Regression tests for the outreach follow-up schedule."""

from datetime import datetime, timezone

import pytest

from llm_lab.outreach import (
    OutreachRecord,
    OutreachState,
    add_business_days,
)

UTC = timezone.utc


def at(day: int, hour: int = 9) -> datetime:
    return datetime(2026, 8, day, hour, tzinfo=UTC)


def test_add_business_days_skips_weekend():
    assert add_business_days(at(3), 5) == at(10)
    assert add_business_days(at(7), 1) == at(10)


def test_first_contact_waits_exactly_five_business_days():
    record = OutreachRecord().send_first(at(3))

    assert record.state is OutreachState.WAITING
    assert record.follow_up_due_at == at(10)
    assert record.refresh(at(10, 8)).state is OutreachState.WAITING
    assert record.refresh(at(10)).state is OutreachState.FOLLOW_UP_DUE


def test_follow_up_is_only_allowed_once_when_due():
    record = OutreachRecord().send_first(at(3))
    with pytest.raises(ValueError, match="not due"):
        record.send_follow_up(at(7))

    followed_up = record.send_follow_up(at(10))
    assert followed_up.state is OutreachState.FOLLOWED_UP
    assert followed_up.follow_up_sent_at == at(10)

    with pytest.raises(ValueError, match="already been sent"):
        followed_up.send_follow_up(at(11))


@pytest.mark.parametrize("accepted, expected", [(True, OutreachState.REPLIED_YES), (False, OutreachState.REPLIED_NO)])
def test_reply_cancels_follow_up_schedule(accepted, expected):
    replied = OutreachRecord().send_first(at(3)).record_reply(accepted=accepted, now=at(4))

    assert replied.state is expected
    assert replied.refresh(at(20)) is replied
    with pytest.raises(ValueError, match="terminal"):
        replied.send_follow_up(at(20))


def test_reply_after_follow_up_preserves_follow_up_history():
    followed_up = OutreachRecord().send_first(at(3)).send_follow_up(at(10))

    replied = followed_up.record_reply(accepted=True, now=at(12))

    assert replied.state is OutreachState.REPLIED_YES
    assert replied.follow_up_sent_at == at(10)
    assert replied.replied_at == at(12)


def test_invalid_timestamp_combinations_are_rejected():
    with pytest.raises(ValueError, match="non-draft"):
        OutreachRecord(OutreachState.WAITING)
    with pytest.raises(ValueError, match="followed_up"):
        OutreachRecord(OutreachState.FOLLOWED_UP, first_sent_at=at(3))
    with pytest.raises(ValueError, match="waiting records"):
        OutreachRecord(OutreachState.WAITING, first_sent_at=at(3), replied_at=at(4))
