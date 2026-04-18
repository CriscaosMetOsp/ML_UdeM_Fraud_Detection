"""Unit tests for API input validation."""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.api.main import TransactionInput


def valid_payload():
    return dict(
        amt=107.23, lat=48.8878, long=-118.2105,
        city_pop=149, merch_lat=49.159, merch_long=-118.186,
        hour=0, day_of_week=1, age=45, category=3, state=7
    )


def test_valid_transaction():
    tx = TransactionInput(**valid_payload())
    assert tx.amt == 107.23


def test_negative_amount_rejected():
    with pytest.raises(ValidationError):
        TransactionInput(**{**valid_payload(), "amt": -10.0})


def test_zero_amount_rejected():
    with pytest.raises(ValidationError):
        TransactionInput(**{**valid_payload(), "amt": 0})


def test_hour_out_of_range():
    with pytest.raises(ValidationError):
        TransactionInput(**{**valid_payload(), "hour": 25})


def test_day_of_week_out_of_range():
    with pytest.raises(ValidationError):
        TransactionInput(**{**valid_payload(), "day_of_week": 7})


def test_negative_city_pop_rejected():
    with pytest.raises(ValidationError):
        TransactionInput(**{**valid_payload(), "city_pop": -1})
