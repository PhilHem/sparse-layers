"""Butterfly factorization utility functions."""

from __future__ import annotations


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1) == 0)


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be a positive integer")
    if _is_power_of_two(value):
        return value
    return 1 << (value - 1).bit_length()
