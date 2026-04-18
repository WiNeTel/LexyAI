"""
Lexy AI - RepetitionDetector.

Sliding-window pattern detector to abort runaway LLM streams. Ported from
Lexy v1 with stronger type hints and structured logging.
"""

from __future__ import annotations

from lexy_core.utils.logging import get_logger

log = get_logger(module="repetition")


class RepetitionDetector:
    """
    Detect repeating text patterns in a streaming LLM response.

    Usage::

        detector = RepetitionDetector()
        async for chunk in stream:
            if detector.check(chunk):
                break
            yield chunk
    """

    def __init__(
        self,
        window_size: int = 200,
        min_pattern_len: int = 20,
        max_repeats: int = 3,
    ) -> None:
        self._buffer: str = ""
        self._window_size = window_size
        self._min_pattern_len = min_pattern_len
        self._max_repeats = max_repeats

    def reset(self) -> None:
        """Clear the buffer for the next stream."""
        self._buffer = ""

    def check(self, token: str) -> bool:
        """
        Append a token; return True if a repeating pattern is detected.

        Pattern detection slides a window over the tail of the buffer and
        checks whether any substring of length ``min_pattern_len`` (or longer)
        repeats at least ``max_repeats`` times in a row.
        """
        self._buffer += token

        if len(self._buffer) < self._min_pattern_len * 2:
            return False

        # Only inspect the tail to keep this O(window_size).
        text = self._buffer[-self._window_size * 2 :]
        max_pattern_len = len(text) // self._max_repeats

        for pattern_len in range(self._min_pattern_len, max_pattern_len + 1):
            pattern = text[-pattern_len:]
            count = 1
            pos = len(text) - pattern_len
            while pos >= pattern_len:
                segment = text[pos - pattern_len : pos]
                if segment != pattern:
                    break
                count += 1
                pos -= pattern_len
            if count >= self._max_repeats:
                log.warning(
                    "repetition.detected",
                    pattern_len=pattern_len,
                    repeats=count,
                    buffer_len=len(self._buffer),
                )
                return True

        return False
