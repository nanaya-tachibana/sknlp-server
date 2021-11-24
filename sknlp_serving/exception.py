from __future__ import annotations


class Error(Exception):
    def __init__(self, code: int, message: str, *args) -> None:
        super().__init__(*args)
        self.code = code
        self.message = message
