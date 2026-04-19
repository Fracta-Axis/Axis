"""Tests del módulo axis.totp — TOTP fractal con ventana anti-replay."""

import pytest
from axis.totp import generate, verify
from axis.totp.fractal_otp import _code_for_slot


class TestGenerate:
    def test_returns_tuple(self):
        result = generate("secret")
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_code_format(self):
        code, expires, prev, nxt = generate("secret")
        assert len(code) == 6
        assert code.isdigit()
        assert len(prev) == 6
        assert len(nxt) == 6

    def test_expires_range(self):
        _, expires, _, _ = generate("secret")
        assert 1 <= expires <= 30

    def test_deterministic_same_slot(self):
        """El mismo slot debe producir el mismo código."""
        t = 1_700_000_000.0  # timestamp fijo
        code1, _, _, _ = generate("secret", _now=t)
        code2, _, _, _ = generate("secret", _now=t)
        assert code1 == code2

    def test_different_secrets_different_codes(self):
        t = 1_700_000_000.0
        code1, _, _, _ = generate("secret1", _now=t)
        code2, _, _, _ = generate("secret2", _now=t)
        assert code1 != code2

    def test_window_adjacency(self):
        """El código anterior del slot t debe coincidir con el código de t-1."""
        t = 1_700_000_030.0  # inicio exacto de un nuevo slot
        code_curr, _, code_prev, _ = generate("secret", _now=t)
        slot = int(t // 30)
        expected_prev = _code_for_slot("secret", slot - 1)
        assert code_prev == expected_prev

    def test_next_window(self):
        t = 1_700_000_000.0
        _, _, _, code_next = generate("secret", _now=t)
        slot = int(t // 30)
        expected_next = _code_for_slot("secret", slot + 1)
        assert code_next == expected_next


class TestVerify:
    def test_current_code_valid(self):
        t = 1_700_000_000.0
        code, _, _, _ = generate("secret", _now=t)
        assert verify("secret", code, _now=t)

    def test_prev_code_valid(self):
        t = 1_700_000_000.0
        _, _, prev, _ = generate("secret", _now=t)
        assert verify("secret", prev, _now=t)

    def test_next_code_valid(self):
        t = 1_700_000_000.0
        _, _, _, nxt = generate("secret", _now=t)
        assert verify("secret", nxt, _now=t)

    def test_wrong_code_invalid(self):
        assert not verify("secret", "000000")
        assert not verify("secret", "999999")

    def test_wrong_secret_invalid(self):
        t = 1_700_000_000.0
        code, _, _, _ = generate("secret", _now=t)
        assert not verify("other_secret", code, _now=t)

    def test_wrong_format_invalid(self):
        assert not verify("secret", "abc123")
        assert not verify("secret", "12345")   # 5 dígitos
        assert not verify("secret", "1234567")  # 7 dígitos
