"""
TOTP Fractal con ventana deslizante anti-replay — axis.totp.fractal_otp

Genera códigos TOTP de 6 dígitos basados en la evolución del campo MFSU.

Ventana deslizante:
    Se generan tres códigos: (t-1, t, t+1).
    Un verificador puede aceptar cualquiera de los tres para tolerar
    desfase de reloj, pero debe marcar cada código como "usado"
    para evitar replay dentro de la misma ventana.

Seguridad:
    iv_t = SHA3-512(secret ‖ t_slot.to_bytes(8) ‖ DOMAIN)
    ψ₀ derivado de iv_t, evolución de 32 pasos
    código = |Σ|ψ|·1e9| % 10^6
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from axis.core import TOTP_WINDOW, TOTP_STEPS, TOTP_DOMAIN, step_mfsu


def _code_for_slot(secret: str, t_slot: int) -> str:
    """Genera el código TOTP para un slot temporal dado."""
    h = hashlib.sha3_512(
        secret.encode("utf-8") + t_slot.to_bytes(8, "big") + TOTP_DOMAIN
    ).digest()
    rng = np.random.default_rng(np.frombuffer(h[:32], dtype=np.uint32))
    psi: np.ndarray = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    for step in range(TOTP_STEPS):
        psi = step_mfsu(psi, h, step, dt=0.01)
    raw = (np.abs(psi) * 1e9).astype(np.int64)
    code = abs(int(raw.sum())) % 1_000_000
    return f"{code:06d}"


def generate(
    secret: str,
    window: int = TOTP_WINDOW,
    *,
    _now: float | None = None,
) -> tuple[str, int, str, str]:
    """
    Genera el código TOTP actual y los códigos de las ventanas adyacentes.

    Args:
        secret: Secreto compartido entre cliente y servidor.
        window: Duración de la ventana en segundos (default 30).
        _now:   Timestamp para inyección en tests (no usar en producción).

    Returns:
        Tupla (código_actual, segundos_hasta_expiración, código_anterior, código_siguiente).
        - código_actual:         El código válido en este momento.
        - segundos_hasta_expiración: Tiempo restante de la ventana actual.
        - código_anterior:       Código de la ventana t-1 (para tolerancia).
        - código_siguiente:      Código de la ventana t+1 (para tolerancia).
    """
    now = _now if _now is not None else time.time()
    t_slot = int(now // window)

    code_prev = _code_for_slot(secret, t_slot - 1)
    code_curr = _code_for_slot(secret, t_slot)
    code_next = _code_for_slot(secret, t_slot + 1)

    expires_in = window - (int(now) % window)
    return code_curr, expires_in, code_prev, code_next


def verify(
    secret: str,
    code: str,
    window: int = TOTP_WINDOW,
    *,
    _now: float | None = None,
) -> bool:
    """
    Verifica si ``code`` es válido (ventana actual ±1).

    En producción, el verificador debe mantener un registro de códigos
    usados para evitar replay dentro de la misma ventana.

    Args:
        secret: Secreto compartido.
        code:   Código de 6 dígitos a verificar.
        window: Duración de la ventana en segundos.
        _now:   Timestamp para inyección en tests.

    Returns:
        True si el código es válido en alguna de las tres ventanas.
    """
    _, _, code_prev, code_next = generate(secret, window, _now=_now)
    curr, *_ = generate(secret, window, _now=_now)
    valid = {curr, code_prev, code_next}
    # Comparación tiempo-constante entre cada candidato
    return any(hmac_compare(code, v) for v in valid)


def hmac_compare(a: str, b: str) -> bool:
    """Comparación de strings en tiempo constante (usa hmac.compare_digest)."""
    import hmac
    return hmac.compare_digest(a.encode(), b.encode())
