"""
KDF Fractal Memory-Hard — axis.kdf.derive

Implementa la derivación de clave en 3 fases:

    FASE 1 — Relleno del scratchpad (secuencial, no paralelizable):
        Evolucionar ψ durante KDF_M pasos con N=KDF_N puntos.
        Guardar cada estado → scratchpad[0..KDF_M-1].
        RAM requerida: KDF_N × KDF_M × 16 B ≈ 8 MB por intento.

    FASE 2 — Mezcla no-lineal (acceso impredecible al scratchpad):
        El índice de acceso depende del estado actual del campo.
        Sin toda la RAM en scratchpad es imposible reproducir la mezcla.
        Análogo a scrypt pero con la SPDE fractal como función de mezcla.

    FASE 3 — Condensación SHA3:
        Estado final → SHA3-512 → HKDF-Expand → clave de longitud arbitraria.
        SHA3 garantiza uniformidad formal; el campo garantiza unicidad fractal.

Velocidad: ~0.5 s/intento en CPU. GPU RTX 4090: 24 GB / 8 MB ≈ 3072 hilos.
"""

from __future__ import annotations

import hashlib

import numpy as np

from axis.core import KDF_N, KDF_M, step_mfsu


def derive(password: str, salt: bytes, key_len: int = 96) -> bytes:
    """
    Deriva una clave criptográfica a partir de (contraseña, salt).

    Args:
        password: Contraseña en texto plano (se codifica en UTF-8).
        salt:     Salt aleatorio de 16 bytes (debe ser único por cifrado).
        key_len:  Longitud de la clave resultante en bytes. Por defecto 96:
                  64 B para cifrado + 32 B de base para el MAC.

    Returns:
        Clave derivada de ``key_len`` bytes.

    Raises:
        ValueError: Si key_len < 1 o salt está vacío.
    """
    if key_len < 1:
        raise ValueError("key_len debe ser >= 1")
    if not salt:
        raise ValueError("salt no puede estar vacío")

    # Condicionamiento inicial — combina contraseña y salt con SHA3-512
    h = hashlib.sha3_512(password.encode("utf-8") + b"\x00" + salt).digest()

    rng = np.random.default_rng(np.frombuffer(h[:32], dtype=np.uint32))
    psi: np.ndarray = rng.standard_normal(KDF_N) + 1j * rng.standard_normal(KDF_N)

    # ── Fase 1: llenar scratchpad ─────────────────────────────────────────
    scratchpad = np.zeros((KDF_M, KDF_N), dtype=np.complex128)
    for step in range(KDF_M):
        psi = step_mfsu(psi, h, step, dt=0.001)
        scratchpad[step] = psi

    # ── Fase 2: mezcla no-lineal (scrypt-fractal) ─────────────────────────
    psi_mix = scratchpad[-1].copy()
    for step in range(KDF_M):
        idx = int(abs(np.real(psi_mix[0])) * 1e9) % KDF_M
        psi_mix = psi_mix + 0.001 * scratchpad[idx]
        psi_mix = psi_mix / max(float(np.max(np.abs(psi_mix))), 1.0)

    # ── Fase 3: condensación ──────────────────────────────────────────────
    state_bytes = (
        (np.real(psi_mix) * 1e10).astype(np.int64).tobytes()
        + (np.imag(psi_mix) * 1e10).astype(np.int64).tobytes()
    )
    k_raw = hashlib.sha3_512(state_bytes + h).digest()  # 64 bytes base

    # HKDF-Expand simplificado para obtener key_len bytes
    if key_len <= 64:
        return k_raw[:key_len]

    result = bytearray()
    prev = b""
    counter = 1
    while len(result) < key_len:
        prev = hashlib.sha3_256(prev + k_raw + counter.to_bytes(1, "big")).digest()
        result.extend(prev)
        counter += 1
    return bytes(result[:key_len])
