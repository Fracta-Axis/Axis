"""
Constantes MFSU — inmutables.

Ecuación central:
    ∂ψ/∂t = −δF·(−Δ)^(β/2)ψ + γ|ψ|²ψ + σ·η(x,t)

donde:
    δF = 0.921   (dimensión fractal del campo)
    β  = 1.079   (exponente del laplaciano fraccional)
    H  = 0.541   (exponente de Hurst del ruido)
    df = 2.921   (dimensión fractal proyectada)
    γ  = δF      (no-linealidad ligada al parámetro fractal)
    σ  = 0.1     (intensidad del ruido)
"""

# ── Parámetros físicos del campo MFSU ────────────────────────────────────────
DELTA_F: float = 0.921
BETA: float = 2.0 - DELTA_F          # 1.079
HURST: float = 0.541
DF_PROJ: float = 2.0 + DELTA_F       # 2.921
GAMMA_NL: float = DELTA_F            # γ = δF
SIGMA_ETA: float = 0.1

# ── Parámetros KDF memory-hard ────────────────────────────────────────────────
# N=2048 puntos × M=256 pasos × 16 B/punto ≈ 8 MB de scratchpad por intento
KDF_N: int = 2048
KDF_M: int = 256

# ── Parámetros del generador de keystream ────────────────────────────────────
KS_N: int = 512          # puntos del campo para keystream
KS_STEPS_MIN: int = 48   # pasos mínimos (derivados de la clave)
KS_STEPS_MAX: int = 112  # pasos máximos

# ── Formato de archivo .fracta v3 ────────────────────────────────────────────
MAGIC: bytes = b"MFSUv3"
VERSION: bytes = b"\x03"
IV_LEN: int = 16
SALT_LEN: int = 16
MAC_SALT_LEN: int = 16
MAC_LEN: int = 32
BLOCK_SIZE: int = 16
HEADER_LEN: int = len(MAGIC) + 1 + IV_LEN + SALT_LEN + MAC_SALT_LEN + MAC_LEN  # 87 bytes

# ── TOTP fractal ──────────────────────────────────────────────────────────────
TOTP_WINDOW: int = 30    # segundos por ventana
TOTP_STEPS: int = 32     # pasos de evolución por código
TOTP_DOMAIN: bytes = b"MFSU_TOTP_v3"
