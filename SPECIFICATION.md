# FYX File Format Specification

**Format Name:** Fractalyx Encrypted Container  
**File Extension:** `.fyx`  
**MIME Type (proposed):** `application/x-fractalyx`  
**Magic Bytes:** `4D 46 53 55 76 34` (`MFSUv4`)  
**Current Version:** 4 (FractalShield)  
**Legacy Version:** 3 (MFSU Standard)  
**Specification Version:** 1.0.0  
**Date:** 2026-04-21  
**Author:** Fracta-Axis  
**Repository:** https://github.com/Fracta-Axis/FractalyxwebLite  

---

## 1. Overview

The `.fyx` format is a binary encrypted container produced by **Fractalyx** (FractalyxwebLite), a cryptographic system based on the **Unified Fractal-Stochastic Model (MFSU)**. It stores arbitrary plaintext data encrypted under a password-derived key using a multi-layer architecture that combines fractal-stochastic dynamics with established cryptographic primitives.

The format is designed around three security goals:

- **Confidentiality** — Plaintext is indistinguishable from random noise without the correct password.
- **Integrity and Authenticity** — Every byte of ciphertext is authenticated via HMAC-SHA3-256 (Encrypt-then-MAC).
- **Deniability (FractalShield mode)** — The ciphertext contains N independent layers, only one of which holds the real plaintext. An attacker cannot determine which layer is real, or whether decryption succeeded, without exhausting all layers.

Two sub-formats share the `.fyx` extension and are identified by their magic bytes:

| Sub-format | Magic Bytes (ASCII) | Version Byte | Description |
|---|---|---|---|
| FractalShield v4 | `MFSUv4` | `0x04` | Multi-layer deniable encryption |
| MFSU Standard v3 | `MFSUv3` | `0x03` | Single-layer authenticated encryption |

---

## 2. Governing Equation — MFSU

All cryptographic material in `.fyx` files is derived from numerical integration of the following stochastic partial differential equation (SPDE):

```
∂ψ/∂t = −δF·(−Δ)^(β/2)ψ  +  γ|ψ|²ψ  +  σ·η(x,t)
```

**Parameters (fixed, immutable):**

| Symbol | Value | Role |
|---|---|---|
| δF | 0.921 | Fractional dissipation coefficient |
| β | 1.079 | Fractional Laplacian exponent (= 2 − δF) |
| H | 0.541 | Hurst exponent of fractional Gaussian noise η |
| df | 2.921 | Fractal dimension of the projected field (= 2 + δF) |
| γ | 0.921 | Non-linear coupling constant (= δF) |
| σ | 0.1 | Noise amplitude |

The field ψ(x,t) is complex-valued and discretized over N spatial points. Its time evolution is computed via an Euler scheme with time-constant normalization (divides by `max(|ψ|, 1)` unconditionally, eliminating timing side-channels).

**Fractional Laplacian** is computed spectrally via FFT:

```
F[(−Δ)^(α/2) f](k) = |k|^α · F[f](k)
```

**Fractional Gaussian noise** η(x,t) has power spectrum `S(k) ~ |k|^(−(2H+1))`, matching the statistical signature of the cosmic microwave background at H = 0.541.

---

## 3. Cryptographic Primitives

### 3.1 Memory-Hard Key Derivation Function (MFSU-KDF)

The KDF derives key material from a password and salt in three phases:

**Phase 1 — Scratchpad fill (sequential, non-parallelizable):**

- Initialise ψ₀ from `SHA3-512(password ‖ 0x00 ‖ salt)`.
- Evolve ψ for `KDF_M = 256` steps over `KDF_N = 2048` spatial points.
- Store every state: `scratchpad[0..255]` — requires ≈ 8 MB RAM.

**Phase 2 — Non-linear mixing (unpredictable scratchpad access):**

- Access index is derived from the current field state: `idx = ⌊|Re(ψ[0])| × 10⁹⌋ mod KDF_M`.
- Without the full scratchpad in memory the mixing sequence cannot be reproduced.
- This is analogous to scrypt, using the SPDE instead of a hash function.

**Phase 3 — Condensation:**

- Final state → `SHA3-512(state_bytes ‖ h)` → raw key material.
- Expanded to the requested length via HKDF-Expand (SHA3-256, counter mode).

**Performance characteristics:**

| Metric | Value |
|---|---|
| RAM per attempt | ≈ 8 MB |
| Time per attempt | ≈ 0.5 s |
| Throughput | ≈ 2 attempts/s |
| GPU parallelism (RTX 4090, 24 GB) | ≈ 3,000 simultaneous threads |

### 3.2 Keystream Generator (MFSU-KS)

Produces a pseudorandom byte stream of arbitrary length using a two-layer architecture:

**Layer 1 — Fractal field (entropy source):**

- Initialise ψ from `SHA3-512(derived_key ‖ iv)`.
- Number of evolution steps: `n_steps = 48 + (h[0] mod 64)` — key-dependent, between 48 and 112.
- Extract Re(ψ) and Im(ψ) at each step as raw bytes.

**Layer 2 — SHA3-256 whitener (uniformity guarantee):**

- XOR each 32-byte block with `SHA3-256(mixer_key ‖ counter)`.
- Role: mathematical guarantee of byte uniformity (the fractal field alone is not uniform; demonstrated by χ² = 1752 without whitener vs. χ² = 254, p = 0.49 with whitener).

### 3.3 Merkle-Damgård Fractal Hash

An internal hash function used for integrity operations:

- The message is split into 64-byte blocks.
- Each block perturbs the field directly: `ψ = ψ + δF · encode(block)`.
- The field evolves 16 steps per block — changes to any block alter the entire subsequent trajectory.
- Final digest: `SHA3-512(state_bytes)` — 128 hex characters.

### 3.4 Authentication — HMAC-SHA3-256 (Encrypt-then-MAC)

All `.fyx` files use Encrypt-then-MAC with a MAC key derived independently from the KDF output and a dedicated MAC salt, ensuring that the MAC key and the encryption key are cryptographically separated.

---

## 4. Binary Format — FractalShield v4 (`.fyx` primary format)

### 4.1 File Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  HEADER                                                         │
│  ├─ MAGIC       6 bytes   "MFSUv4"                              │
│  ├─ VERSION     1 byte    0x04                                  │
│  ├─ LEVEL       1 byte    Shield level (1, 2, or 3)             │
│  ├─ N_LAYERS    1 byte    Number of layers                      │
│  ├─ SALT_G     16 bytes   Global KDF salt (random)              │
│  ├─ IV_ORD     16 bytes   IV for order encryption (random)      │
│  ├─ ORD_LEN     2 bytes   Length of ORDER_ENC (big-endian)      │
│  └─ ORDER_ENC  variable   Encrypted layer order                 │
├─────────────────────────────────────────────────────────────────┤
│  MAC           32 bytes   HMAC-SHA3-256(header ‖ layer_blob)    │
├─────────────────────────────────────────────────────────────────┤
│  LAYER_BLOB    variable   N layers, in shuffled order           │
│  Each layer:                                                    │
│  ├─ SALT_i     16 bytes   Per-layer KDF salt (random)           │
│  ├─ IV_i       16 bytes   Per-layer IV (random)                 │
│  └─ CTEXT_i    L bytes    Encrypted layer payload               │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Field Definitions

| Field | Offset | Size | Description |
|---|---|---|---|
| MAGIC | 0 | 6 | ASCII `MFSUv4` — identifies format |
| VERSION | 6 | 1 | `0x04` |
| LEVEL | 7 | 1 | FractalShield level: `0x01`, `0x02`, or `0x03` |
| N_LAYERS | 8 | 1 | Number of layers: 3, 4, or 5 (equals `len(SHIELD_LEVELS[level])`) |
| SALT_G | 9 | 16 | Global salt for MAC key derivation; cryptographically random |
| IV_ORD | 25 | 16 | IV for encrypting the layer order; cryptographically random |
| ORD_LEN | 41 | 2 | Big-endian uint16: byte length of ORDER_ENC |
| ORDER_ENC | 43 | ORD_LEN | MFSU-KS encrypted, PKCS7-padded layer order array |
| MAC | 43 + ORD_LEN | 32 | HMAC-SHA3-256 over `header ‖ layer_blob` |
| LAYER_BLOB | 75 + ORD_LEN | variable | Concatenated encrypted layers in shuffled order |

### 4.3 FractalShield Levels

| Level | Layers (N) | KDF_M per layer | User time | Attacker cost |
|---|---|---|---|---|
| 1 — Standard | 3 | [256, 512, 1024] | ≈ 0.5 s | 3.5× |
| 2 — Reinforced | 4 | [256, 512, 1024, 2048] | ≈ 0.7 s | 7.5× |
| 3 — Maximum | 5 | [256, 512, 1024, 2048, 4096] | ≈ 1.3 s | 15.5× |

### 4.4 Layer Payload Structure

**Real layer (layer index 0 before shuffling):**
```
PKCS7-padded( REAL_MAGIC(5B) ‖ plaintext )
```
where `REAL_MAGIC = b"MFSU\x04"`.

**Decoy layers:**
Pseudorandom bytes of the same length, derived from the password and per-layer salt — statistically indistinguishable from the real layer.

### 4.5 Layer Order Encryption

The shuffled order array is encrypted using MFSU-KS with a fixed salt (`MFSU_ORDER_SALT_`, 16 bytes) and `IV_ORD`. Without the correct password, the order cannot be recovered and all layers must be tried.

---

## 5. Binary Format — MFSU Standard v3 (`.fyx` legacy compatible)

### 5.1 File Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  MAGIC      6 bytes   "MFSUv3"                                  │
│  VERSION    1 byte    0x03                                      │
│  IV        16 bytes   Encryption IV (random)                    │
│  SALT      16 bytes   KDF salt (random)                         │
│  MSALT     16 bytes   MAC salt (random, independent of KDF)     │
│  MAC       32 bytes   HMAC-SHA3-256 (Encrypt-then-MAC)          │
│  CTEXT      N bytes   PKCS7-padded plaintext XOR keystream      │
└─────────────────────────────────────────────────────────────────┘
Header: 87 bytes fixed
```

### 5.2 Key Derivation

```
key_material (96B) = MFSU-KDF(password, salt)
enc_key      (64B) = key_material[0:64]
mac_key_base (32B) = key_material[64:96]
mac_key      (32B) = SHA3-256(mac_key_base ‖ msalt)
```

### 5.3 Encryption

```
padded     = PKCS7(plaintext)
keystream  = MFSU-KS(enc_key, iv, len(padded))
ciphertext = padded XOR keystream
```

### 5.4 Authentication

```
auth_data = iv ‖ salt ‖ msalt ‖ ciphertext
mac       = HMAC-SHA3-256(mac_key, auth_data)
```

MAC is verified before any decryption attempt using `hmac.compare_digest` (constant-time).

---

## 6. Format Detection

An implementation shall determine the sub-format as follows:

```
if file[0:6] == b"MFSUv4":
    → FractalShield v4 (primary .fyx format)
elif file[0:6] == b"MFSUv3":
    → MFSU Standard v3 (legacy .fyx format)
else:
    → Not a valid .fyx file
```

The version byte at offset 6 provides a secondary check. Both sub-formats may coexist under the `.fyx` extension; implementations must handle both.

---

## 7. Security Properties

### 7.1 Confidentiality

The MFSU-KDF requires ≈ 8 MB of working memory per derivation attempt. This limits GPU-based brute-force attacks to approximately 3,000 parallel threads on current hardware (RTX 4090, 24 GB VRAM), compared to millions per second for direct hash-based KDFs.

### 7.2 Integrity

HMAC-SHA3-256 in Encrypt-then-MAC mode provides authenticated encryption. A single tampered byte causes MAC verification to fail before any decryption is attempted. Error messages are intentionally generic to prevent oracle attacks.

### 7.3 Deniability (FractalShield v4 only)

The file contains N layers of equal size. All layers are statistically identical under ciphertext-only analysis. The attacker does not know which layer, if any, contains the real plaintext, nor whether a given password attempt was correct. At Level 3, breaking the file requires 15.5× the computational cost of a single decryption attempt.

### 7.4 Timing Side-Channels

Field normalization always divides by `max(|ψ|, 1)` unconditionally, eliminating the conditional branch present in v2 (`if max > 1`). MAC verification uses `hmac.compare_digest` throughout.

### 7.5 Two-Time Pad Prevention

Every encryption operation generates a fresh 16-byte IV and 16-byte salt via `os.urandom`. Keystream reuse is computationally infeasible.

### 7.6 Known Limitations

The MFSU cryptographic architecture has not been independently audited. The fractal field serves as an entropy source; SHA3-256 provides the uniformity guarantee. The security of the encryption ultimately rests on the computational hardness of the KDF and the strength of SHA3/HMAC-SHA3-256. Users requiring formally verified cryptography should consider this an experimental format.

---

## 8. 2FA Sub-Protocol (MFSU-TOTP)

Fractalyx includes a TOTP implementation based on the MFSU field. While not encoded in the `.fyx` file itself, it is part of the Fractalyx security suite:

- Time slot: 30-second windows (`t_slot = floor(unix_time / 30)`).
- Code derivation: `SHA3-512(secret ‖ t_slot ‖ "MFSU_TOTP_v3")` → initialise ψ → evolve 32 steps → `abs(sum(|ψ|×10⁹)) mod 10⁶`.
- Sliding window tolerance: codes for `t_slot − 1`, `t_slot`, and `t_slot + 1` are all valid, with replay protection via slot marking.

---

## 9. Version History

| Version | Magic | Format | Key improvement |
|---|---|---|---|
| v1 | `MFSUv1` | — | Initial proof of concept; no IV, SHA-512 direct KDF |
| v2 | `MFSUv2` | — | Added IV 16B, separate MAC salt, PKCS7; timing branch present |
| v3 | `MFSUv3` | MFSU Standard | Memory-hard KDF 8 MB, time-constant normalization, Merkle-Damgård fractal hash |
| v4 | `MFSUv4` | FractalShield | Multi-layer deniable encryption; encrypted layer order; `.fyx` extension |

---

## 10. Implementation Notes

### 10.1 Reference Implementation

The canonical implementation is `app.py` in the FractalyxwebLite repository. It is written in Python 3 and requires:

```
streamlit
numpy
scipy
matplotlib
pandas
```

### 10.2 Interoperability

Any conforming implementation must:

1. Correctly identify the sub-format from the first 6 bytes.
2. Verify the MAC before attempting decryption (Encrypt-then-MAC).
3. Use constant-time comparison for MAC verification.
4. Treat error messages as generic (do not reveal whether failure was due to wrong password or file corruption).
5. Reject files with unknown version bytes at offset 6.

### 10.3 File Naming Convention

Encrypted files should be named by appending `.fyx` to the original filename:

```
document.pdf  →  document.pdf.fyx
photo.jpg     →  photo.jpg.fyx
```

The original filename is recovered by stripping the `.fyx` suffix.

---

## 11. MFSU Constants (Normative)

The following constants are normative. Any implementation claiming compatibility with `.fyx` must use these exact values:

```
DELTA_F   = 0.921
BETA      = 1.079        # 2.0 − DELTA_F
HURST     = 0.541
DF_PROJ   = 2.921        # 2.0 + DELTA_F
GAMMA_NL  = 0.921        # = DELTA_F
SIGMA_ETA = 0.1

KDF_N     = 2048         # Spatial points for KDF field
KDF_M     = 256          # KDF evolution steps
KS_N      = 512          # Spatial points for keystream field

MAGIC_V4        = b"MFSUv4"
VERSION_V4      = b"\x04"
REAL_MAGIC      = b"MFSU\x04"
ORDER_SALT_FS   = b"MFSU_ORDER_SALT_"   # 16 bytes, fixed

MAGIC_V3        = b"MFSUv3"
VERSION_V3      = b"\x03"
IV_LEN          = 16
SALT_LEN        = 16
MAC_SALT_LEN    = 16
MAC_LEN         = 32
BLOCK_SIZE      = 16
```

---

## 12. License

This specification is released under the **Apache-2.0 License**, consistent with the FractalyxwebLite repository.

---

*End of FYX File Format Specification v1.0.0*
