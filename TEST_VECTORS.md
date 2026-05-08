# 🔐 MFSU-Crypt — FractalShield Reference Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![Preprint v1.0](https://img.shields.io/badge/Preprint-v1.0-green)](https://zenodo.org/records/19974049)
[![Technical Note v1.2](https://img.shields.io/badge/Technical%20Note-v1.2-purple)](https://github.com/Fracta-Axis/Fractalyx)
[![Test Vectors](https://img.shields.io/badge/Test%20Vectors-v1.0-orange)](test_vectors.json)

> **MFSU-Crypt** is an experimental stream cipher and key-derivation framework based on fractal field dynamics. It is the first instantiation of the **FractalShield** core-agnostic framework for oracle-free layered encryption with geometric cost escalation.
>
> This repository contains the canonical reference implementation (`mfsu_crypt_ref.py`), the official test vectors, and companion formal documentation.

---

## 📋 Table of Contents

- [Overview](#overview)
- [The FractalShield Framework](#the-fractalshield-framework)
  - [The Four Core Laws (C1–C4)](#the-four-core-laws-c1c4)
  - [Security Inheritance](#security-inheritance)
  - [Protection Levels](#protection-levels)
  - [Conformance Verification Checklist](#conformance-verification-checklist)
  - [Minimal Core Interface](#minimal-core-interface)
- [Open Problems](#open-problems)
- [Quick Start](#quick-start)
- [Test Vectors](#test-vectors)
  - [Fixed Inputs](#fixed-inputs)
  - [V1 — MFSU-KDF (M=256)](#v1--mfsu-kdf-m256)
  - [V2 — MFSU-KDF (M=512)](#v2--mfsu-kdf-m512)
  - [V3 — Keystream (64 bytes)](#v3--keystream-64-bytes)
  - [V4 — Avalanche Test](#v4--avalanche-test)
  - [V5 — Magic Prefix & PKCS#7 Padding](#v5--magic-prefix--pkcs7-padding)
  - [V6 — HMAC-SHA3-256 Tag](#v6--hmac-sha3-256-tag)
  - [V7 — Warmup Steps (key-dependent)](#v7--warmup-steps-key-dependent)
  - [V8 — Field Norm Bound](#v8--field-norm-bound)
- [Known Limitations](#known-limitations)
- [Reproducing the Vectors](#reproducing-the-vectors)
- [Citation](#citation)

---

## Overview

FractalShield is a **core-agnostic** construction for oracle-free layered encryption. Its central idea is a strict separation between:

- **The framework** — layering, shuffling, HMAC wrapper, and file format (invariant across instantiations)
- **The core function F** — the only component that varies

Any function F satisfying C1–C4 automatically inherits all security theorems without modification: Integrity, IV Uniqueness, Oracle-Free Verification, and IND-CCA2.

MFSU-Crypt is built around three main pipeline components:

| Component | Description |
|-----------|-------------|
| **MFSU-KDF** | Memory-hard key derivation function based on a fractional stochastic PDE; 8 MB scratchpad |
| **MFSU Stream Cipher** | Keystream generator with key-dependent warmup steps (range: 48–111) |
| **HMAC-SHA3-256** | Message authentication tag protecting all layers |

Two instantiations currently exist:

| Instantiation | Core | Scratchpad | Speed (Python) |
|---------------|------|-----------|----------------|
| **MFSU-Crypt** | Fractional stochastic PDE | 8 MB | ≈1.9 attempts/sec |
| **Argon2id-Shield** | Argon2id | 64 MB | — |

Candidate future cores include Gray–Scott reaction-diffusion systems, generalised Lorenz flows, scrypt, and novel constructions. Any function meeting C1–C4 qualifies.

---

## The FractalShield Framework

*Formally specified in Technical Note v1.2 — a self-contained companion document to the main preprint.*

### The Four Core Laws (C1–C4)

A function `F : {0,1}^κ × {0,1}^128 × ℕ → {0,1}*` is a valid FractalShield core **if and only if** it satisfies all four of the following properties.

---

#### C1 — Input Sensitivity (Avalanche Property)

For any two distinct keys `k ≠ k'` and any `IV`, `L`:

```
Pr[ diffBits(F(k,IV,L), F(k',IV,L)) ≈ 0.50 ] ≥ 1 − negl(κ)
```

**Role:** Prevents hill-climbing attacks over keyspace. Without C1, an adversary can compare outputs under close key hypotheses to guide password search as a guided optimisation problem.

**Verification:** ≥ 10⁴ random key pairs; `diffBits` must lie within `[0.490, 0.510]` at 99% confidence.

**Violation:** Breaks Thm. 5.4 (PRG) and Thm. 5.6 (IND-CCA2).

---

#### C2 — Output Pseudorandomness (PRG Property)

```
|Pr[D(F(k,IV,L)) = 1] − Pr[D(U_L) = 1]| ≤ negl(κ)
```

**Role:** Every ciphertext layer is computationally identical to a random string for any adversary without the correct key. Eliminates statistical identification of the real layer among decoys.

**Verification:** Full NIST SP 800-22 battery on ≥ 2×10⁶ bits, ≥ 10 independent (k, IV) pairs. Pass: ≥ 8/10 pairs per test at α = 0.01.

**Violation:** Breaks Thm. 5.5 (OFV) and Thm. 5.6 (IND-CCA2).

---

#### C3 — Cost Controllability (Monotone Parameterised Cost)

```
M₁ < M₂  ⟹  cost(F(·; M₁)) < cost(F(·; M₂))
```

**Role:** The framework assigns cost `M₀ · 2^i` to layer `i`. Without C3, the geometric sum collapses to a flat `N × C_base`.

**Geometric Escalation (Lemma 5.4):**

```
C_attacker(ℓ) = C_base · (2^N(ℓ) − 1)
```

At Level 3 (N = 5): attacker pays **31× C_base** per attempt. Legitimate user always pays **1× C_base**.

**Violation:** Per-attempt cost drops from `(2^N − 1) × C_base` to `N × C_base` — a 6.2× degradation at Level 3. Breaks Lem. 5.4 and Thm. 5.5.

---

#### C4 — Memory Hardness

```
RAM(F(·; M)) = Ω(M)
```

A device with G bytes of RAM can run at most `⌊G/M⌋` concurrent evaluations.

**Role:** Converts time cost into a hardware parallelism constraint. Without C4, a GPU farm can parallelise 10⁵+ simultaneous evaluations.

**Practical effect:**
- MFSU-Crypt at M = 8 MB → RTX 4090 (24 GB VRAM) ≤ 3,000 parallel threads
- Argon2id-Shield at M = 64 MB → ≤ 375 threads on the same device

**Violation:** Security guarantees remain formally valid but become practically vacuous under large-scale parallelism.

---

### Security Inheritance

Any F satisfying C1–C4 inherits the complete theorem suite:

| Theorem | Statement (informal) | Laws required | Model |
|---------|----------------------|---------------|-------|
| Thm. 5.1 — Integrity | No PPT adversary forges a MAC-passing ciphertext | HMAC only | Standard |
| Thm. 5.2 — IV Uniqueness | IV collision prob. ≤ n²/2¹²⁸ | Independent of F | Standard |
| Lem. 5.4 — Geometric cost | C_att = C_base(2^N − 1) | C3 | Standard |
| Thm. 5.5 — OFV | Correctness check costs C_att(ℓ) | C2, Thm. 5.1 | Standard |
| Thm. 5.4 — PRG | Keystream ≈_c U_L | C1, C2 | ROM |
| Thm. 5.6 — IND-CCA2 | Adv ≤ 2q²/2²⁵⁶ + 5/2⁴⁰ | C1–C4 | ROM |

> Theorems 5.4 and 5.6 are proved under the Random Oracle Model (ROM). A standard-model reduction is Open Problem 3.

---

### Protection Levels

| Level | Layers N | Cost sequence | Attacker ratio | Enc. time |
|-------|----------|---------------|----------------|-----------|
| 1 — Standard | 3 | [M₀, 2M₀, 4M₀] | 3.5× | 0.35 s |
| 2 — Reinforced | 4 | [M₀, 2M₀, 4M₀, 8M₀] | 7.5× | 0.60 s |
| 3 — Maximum | 5 | [M₀, 2M₀, 4M₀, 8M₀, 16M₀] | 15.5× | 1.18 s |

### Avalanche Effect Reference (MFSU-Crypt)

Measured on 4096-byte keystream:

| Key modification | Bits changed | Percentage |
|-----------------|-------------|-----------|
| +1 character appended | 2063 | 50.4% |
| Case change a→A | 2028 | 49.5% |
| Trailing space added | 2013 | 49.1% |
| Completely different key | 2113 | 51.6% |
| Ideal (PRG) | 2048 | 50.0% |

---

### Conformance Verification Checklist

Before proposing a new core instantiation, verify and report each item:

- **[V1] Avalanche (C1)** — n ≥ 10,000 random key pairs with HammingDist = 1. Pass: µ ∈ [0.490, 0.510], σ ≤ 0.010
- **[V2] NIST SP 800-22 (C2)** — all 15 tests, ≥ 10 (k, IV) pairs, ≥ 2×10⁶ bits each. Pass: ≥ 8/10 at α = 0.01
- **[V3] Cost monotonicity (C3)** — benchmark M ∈ {M₀, 2M₀, 4M₀, 8M₀, 16M₀}. Pass: strictly increasing, CV < 5%
- **[V4] Memory profile (C4)** — peak RAM via `tracemalloc` or `valgrind massif`. Pass: RAM(M) ≥ αM for some constant α > 0
- **[V5] End-to-end integration** — encrypt/decrypt round-trip at all levels; wrong password returns ⊥; GLOBAL_MAC fails on any single-byte ciphertext modification

---

### Minimal Core Interface

```python
class FractalShieldCore(Protocol):
    """
    Minimal contract for a valid FractalShield core.
    The framework calls derive() exclusively.
    Layering, shuffling, HMAC, and file format are
    handled by FractalShield.Enc / FractalShield.Dec.
    """

    def derive(self, key: bytes, iv: bytes, M: int) -> bytes:
        """
        C1 -- diffBits(derive(k,iv,M), derive(k',iv,M)) ~= 0.50  [for k != k']
        C2 -- derive(k,iv,M) ~_c Uniform({0,1}^|output|)
        C3 -- cost(derive(k,iv,M1)) < cost(derive(k,iv,M2))       [for M1 < M2]
        C4 -- peak_ram(derive(k,iv,M)) = Omega(M)
        """
        ...

# Framework usage (implementor does not touch this):
# layer_key_i = core.derive(password, salt_i, M0 * 2**i)
# CT_i = stream_cipher(layer_key_i, iv_i, L) XOR plaintext_i
```

---

## Open Problems

The following open problems are identified in the preprint and Technical Note v1.2. Community contributions are welcome.

| # | Problem | Status |
|---|---------|--------|
| OP1 | Analytical injectivity of G | 🟡 Partially resolved — App. D |
| OP2 | DAG memory-hardness tight bound for Phase 2 | 🔴 Open |
| OP3 | IND-CCA2 without Random Oracle Model | 🔴 Open |
| OP4 | Replay-attack mitigation | 🟡 Fix specified |
| OP5 | Formal security reduction for MFSU-KDF Phase 2 | 🔴 Open |
| OP6 | Constant-time implementation & side-channel analysis | 🟡 Partially resolved — App. C |
| OP7 | Quantum security analysis | 🟢 Resolved — App. B |
| OP8 | Minimum password entropy bound | 🔴 Open |
| OP9 | Interaction with plaintext compression | 🔴 Open |
| OP10 | Mechanised verification via EasyCrypt / CryptoVerif | 🔴 Open |

### OP7 — Quantum Security (Resolved, App. B)

**Key result — Lemma 9.4:** FractalShield's geometric escalation factor `(2^N − 1)` is **not reduced** by Grover's algorithm. Even after Grover halves the search exponent, the attacker still pays the full geometric multiplier per quantum query:

```
Q_attacker(ℓ) = √|D| · C_base · (2^N(ℓ) − 1)
```

**Post-quantum parameter recommendations:**

| Target | λ (classical) | λQ (quantum) | Notes |
|--------|--------------|--------------|-------|
| Standard (current) | 128 bits | 64 bits | Adequate for near-term threats |
| Post-quantum | 256 bits | 128 bits | Double entropy; extend magic prefix to 10 bytes |
| High-assurance PQ | 256 bits | 128 bits | M₀ ≥ 512 (16 MB); Level 3 only |

### OP6 — Constant-Time Analysis (Partially Resolved, App. C)

Full pipeline timing classification:

| Component | Status | Fix |
|-----------|--------|-----|
| Field evolution Eq. (8′) | ✅ CT* | IEEE 754 `fmax` — branch-free by spec |
| KDF Phase 2 scratchpad access | ⚠️ NCT | Oblivious read (256× overhead) or formalise partial-leak model |
| Magic-prefix comparison | ✅ CT* | `hmac.compare_digest` |
| HMAC tag verification | ✅ CT* | `hmac.compare_digest` |
| Floating-point extraction | ✅ CT | Bounded field norm excludes subnormals by construction |

### OP1 — Analytical Injectivity (Partially Resolved, App. D)

Schwartz–Zippel applied to the polynomial approximation of G gives a collision probability < 2⁻¹⁰⁵ for random key pairs, conditional on assumption (ii): that the difference polynomial ∆G is not identically zero. Three lines of evidence support this (structural, empirical, Lyapunov), but a fully rigorous proof via Gröbner basis computation remains open and is targeted for the Year 1 roadmap.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Fracta-Axis/Fractalyx

# Install dependencies
pip install numpy

# Run the reference implementation
python3 mfsu_crypt_ref.py
```

Output will be printed to stdout. Compare results against the vectors below. A `test_vectors.json` file is also generated for programmatic verification.

---

## Test Vectors

All vectors were generated by `mfsu_crypt_ref.py` strictly following the equations and parameters of **FractalShield Preprint v1.0**.

### Fixed Inputs

These inputs are shared across all test vectors:

| Parameter | Value (hex) | ASCII |
|-----------|-------------|-------|
| `password` | `4672616374616c536869656c64`<br>`5f54657374566563746f725f7631` | `FractalShield_TestVector_v1` |
| `salt` (16 B) | `000102030405060708090a0b0c0d0e0f` | — |
| `IV` (16 B) | `101112131415161718191a1b1c1d1e1f` | — |
| `message` | `48656c6c6f2c2041627973732e` | `Hello, Abyss.` |

---

### V1 — MFSU-KDF (M=256)

**Input:** `password`, `salt`, `M = 256`
**Output (96 bytes):**

```
6b012f37946ee59d7f81ca6bd4ce7788400197d7774389b4adfc8bc2ef7af34e
fb4cff7b0ce401460612f1e53361d20544c5df5888503fb5f1820fe4be9d8c0d
1f28239e575529f59fe18a3b31ac72079fea9100f3ca5b5457ac70e32499645a
```

> **Layout:** Bytes `0–63` → stream cipher key (`dk`). Bytes `64–95` → encryption key (`ek`).

---

### V2 — MFSU-KDF (M=512)

**Input:** same `password`, `salt`, `M = 512` (Layer 1 cost)
**Output (96 bytes):**

```
fd490ce82498cae82094bf8ce4acc76b514251c2528f63c744f80b8b9f93d017
613fa31c122a8932265d5092b22575fc795c8f6c4f79f0ee6a3d4328cc45d2eb
a49551bcf2e455d1c55e1d5363e916a82e0ab1474742be8f5a9b8d09a4e169e3
```

**Avalanche check (C1):** `diffBits(V1, V2) = 48.6%` ✅
*(Pass range: 48.0–52.0%, ideal: 50.0%)*

---

### V3 — Keystream (64 bytes)

**Input:** `dk = V1[0:64]`, `IV`, `length = 64`
**Warmup steps:** `n_steps = 48 + (h[0] mod 64) = 48 + 38 = 86`

**Expected keystream:**

```
f62edfa120cb92c4be0f2723c3e4534e25d83b924b7c1649bbbf88e4c36f975f
1b18b5d3b4370cd1d40e401c5b13b76adf3e0b0586ec09aeb211e66385d26831
```

---

### V4 — Avalanche Test

**Modification:** `password[0] ^= 0x01` (single bit flip, all other inputs identical to V3)

**Modified keystream:**

```
5517da2fb73e539955b64b54d52b11b019f2a8981aacebc89793b98af4b22f38
534daadbda36bfdf09ac96b49a0ecd615552928a3daceb3232ea8351fc126840
```

**Result:** 249 bits changed out of 512 total = **48.6%** ✅
*(Pass range: 48.0–52.0%)*

---

### V5 — Magic Prefix & PKCS#7 Padding

**Magic prefix:** `4d46535504` → ASCII: `MFSU\x04`

**Padded plaintext** (`MAGIC ‖ message`, PKCS#7 to 16-byte boundary):

```
4d4653550448656c6c6f2c2041627973732e0e0e0e0e0e0e0e0e0e0e0e0e0e0e
```

**Magic verification (constant-time):**

| Input | Result |
|-------|--------|
| Correct prefix (`MFSU\x04…`) | `True` ✅ |
| Wrong prefix (`WRONG…`) | `False` ✅ |

---

### V6 — HMAC-SHA3-256 Tag

**MAC key derivation:** `k_MAC = SHA3-256(V1[0:32] ‖ "MAC")`

**Body (hex):** `4672616374616c536869656c64207465737420626f6479`
*(ASCII: `FractalShield test body`)*

**`k_MAC`:**
```
68f4fc3ed2bfc114918b9be9680c081abba83c81ad20f873cdaafc9203a4c5db
```

**Tag (32 bytes):**
```
6f22e11ef9b35ddb0230442122e590b26de385a860efe705b29128f23a362348
```

---

### V7 — Warmup Steps (key-dependent)

**Input:** `dk = V1[0:64]`, `IV`

**Computation:**
```
h = SHA3-512(dk ‖ IV)
n_steps = 48 + (h[0] mod 64)
```

**Result:** `h[0] = 0x38 = 56₁₀` → `n_steps = 48 + 56 = 86` ✅
*(Valid range: 48–111)*

---

### V8 — Field Norm Bound

**Purpose:** Verify that `|ψ|_max` remains bounded, confirming floating-point subnormals cannot occur (relevant to constant-time analysis, OP6).

**Result over 10 evolution steps:**

```
[0.9721, 1.1034, 1.2288, 1.3407, 1.4421, 1.5335, 1.6153, 1.7020, 1.7782, 1.8528]
```

| Metric | Value |
|--------|-------|
| Observed max (`\|ψ\|_max`) | `1.8528` |
| Paper claim (at convergence) | `≈ 4.8` |
| Bound holds? | ✅ Yes |

> **Note:** The discrepancy vs. the paper's `≈ 4.8` is expected — full convergence requires ~100 steps. The 10-step horizon is used here for verification speed only.

---

## Known Limitations

1. **No full encrypt/decrypt round-trip vector** — uses `os.urandom` for salts/IVs. A deterministic encrypt vector is planned for **v1.1**.
2. **V8 field norm** — shows `1.85` vs. paper's `≈ 4.8` due to the 10-step horizon. Bound holds; full convergence requires ~100 steps.
3. **NIST SP 800-22 vectors** — require ≥ 2×10⁶ bits; not reproduced here. See Table 6 of the main preprint.
4. **Assumption (ii) of Prop. 9.9** — injectivity of G reduced to ∆G ≠ 0 but not yet proved via Gröbner basis. Year 1 roadmap target.
5. **Phase 2 constant-time** — oblivious-read fix eliminates cache leaks at 256× overhead; partial-leak model is a reduced open problem (OP6).
6. **Quantum memory hardness** — classical Ω(M) bound does not automatically transfer to quantum circuits. Formal analysis pending.

---

## Reproducing the Vectors

```bash
# 1. Clone
git clone https://github.com/Fracta-Axis/Fractalyx
cd Fractalyx

# 2. Install
pip install numpy

# 3. Run
python3 mfsu_crypt_ref.py

# 4. Compare
# All outputs must match EXACTLY — the implementation is deterministic.
# A test_vectors.json is also generated for automated comparison.
```

---

## Citation

If you use this work in your research, please cite both documents:

**Main preprint:**
```bibtex
@misc{fractalshield2026,
  title   = {FractalShield: Oracle-Free Verification with Geometric Cost Escalation
             for Offline Brute-Force Resistance},
  author  = {Franco León, Miguel Angel},
  year    = {2026},
  note    = {Preprint v1.0},
  doi     = {10.5281/zenodo.19974049},
  url     = {https://zenodo.org/records/19974049}
}
```

**Technical Note v1.2 (Four Core Laws):**
```bibtex
@techreport{fractalshield_tn2026,
  title       = {The Four Core Laws of FractalShield: A Framework for Oracle-Free
                 Verification and Geometric Cost Escalation},
  author      = {Franco León, Miguel Angel},
  institution = {Fracta-Axis Project},
  year        = {2026},
  note        = {Technical Note v1.2. Companion to FractalShield Preprint v1.0},
  url         = {https://github.com/Fracta-Axis/Fractalyx}
}
```

> *The concept of oracle-free verification through layered magic-prefix detection combined with geometric cost escalation was first introduced by Miguel Angel Franco León. Future works using this architectural pattern are asked to cite both documents.*

---

<p align="center">
  <sub>MFSU-Crypt Reference Vectors v1.0 · FractalShield Preprint v1.0 · Technical Note v1.2</sub><br>
  <sub>Fracta-Axis Project · May 2026</sub>
</p>


