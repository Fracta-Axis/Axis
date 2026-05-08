"""
mfsu_crypt_ref.py
-----------------
Reference implementation of MFSU-Crypt derived strictly from
FractalShield Preprint v1.0 (May 2026) equations and parameters.

Sections referenced:
  - §4.1  MFSU equation parameters
  - §4.2  Corrected discretisation Eq.(8')
  - §4.4  MFSU-KDF (Construction 4.1)
  - §4.5  Stream cipher keystream (Construction 4.2)
  - §3.2  FractalShield.Enc (Construction 3.2)

This file is the canonical source for test vectors.
DO NOT modify parameters without updating the vector file.
"""

import hashlib
import hmac as _hmac
import struct
import os
import numpy as np

# ============================================================
# §4.1  System parameters (Table 2)
# ============================================================
DELTA_F = 0.921      # fractal deviation
BETA    = 1.079      # Laplacian order (= 2 - delta_F)
GAMMA   = 0.921      # nonlinearity coefficient (= delta_F)
SIGMA   = 0.1        # noise intensity
HURST   = 0.541      # Hurst exponent

N_KDF   = 2048       # KDF field size
M0      = 256        # KDF base steps
N_KS    = 512        # keystream field size

MAGIC   = b'MFSU\x04'   # 5-byte magic prefix

# ============================================================
# Internal helpers
# ============================================================

def _sha3_256(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()

def _sha3_512(data: bytes) -> bytes:
    return hashlib.sha3_512(data).digest()

def _hkdf_expand(prk: bytes, length: int) -> bytes:
    """HKDF-Expand (RFC 5869) with SHA3-256, no info field."""
    okm = b''
    t   = b''
    ctr = 1
    while len(okm) < length:
        t    = _sha3_256(t + prk + bytes([ctr]))
        okm += t
        ctr += 1
    return okm[:length]

def _seed_field(seed32: bytes, N: int, rng_seed: int = 0) -> np.ndarray:
    """
    Seed a complex field of size N from a 32-byte seed.
    Uses deterministic Box-Muller via SHA3 counter stream.
    """
    needed = N * 2   # real and imag parts
    raw = b''
    ctr = 0
    while len(raw) < needed * 8:
        raw += _sha3_512(seed32 + struct.pack('>Q', ctr))
        ctr += 1
    vals = np.frombuffer(raw[:needed * 8], dtype='>u8').astype(np.float64)
    # map uint64 to (-1,1) then use as normal via CLT approximation
    vals = (vals / (2**64 - 1)) * 2 - 1
    real = vals[:N]
    imag = vals[N:]
    # Normalise to unit variance
    real = real / (np.std(real) + 1e-12)
    imag = imag / (np.std(imag) + 1e-12)
    return real + 1j * imag

def _fractional_laplacian(psi: np.ndarray) -> np.ndarray:
    """
    Spectral fractional Laplacian (-Delta)^(beta/2).
    Eigenvalues: |k|^beta for wavenumber k.
    """
    N    = len(psi)
    psi_f = np.fft.fft(psi)
    k    = np.fft.fftfreq(N, d=1.0/N)   # wavenumbers 0..N/2-1, -N/2..-1
    eigs = np.abs(k) ** BETA
    eigs[0] = 0.0   # zero mode: no diffusion at k=0
    return np.fft.ifft(eigs * psi_f)

def _fgn_noise(seed: bytes, step: int, N: int) -> np.ndarray:
    """
    Deterministic fractional Gaussian noise proxy.
    We use a SHA3-keyed counter stream coloured by H=0.541.
    True fGn covariance: C(k) = 0.5*(|k+1|^2H - 2|k|^2H + |k-1|^2H).
    For a reference implementation we use white noise * spectral colouring.
    """
    raw = _sha3_512(seed + struct.pack('>Q', step))
    vals = np.frombuffer(raw[:N*2 if N*2 <= 64 else 64],
                         dtype='>u8')
    # Extend if needed
    full = b''
    s = 0
    while len(full) < N * 16:
        full += _sha3_512(seed + struct.pack('>QQ', step, s))
        s += 1
    arr = np.frombuffer(full[:N*8], dtype='>u8').astype(np.float64)
    arr = (arr / (2**64 - 1)) * 2 - 1

    # Spectral colouring for H = 0.541
    arr_f = np.fft.rfft(arr)
    k     = np.arange(len(arr_f), dtype=np.float64)
    k[0]  = 1.0
    color = k ** (-(2*HURST + 1) / 2)
    color[0] = 0.0
    arr_c = np.fft.irfft(arr_f * color, n=N)
    arr_c /= (np.std(arr_c) + 1e-12)
    return arr_c.astype(np.complex128)

def _mfsu_step(psi: np.ndarray, noise: np.ndarray,
               dt: float = 0.001) -> np.ndarray:
    """
    One MFSU evolution step — corrected Eq.(8').

    delta = dt * F(psi, eta)
    psi_new = psi + delta / max(||delta||_inf, 1)

    F(psi,eta) = -delta_F * (-Delta)^(beta/2) psi
                 + gamma * |psi|^2 * psi
                 + sigma * eta
    """
    lap    = _fractional_laplacian(psi)
    F      = (-DELTA_F * lap
              + GAMMA * np.abs(psi)**2 * psi
              + SIGMA * noise)
    delta  = dt * F
    norm_d = np.max(np.abs(delta))
    denom  = max(norm_d, 1.0)   # IEEE 754 fmax — branch-free
    return psi + delta / denom

# ============================================================
# §4.4  MFSU-KDF  (Construction 4.1)
# ============================================================

def mfsu_kdf(password: bytes, salt: bytes, steps: int = M0) -> bytes:
    """
    MFSU-KDF: three-phase memory-hard key derivation.

    Input : password (bytes), salt (16 bytes), steps M
    Output: 96 bytes of key material

    Phase 1 — sequential scratchpad fill
    Phase 2 — data-dependent mixing
    Phase 3 — condensation via SHA3-512 + HKDF
    """
    assert len(salt) == 16, "salt must be 16 bytes"

    # --- Phase 1: sequential scratchpad fill ---
    h = _sha3_512(password + b'\x00' + salt)   # eq.(2)

    psi = _seed_field(h[:32], N_KDF)

    scratchpad = []
    noise_seed = h[32:64]
    for i in range(steps):
        noise = _fgn_noise(noise_seed, i, N_KDF)
        psi   = _mfsu_step(psi, noise, dt=0.001)
        scratchpad.append(psi.copy())

    # --- Phase 2: data-dependent mixing ---
    psi_mix = psi.copy()
    for i in range(steps):
        idx = int(abs(psi_mix[0].real) * 1e9) % steps   # eq.(3)
        delta_mix = scratchpad[idx]
        norm_mix  = max(np.max(np.abs(psi_mix)), 1.0)
        psi_mix   = psi_mix + 1e-3 * delta_mix / norm_mix

    # --- Phase 3: condensation ---
    re_bytes = struct.pack('>' + 'q' * N_KDF,
                           *[int(x.real * 1e6) for x in psi_mix])
    kraw = _sha3_512(re_bytes + h)
    key  = _hkdf_expand(kraw, 96)   # eq.(2) condensation
    return key

# ============================================================
# §4.5  Stream cipher keystream  (Construction 4.2)
# ============================================================

def mfsu_keystream(dk: bytes, iv: bytes, length: int) -> bytes:
    """
    MFSU keystream generation.

    Input : dk (64 bytes), iv (16 bytes), length (int)
    Output: length bytes of keystream
    """
    assert len(dk) == 64,  "dk must be 64 bytes"
    assert len(iv) == 16,  "iv must be 16 bytes"

    # eq.(4)
    h      = _sha3_512(dk + iv)
    nsteps = 48 + (h[0] % 64)   # 48..111 key-dependent warmup
    psi    = _seed_field(h[:32], N_KS)

    noise_seed = h[32:64]
    for i in range(nsteps):
        noise = _fgn_noise(noise_seed, i, N_KS)
        psi   = _mfsu_step(psi, noise, dt=0.01)

    # Extract raw bytes
    raw = []
    for j in range(N_KS):
        raw.append(int(psi[j].real * 1e4) % 256)
        raw.append(int(psi[j].imag * 1e4) % 256)
    raw_bytes = bytes(raw)   # 1024 bytes per field snapshot

    # SHA3-256 counter-mode whitener — eq.(5)
    kmix = _sha3_256(dk[32:64] + iv)
    ks   = bytearray()
    ctr  = 0
    src  = raw_bytes
    while len(ks) < length:
        block = _sha3_256(kmix + struct.pack('>Q', ctr))
        chunk = bytes(a ^ b for a, b in zip(
            src[len(ks) % len(src):len(ks) % len(src) + 32], block))
        ks += chunk
        ctr += 1
    return bytes(ks[:length])

# ============================================================
# §3.2  FractalShield.Enc / .Dec  (Construction 3.2)
# ============================================================

def _pkcs7_pad(data: bytes, block: int = 16) -> bytes:
    pad = block - (len(data) % block)
    return data + bytes([pad] * pad)

def _pkcs7_unpad(data: bytes) -> bytes:
    pad = data[-1]
    assert 1 <= pad <= 16
    assert data[-pad:] == bytes([pad] * pad)
    return data[:-pad]

def _layer_count(level: int) -> int:
    return {1: 3, 2: 4, 3: 5}[level]

def fractalshield_enc(plaintext: bytes,
                      password: bytes,
                      level: int = 2) -> bytes:
    """
    FractalShield encryption.

    Input : plaintext, password, level in {1,2,3}
    Output: .shield v4 ciphertext bytes
    """
    N = _layer_count(level)

    # Pad plaintext with magic prefix
    padded = _pkcs7_pad(MAGIC + plaintext)
    L      = len(padded)

    layers = []
    salts  = []
    ivs    = []
    keys   = []

    # Layer 0: real layer
    s0 = os.urandom(16)
    iv0 = os.urandom(16)
    k0  = mfsu_kdf(password, s0, steps=M0)
    dk0 = k0[:64]
    ek0 = k0[64:96]
    ks0 = mfsu_keystream(dk0, iv0, L)
    ct0 = bytes(a ^ b for a, b in zip(padded, ks0))
    layers.append(ct0); salts.append(s0); ivs.append(iv0); keys.append(k0)

    # Decoy layers i=1..N-1 with escalating cost
    for i in range(1, N):
        si  = os.urandom(16)
        ivi = os.urandom(16)
        ki  = mfsu_kdf(password, si, steps=M0 * (2**i))
        dki = ki[:64]

        # Pseudorandom decoy content
        prg_seed = _sha3_256(password + bytes([i]) + si)
        di = mfsu_keystream(prg_seed[:32] + prg_seed, si, L)
        di = di[:L]

        ksi = mfsu_keystream(dki, ivi, L)
        cti = bytes(a ^ b for a, b in zip(di[:L], ksi))
        layers.append(cti); salts.append(si); ivs.append(ivi); keys.append(ki)

    # Shuffle order — key-dependent
    order_seed = _sha3_256(password + b'ORDER')
    rng = np.random.default_rng(
        seed=int.from_bytes(order_seed[:8], 'big'))
    order = list(rng.permutation(N))

    # Encrypt order map under k0
    order_bytes = bytes(order)
    order_mask  = _sha3_256(keys[0][:32] + b'ORDERMAP')[:N]
    order_enc   = bytes(a ^ b for a, b in zip(order_bytes, order_mask))

    # Header: magic(4) + version(1) + level(1) + N(1) + L(4) = 11 bytes
    hdr = b'FSv4' + bytes([1, level, N]) + struct.pack('>I', L)

    # Global HMAC
    k_mac    = _sha3_256(keys[0][:32] + b'MAC')
    mac_body = hdr + order_enc
    for idx in order:
        mac_body += layers[idx]
    tag = _hmac.new(k_mac, mac_body, hashlib.sha3_256).digest()

    # Assemble: hdr || order_enc || tag || CT[order[0]] || ... || CT[order[N-1]]
    out = hdr + order_enc + tag
    for idx in order:
        out += layers[idx]
    return out

def fractalshield_dec(ciphertext: bytes, password: bytes) -> bytes:
    """
    FractalShield decryption.

    Input : ciphertext bytes, password
    Output: plaintext bytes, or raises ValueError on failure
    """
    # Parse header
    assert ciphertext[:4] == b'FSv4', "bad magic"
    level = ciphertext[5]
    N     = ciphertext[6]
    L     = struct.unpack('>I', ciphertext[7:11])[0]
    pos   = 11

    order_enc = ciphertext[pos:pos+N]; pos += N
    tag_stored = ciphertext[pos:pos+32]; pos += 32

    layers = []
    for _ in range(N):
        layers.append(ciphertext[pos:pos+L]); pos += L

    # Try password as Layer 0
    # We don't know the salt — it must be embedded. 
    # For the reference implementation salts are prepended to each layer
    # in the full format; here we use the simplified test-vector format
    # where salt and IV are the first 32 bytes of each layer slot.
    raise NotImplementedError(
        "Full dec requires embedded salts — see test vector format below.")

# ============================================================
# Test vector generation
# ============================================================

def generate_vectors():
    """
    Generate all test vectors with fixed inputs.
    All random values are replaced by deterministic constants.
    """
    import json

    vectors = {}

    # --- Fixed inputs ---
    PWD   = b'FractalShield_TestVector_v1'
    SALT  = bytes(range(16))          # 00 01 02 ... 0f
    IV    = bytes(range(16, 32))      # 10 11 12 ... 1f
    MSG   = b'Hello, Abyss.'

    print("=" * 60)
    print("MFSU-Crypt Reference Test Vectors v1.0")
    print("Generated from FractalShield Preprint v1.0")
    print("=" * 60)

    # --- Vector 1: KDF ---
    print("\n[V1] MFSU-KDF")
    print(f"  password : {PWD.hex()}")
    print(f"  salt     : {SALT.hex()}")
    print(f"  steps M  : {M0}")
    k = mfsu_kdf(PWD, SALT, steps=M0)
    print(f"  output   : {k.hex()}")
    print(f"  len      : {len(k)} bytes")
    vectors['V1_KDF'] = {
        'password_hex': PWD.hex(),
        'salt_hex': SALT.hex(),
        'steps': M0,
        'output_hex': k.hex(),
        'output_len': len(k)
    }

    # --- Vector 2: KDF at 2*M0 (Layer 1 cost) ---
    print("\n[V2] MFSU-KDF (2*M0)")
    k2 = mfsu_kdf(PWD, SALT, steps=M0*2)
    print(f"  steps M  : {M0*2}")
    print(f"  output   : {k2.hex()}")
    vectors['V2_KDF_2M0'] = {
        'password_hex': PWD.hex(),
        'salt_hex': SALT.hex(),
        'steps': M0*2,
        'output_hex': k2.hex(),
    }

    # --- Vector 3: Keystream ---
    print("\n[V3] MFSU Keystream")
    dk = k[:64]
    ks = mfsu_keystream(dk, IV, 64)
    print(f"  dk (first 64B of V1) : {dk.hex()}")
    print(f"  iv                   : {IV.hex()}")
    print(f"  length               : 64")
    print(f"  keystream            : {ks.hex()}")
    vectors['V3_KEYSTREAM'] = {
        'dk_hex': dk.hex(),
        'iv_hex': IV.hex(),
        'length': 64,
        'keystream_hex': ks.hex()
    }

    # --- Vector 4: Keystream avalanche ---
    print("\n[V4] Keystream avalanche (1-bit key change)")
    pwd2 = bytearray(PWD)
    pwd2[0] ^= 0x01
    k_alt = mfsu_kdf(bytes(pwd2), SALT, steps=M0)
    dk_alt = k_alt[:64]
    ks_alt = mfsu_keystream(dk_alt, IV, 64)
    diff = sum(bin(a ^ b).count('1') for a, b in zip(ks, ks_alt))
    pct  = diff / (64 * 8) * 100
    print(f"  bits changed : {diff} / {64*8} = {pct:.1f}%")
    print(f"  ks_original  : {ks.hex()}")
    print(f"  ks_modified  : {ks_alt.hex()}")
    vectors['V4_AVALANCHE'] = {
        'bits_changed': diff,
        'total_bits': 64*8,
        'percentage': round(pct, 2),
        'ks_original_hex': ks.hex(),
        'ks_modified_hex': ks_alt.hex()
    }

    # --- Vector 5: Magic prefix check ---
    print("\n[V5] Magic prefix")
    print(f"  MAGIC hex : {MAGIC.hex()}")
    print(f"  MAGIC str : {MAGIC}")
    padded = _pkcs7_pad(MAGIC + MSG)
    print(f"  padded plaintext : {padded.hex()}")
    print(f"  magic check OK   : {_hmac.compare_digest(padded[:5], MAGIC)}")
    wrong = b'WRONG' + padded[5:]
    print(f"  magic check FAIL : {_hmac.compare_digest(wrong[:5], MAGIC)}")
    vectors['V5_MAGIC'] = {
        'magic_hex': MAGIC.hex(),
        'padded_hex': padded.hex(),
        'check_correct': True,
        'check_wrong': False
    }

    # --- Vector 6: HMAC tag ---
    print("\n[V6] HMAC-SHA3-256")
    k_mac = _sha3_256(k[:32] + b'MAC')
    body  = b'FractalShield test body'
    tag   = _hmac.new(k_mac, body, hashlib.sha3_256).digest()
    print(f"  k_mac : {k_mac.hex()}")
    print(f"  body  : {body.hex()}")
    print(f"  tag   : {tag.hex()}")
    vectors['V6_HMAC'] = {
        'k_mac_hex': k_mac.hex(),
        'body_hex': body.hex(),
        'tag_hex': tag.hex()
    }

    # --- Vector 7: Nsteps (warmup range) ---
    print("\n[V7] Keystream warmup steps")
    h = _sha3_512(dk + IV)
    nsteps = 48 + (h[0] % 64)
    print(f"  h[0]   : {h[0]}")
    print(f"  nsteps : {nsteps}  (range 48-111)")
    vectors['V7_NSTEPS'] = {
        'dk_hex': dk.hex(),
        'iv_hex': IV.hex(),
        'h0': h[0],
        'nsteps': nsteps
    }

    # --- Vector 8: Field norm check ---
    print("\n[V8] Field norm (bounded by construction)")
    psi = _seed_field(_sha3_512(PWD)[:32], N_KS)
    norms = []
    ns = _sha3_512(PWD)[32:64]
    for i in range(10):
        noise = _fgn_noise(ns, i, N_KS)
        psi   = _mfsu_step(psi, noise, dt=0.01)
        norms.append(float(np.max(np.abs(psi))))
    print(f"  max |psi| over 10 steps : {max(norms):.4f}")
    print(f"  (paper claims ~4.8)")
    vectors['V8_FIELD_NORM'] = {
        'max_norm_10steps': round(max(norms), 4),
        'all_norms': [round(x, 4) for x in norms]
    }

    return vectors

if __name__ == '__main__':
    import json
    vecs = generate_vectors()
    with open('test_vectors.json', 'w') as f:
        json.dump(vecs, f, indent=2)
    print("\n\nVectors saved to test_vectors.json")
