import os
import struct
import subprocess
from tempfile import TemporaryDirectory
from typing import Literal, Tuple

import numpy as np
import torch

_MAGIC = b"ZXYZ1\n"
_MAGIC_Q = b"ZXYQ1\n"

# Quantized payload packing flags (stored in the 2-byte reserved field)
_FLAG_PLANAR = 1  # payload is stored as (3, N) instead of (N, 3)
_FLAG_ZIGZAG = 2  # int payload is zigzag-encoded to unsigned before storing

QuantMode = Literal["none", "f16", "int16", "int8", "int16_dz", "int8_dz"]
BiasMode = Literal["zero", "mean"]


def _write_xyz_raw(xyz: torch.Tensor, raw_path: str) -> Tuple[int, int]:
    """Write xyz tensor to a raw file with a tiny header.

    Format:
    - magic: 6 bytes (ZXYZ1\n)
    - N: uint32
    - C: uint32 (always 3)
    - payload: float32 little-endian, row-major, shape (N, 3)
    """
    if not isinstance(xyz, torch.Tensor):
        xyz = torch.as_tensor(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N,3), got {tuple(xyz.shape)}")

    xyz_cpu = xyz.detach().to(device="cpu", dtype=torch.float32).contiguous()
    n, c = int(xyz_cpu.shape[0]), int(xyz_cpu.shape[1])

    with open(raw_path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<II", n, c))
        f.write(xyz_cpu.numpy().astype(np.float32, copy=False).tobytes(order="C"))

    return n, c


def _write_xyz_quantized_raw(
    xyz: torch.Tensor,
    raw_path: str,
    quant_mode: QuantMode,
    step: float,
    bias_mode: BiasMode,
) -> Tuple[int, int]:
    """Write xyz with optional lossy quantization.

    Format:
    - magic: 6 bytes (ZXYQ1\n)
    - N: uint32
    - C: uint32 (always 3)
    - quant_mode: uint8 (1=f16, 2=int16, 3=int8, 4=int16_dz, 5=int8_dz)
    - bias_mode: uint8 (0=zero, 1=mean)
    - reserved: 2 bytes (bit flags)
    - step: float32
    - bias: float32 * 3
    - deadzone: float32   (only for *_dz)
    - payload: depends on quant_mode
        - f16: float16 (N,3)
        - int16: int16 (N,3) where q = round((x-bias)/step)
        - int8: int8 (N,3) where q = round((x-bias)/step)
    """
    if quant_mode == "none":
        return _write_xyz_raw(xyz, raw_path)

    if not isinstance(xyz, torch.Tensor):
        xyz = torch.as_tensor(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N,3), got {tuple(xyz.shape)}")

    xyz_cpu = xyz.detach().to(device="cpu", dtype=torch.float32).contiguous()
    n, c = int(xyz_cpu.shape[0]), int(xyz_cpu.shape[1])

    if bias_mode == "mean":
        bias = xyz_cpu.mean(dim=0)
    elif bias_mode == "zero":
        bias = torch.zeros((3,), dtype=torch.float32)
    else:
        raise ValueError(f"Unknown bias_mode: {bias_mode}")

    step_f = float(step)
    if not (step_f > 0.0):
        raise ValueError(f"step must be > 0, got {step}")

    # For compression ratio, we default to planar layout and zigzag for integer payloads.
    flags = 0

    if quant_mode == "f16":
        arr = (xyz_cpu - bias).to(torch.float16).numpy()
        # Planar tends to compress better with zstd.
        flags |= _FLAG_PLANAR
        payload = np.ascontiguousarray(arr.T).tobytes(order="C")
        quant_id = 1
    elif quant_mode == "int16":
        q = torch.round((xyz_cpu - bias) / step_f).to(torch.int32)
        q = torch.clamp(q, -32768, 32767).to(torch.int16)
        q_np = q.numpy()
        flags |= _FLAG_PLANAR
        flags |= _FLAG_ZIGZAG
        # ZigZag: signed int16 -> unsigned uint16
        q_i32 = q_np.astype(np.int32)
        u_u16 = ((q_i32 << 1) ^ (q_i32 >> 15)).astype(np.uint16)
        payload = np.ascontiguousarray(u_u16.T).tobytes(order="C")
        quant_id = 2
    elif quant_mode == "int8":
        q = torch.round((xyz_cpu - bias) / step_f).to(torch.int32)
        q = torch.clamp(q, -128, 127).to(torch.int8)
        q_np = q.numpy()
        flags |= _FLAG_PLANAR
        flags |= _FLAG_ZIGZAG
        # ZigZag: signed int8 -> unsigned uint8
        q_i32 = q_np.astype(np.int32)
        u_u8 = ((q_i32 << 1) ^ (q_i32 >> 7)).astype(np.uint8)
        payload = np.ascontiguousarray(u_u8.T).tobytes(order="C")
        quant_id = 3
    elif quant_mode in ("int16_dz", "int8_dz"):
        # Deadzone quantization: values within deadzone are forced to 0.
        # Outside deadzone we encode sign and a positive index: q = sign * (1 + round((|x|-dz)/step)).
        dz = float(step_f) if step_f > 0 else 0.0
        # NOTE: we interpret `step` as the step size, and users control deadzone separately via the public API.
        # The actual dz value is set by the caller in the wrapper; here we expect it to be packed in the raw header.
        raise RuntimeError("Internal error: *_dz quantization must be handled by _write_xyz_quantized_raw_dz")
    else:
        raise ValueError(f"Unknown quant_mode: {quant_mode}")

    bias_id = 1 if bias_mode == "mean" else 0
    with open(raw_path, "wb") as f:
        f.write(_MAGIC_Q)
        f.write(struct.pack("<II", n, c))
        f.write(struct.pack("<BBH", quant_id, bias_id, int(flags)))
        f.write(struct.pack("<f", step_f))
        f.write(bias.numpy().astype(np.float32, copy=False).tobytes(order="C"))
        f.write(payload)

    return n, c


def _read_xyz_raw(raw_path: str, device: str | torch.device) -> torch.Tensor:
    with open(raw_path, "rb") as f:
        magic = f.read(len(_MAGIC))
        if magic == _MAGIC:
            n, c = struct.unpack("<II", f.read(8))
            if c != 3:
                raise ValueError(f"Invalid channel count (expected 3) in {raw_path}: {c}")
            payload = f.read()
            arr = np.frombuffer(payload, dtype=np.float32)
            if arr.size != n * c:
                raise ValueError(f"Corrupted payload in {raw_path}: got {arr.size} floats, expected {n*c}")
            arr = arr.reshape((n, c))
            return torch.from_numpy(arr).to(device=device)

        if magic == _MAGIC_Q:
            n, c = struct.unpack("<II", f.read(8))
            if c != 3:
                raise ValueError(f"Invalid channel count (expected 3) in {raw_path}: {c}")
            quant_id, bias_id, flags = struct.unpack("<BBH", f.read(4))
            (step_f,) = struct.unpack("<f", f.read(4))
            bias = np.frombuffer(f.read(12), dtype=np.float32).reshape((1, 3))
            dz = None
            if quant_id in (4, 5):
                (dz,) = struct.unpack("<f", f.read(4))
            payload = f.read()

            planar = (flags & _FLAG_PLANAR) != 0
            zigzag = (flags & _FLAG_ZIGZAG) != 0

            if bias_id not in (0, 1):
                raise ValueError(f"Invalid bias_id in {raw_path}: {bias_id}")

            if quant_id == 1:
                arr = np.frombuffer(payload, dtype=np.float16)
                if arr.size != n * c:
                    raise ValueError(f"Corrupted f16 payload in {raw_path}: got {arr.size}, expected {n*c}")
                if planar:
                    arr = arr.reshape((c, n)).T
                else:
                    arr = arr.reshape((n, c))
                arr = arr.astype(np.float32)
                arr = arr + bias
                return torch.from_numpy(arr).to(device=device)

            if quant_id == 2:
                if zigzag:
                    u = np.frombuffer(payload, dtype=np.uint16)
                    if u.size != n * c:
                        raise ValueError(f"Corrupted int16 payload in {raw_path}: got {u.size}, expected {n*c}")
                    if planar:
                        u = u.reshape((c, n)).T
                    else:
                        u = u.reshape((n, c))
                    u_i32 = u.astype(np.int32)
                    q_i32 = (u_i32 >> 1) ^ (-(u_i32 & 1))
                    q = q_i32.astype(np.float32)
                else:
                    q = np.frombuffer(payload, dtype=np.int16)
                    if q.size != n * c:
                        raise ValueError(f"Corrupted int16 payload in {raw_path}: got {q.size}, expected {n*c}")
                    if planar:
                        q = q.reshape((c, n)).T
                    else:
                        q = q.reshape((n, c))
                    q = q.astype(np.float32)
                arr = q * float(step_f) + bias
                return torch.from_numpy(arr).to(device=device)

            if quant_id == 3:
                if zigzag:
                    u = np.frombuffer(payload, dtype=np.uint8)
                    if u.size != n * c:
                        raise ValueError(f"Corrupted int8 payload in {raw_path}: got {u.size}, expected {n*c}")
                    if planar:
                        u = u.reshape((c, n)).T
                    else:
                        u = u.reshape((n, c))
                    u_i32 = u.astype(np.int32)
                    q_i32 = (u_i32 >> 1) ^ (-(u_i32 & 1))
                    q = q_i32.astype(np.float32)
                else:
                    q = np.frombuffer(payload, dtype=np.int8)
                    if q.size != n * c:
                        raise ValueError(f"Corrupted int8 payload in {raw_path}: got {q.size}, expected {n*c}")
                    if planar:
                        q = q.reshape((c, n)).T
                    else:
                        q = q.reshape((n, c))
                    q = q.astype(np.float32)
                arr = q * float(step_f) + bias
                return torch.from_numpy(arr).to(device=device)

            if quant_id in (4, 5):
                # Deadzone decoding: q==0 => 0; else sign(q) * (dz + (|q|-1)*step)
                if dz is None:
                    raise ValueError(f"Missing deadzone in {raw_path}")
                if quant_id == 4:
                    if zigzag:
                        u = np.frombuffer(payload, dtype=np.uint16)
                        if u.size != n * c:
                            raise ValueError(f"Corrupted deadzone payload in {raw_path}: got {u.size}, expected {n*c}")
                        if planar:
                            u = u.reshape((c, n)).T
                        else:
                            u = u.reshape((n, c))
                        u_i32 = u.astype(np.int32)
                        q_i32 = (u_i32 >> 1) ^ (-(u_i32 & 1))
                        q = q_i32.astype(np.float32)
                    else:
                        q = np.frombuffer(payload, dtype=np.int16)
                        if q.size != n * c:
                            raise ValueError(f"Corrupted deadzone payload in {raw_path}: got {q.size}, expected {n*c}")
                        if planar:
                            q = q.reshape((c, n)).T
                        else:
                            q = q.reshape((n, c))
                        q = q.astype(np.float32)
                else:
                    if zigzag:
                        u = np.frombuffer(payload, dtype=np.uint8)
                        if u.size != n * c:
                            raise ValueError(f"Corrupted deadzone payload in {raw_path}: got {u.size}, expected {n*c}")
                        if planar:
                            u = u.reshape((c, n)).T
                        else:
                            u = u.reshape((n, c))
                        u_i32 = u.astype(np.int32)
                        q_i32 = (u_i32 >> 1) ^ (-(u_i32 & 1))
                        q = q_i32.astype(np.float32)
                    else:
                        q = np.frombuffer(payload, dtype=np.int8)
                        if q.size != n * c:
                            raise ValueError(f"Corrupted deadzone payload in {raw_path}: got {q.size}, expected {n*c}")
                        if planar:
                            q = q.reshape((c, n)).T
                        else:
                            q = q.reshape((n, c))
                        q = q.astype(np.float32)
                sign = np.sign(q)
                mag = np.abs(q)
                out = np.zeros_like(q, dtype=np.float32)
                nonzero = mag > 0
                out[nonzero] = sign[nonzero] * (float(dz) + (mag[nonzero] - 1.0) * float(step_f))
                out = out + bias
                return torch.from_numpy(out).to(device=device)

            raise ValueError(f"Unknown quant_id in {raw_path}: {quant_id}")

        raise ValueError(f"Invalid magic header in {raw_path}")


def zstd_compress_xyz_to_file(
    xyz: torch.Tensor,
    out_zst_path: str,
    zstd_bin: str = "zstd",
    level: int = 19,
    long: int | None = None,
    threads: int = 0,
) -> None:
    """Losslessly compress xyz tensor using zstd CLI.

    This preserves point order exactly because we compress raw float32 bytes.
    """
    os.makedirs(os.path.dirname(out_zst_path), exist_ok=True)

    with TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, "xyz.raw")
        _write_xyz_raw(xyz, raw_path)

        cmd = [zstd_bin]

        # Higher levels improve ratio but cost time.
        level_i = int(level)
        if level_i > 19:
            cmd.append("--ultra")
        cmd.append(f"-{level_i}")

        # Larger window can improve ratio on large/structured data.
        if long is not None:
            cmd.append(f"--long={int(long)}")

        # Multi-threading (0 => use all cores).
        cmd.append(f"-T{int(threads)}")

        cmd += ["-q", "-f", "-o", out_zst_path, raw_path]
        subprocess.run(cmd, check=True)


def zstd_decompress_xyz_from_file(
    zst_path: str,
    device: str | torch.device = "cuda",
    zstd_bin: str = "zstd",
) -> torch.Tensor:
    """Decompress xyz tensor previously written by zstd_compress_xyz_to_file."""
    if not os.path.exists(zst_path):
        raise FileNotFoundError(f"zstd bitstream not found: {zst_path}")

    with TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, "xyz.raw")
        cmd = [zstd_bin, "-d", "-q", "-f", "-o", raw_path, zst_path]
        subprocess.run(cmd, check=True)
        return _read_xyz_raw(raw_path, device=device)


def zstd_compress_xyz_quantized_to_file(
    xyz: torch.Tensor,
    out_zst_path: str,
    quant_mode: QuantMode = "int16",
    step: float = 1e-4,
    deadzone: float | None = None,
    bias_mode: BiasMode = "mean",
    zstd_bin: str = "zstd",
    level: int = 22,
    long: int | None = 31,
    threads: int = 0,
) -> None:
    """Lossy quantize xyz then compress using zstd.

    - Preserves point order.
    - quant_mode='int16' typically compresses better than 'f16'.
    """
    os.makedirs(os.path.dirname(out_zst_path), exist_ok=True)
    with TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, "xyz_q.raw")
        if quant_mode in ("int16_dz", "int8_dz"):
            _write_xyz_quantized_raw_dz(
                xyz,
                raw_path,
                quant_mode=quant_mode,
                step=step,
                deadzone=(float(deadzone) if deadzone is not None else float(step)),
                bias_mode=bias_mode,
            )
        else:
            _write_xyz_quantized_raw(xyz, raw_path, quant_mode=quant_mode, step=step, bias_mode=bias_mode)

        cmd = [zstd_bin]
        level_i = int(level)
        if level_i > 19:
            cmd.append("--ultra")
        cmd.append(f"-{level_i}")
        if long is not None:
            cmd.append(f"--long={int(long)}")
        cmd.append(f"-T{int(threads)}")
        cmd += ["-q", "-f", "-o", out_zst_path, raw_path]
        subprocess.run(cmd, check=True)


def zstd_roundtrip_xyz_quantized(
    xyz: torch.Tensor,
    out_zst_path: str,
    device: str | torch.device,
    quant_mode: QuantMode = "int16",
    step: float = 1e-4,
    deadzone: float | None = None,
    bias_mode: BiasMode = "mean",
    zstd_bin: str = "zstd",
    level: int = 22,
    long: int | None = 31,
    threads: int = 0,
) -> torch.Tensor:
    """Lossy quantize + zstd compress then decompress back to float32."""
    zstd_compress_xyz_quantized_to_file(
        xyz,
        out_zst_path,
        quant_mode=quant_mode,
        step=step,
        deadzone=deadzone,
        bias_mode=bias_mode,
        zstd_bin=zstd_bin,
        level=level,
        long=long,
        threads=threads,
    )
    return zstd_decompress_xyz_from_file(out_zst_path, device=device, zstd_bin=zstd_bin)


def _write_xyz_quantized_raw_dz(
    xyz: torch.Tensor,
    raw_path: str,
    quant_mode: QuantMode,
    step: float,
    deadzone: float,
    bias_mode: BiasMode,
) -> Tuple[int, int]:
    if quant_mode not in ("int16_dz", "int8_dz"):
        raise ValueError(f"quant_mode must be *_dz, got {quant_mode}")
    if not isinstance(xyz, torch.Tensor):
        xyz = torch.as_tensor(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N,3), got {tuple(xyz.shape)}")

    xyz_cpu = xyz.detach().to(device="cpu", dtype=torch.float32).contiguous()
    n, c = int(xyz_cpu.shape[0]), int(xyz_cpu.shape[1])

    if bias_mode == "mean":
        bias = xyz_cpu.mean(dim=0)
    elif bias_mode == "zero":
        bias = torch.zeros((3,), dtype=torch.float32)
    else:
        raise ValueError(f"Unknown bias_mode: {bias_mode}")

    step_f = float(step)
    dz_f = float(deadzone)
    if not (step_f > 0.0):
        raise ValueError(f"step must be > 0, got {step}")
    if dz_f < 0.0:
        raise ValueError(f"deadzone must be >= 0, got {deadzone}")

    centered = xyz_cpu - bias
    abs_centered = torch.abs(centered)
    sign = torch.sign(centered)
    # k = 0 if |x| <= dz else 1 + round((|x|-dz)/step)
    k = torch.zeros_like(centered, dtype=torch.int32)
    mask = abs_centered > dz_f
    k_val = 1 + torch.round((abs_centered[mask] - dz_f) / step_f).to(torch.int32)
    k[mask] = k_val
    q = (sign.to(torch.int32) * k).to(torch.int32)

    if quant_mode == "int16_dz":
        q = torch.clamp(q, -32768, 32767).to(torch.int16)
        q_np = q.numpy()
        # Planar + ZigZag for better compression ratio.
        q_i32 = q_np.astype(np.int32)
        u_u16 = ((q_i32 << 1) ^ (q_i32 >> 15)).astype(np.uint16)
        payload = np.ascontiguousarray(u_u16.T).tobytes(order="C")
        quant_id = 4
    else:
        q = torch.clamp(q, -128, 127).to(torch.int8)
        q_np = q.numpy()
        q_i32 = q_np.astype(np.int32)
        u_u8 = ((q_i32 << 1) ^ (q_i32 >> 7)).astype(np.uint8)
        payload = np.ascontiguousarray(u_u8.T).tobytes(order="C")
        quant_id = 5

    bias_id = 1 if bias_mode == "mean" else 0
    with open(raw_path, "wb") as f:
        f.write(_MAGIC_Q)
        f.write(struct.pack("<II", n, c))
        flags = int(_FLAG_PLANAR | _FLAG_ZIGZAG)
        f.write(struct.pack("<BBH", quant_id, bias_id, flags))
        f.write(struct.pack("<f", step_f))
        f.write(bias.numpy().astype(np.float32, copy=False).tobytes(order="C"))
        f.write(struct.pack("<f", dz_f))
        f.write(payload)

    return n, c


def zstd_roundtrip_xyz(
    xyz: torch.Tensor,
    out_zst_path: str,
    device: str | torch.device,
    zstd_bin: str = "zstd",
    level: int = 19,
    long: int | None = None,
    threads: int = 0,
) -> torch.Tensor:
    """Compress then decompress xyz via zstd (order-preserving, lossless)."""
    zstd_compress_xyz_to_file(xyz, out_zst_path, zstd_bin=zstd_bin, level=level, long=long, threads=threads)
    return zstd_decompress_xyz_from_file(out_zst_path, device=device, zstd_bin=zstd_bin)
