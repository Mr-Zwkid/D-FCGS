import os
from tempfile import TemporaryDirectory
from typing import Tuple

import numpy as np
from plyfile import PlyData
import torch

# 量化到体素网格的精度，和项目里现有 GPCC 用法保持一致
VOXELIZE_SCALE_FACTOR = 16


def gpcc_encode(codec_path: str, ply_path: str, bin_path: str) -> None:
    """用 MPEG G-PCC (tmc3) 对几何点云进行编码。"""
    enc_cmd = (
        f"{codec_path} "
        "--mode=0 --trisoupNodeSizeLog2=0 --mergeDuplicatedPoints=0 --neighbourAvailBoundaryLog2=8 "
        "--intra_pred_max_node_size_log2=3 --positionQuantizationScale=1 --inferredDirectCodingMode=3 "
        "--maxNumQtBtBeforeOt=2 --minQtbtSizeLog2=0 --planarEnabled=0 --planarModeIdcmUse=0 --cabac_bypass_stream_enabled_flag=1 "
        f"--uncompressedDataPath={ply_path} --compressedStreamPath={bin_path} "
    )
    enc_cmd += "> nul 2>&1" if os.name == "nt" else "> /dev/null 2>&1"
    exit_code = os.system(enc_cmd)
    if exit_code != 0:
        raise RuntimeError(f"GPCC encoder failed with exit code {exit_code} (cmd: {enc_cmd})")


def gpcc_decode(codec_path: str, bin_path: str, recon_ply_path: str) -> None:
    """用 MPEG G-PCC (tmc3) 对几何点云进行解码，输出 binary ply。"""
    dec_cmd = (
        f"{codec_path} "
        "--mode=1 --outputBinaryPly=1 "
        f"--compressedStreamPath={bin_path} --reconstructedDataPath={recon_ply_path} "
    )
    dec_cmd += "> nul 2>&1" if os.name == "nt" else "> /dev/null 2>&1"
    exit_code = os.system(dec_cmd)
    if exit_code != 0:
        raise RuntimeError(f"GPCC decoder failed with exit code {exit_code} (cmd: {dec_cmd})")


def write_ply_geo_ascii(points_xyz_int: np.ndarray, ply_path: str) -> None:
    """把 (N,3) 的整数几何点写成 ASCII ply（tmc3 可直接读取）。"""
    if not ply_path.endswith(".ply"):
        raise ValueError("Destination path must be a .ply file.")
    if points_xyz_int.ndim != 2 or points_xyz_int.shape[1] != 3:
        raise ValueError("Input data must have shape (N, 3).")

    points_xyz_int = points_xyz_int.astype(int)
    with open(ply_path, "w") as f:
        f.writelines(
            [
                "ply\n",
                "format ascii 1.0\n",
                f"element vertex {points_xyz_int.shape[0]}\n",
                "property float x\n",
                "property float y\n",
                "property float z\n",
                "end_header\n",
            ]
        )
        for x, y, z in points_xyz_int:
            f.write(f"{x} {y} {z}\n")


def read_ply_geo_bin(ply_path: str) -> np.ndarray:
    """从 binary ply 读取 (N,3) 的 xyz。"""
    if not ply_path.endswith(".ply"):
        raise ValueError("Source path must be a .ply file.")

    ply_data = PlyData.read(ply_path).elements[0]
    xyz = np.stack([ply_data.data[name] for name in ["x", "y", "z"]], axis=1)
    return xyz


def _voxelize(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把连续 xyz 归一化到 [0,1] 再量化到整数体素坐标。"""
    xyz_min, xyz_max = xyz.min(axis=0), xyz.max(axis=0)
    denom = (xyz_max - xyz_min)
    # 避免退化情况除 0
    denom = np.where(denom == 0, 1.0, denom)

    voxelized = (xyz - xyz_min) / denom
    voxelized = np.round(voxelized * (2**VOXELIZE_SCALE_FACTOR - 1))
    return voxelized, xyz_min, xyz_max


def _devoxelize(voxelized: np.ndarray, xyz_min: np.ndarray, xyz_max: np.ndarray) -> np.ndarray:
    voxelized = voxelized.astype(np.float32)
    xyz_min = xyz_min.astype(np.float32)
    xyz_max = xyz_max.astype(np.float32)

    denom = (xyz_max - xyz_min)
    denom = np.where(denom == 0, 1.0, denom)

    xyz = voxelized / (2**VOXELIZE_SCALE_FACTOR - 1) * denom + xyz_min
    return xyz


def _sorted_voxels(voxelized: np.ndarray) -> np.ndarray:
    # 使用一个简单的 Morton-like 排序（与项目现有实现保持一致）
    base = voxelized.max() + 1
    keys = voxelized @ np.power(base, np.arange(voxelized.shape[1]))
    idx = np.argsort(keys, axis=0)
    return voxelized[idx]


def _write_binary_blob(dst_fh, src_path: str) -> None:
    with open(src_path, "rb") as f:
        data = f.read()
    dst_fh.write(np.array([len(data)], dtype=np.uint32).tobytes())
    dst_fh.write(data)


def _read_binary_blob(dst_path: str, src_fh) -> None:
    length = int(np.frombuffer(src_fh.read(4), dtype=np.uint32)[0])
    with open(dst_path, "wb") as f:
        f.write(src_fh.read(length))


def gpcc_compress_xyz_to_file(xyz: torch.Tensor, out_bin_path: str, gpcc_codec_path: str) -> None:
    """把 (N,3) 的 float xyz 用 GPCC 压缩到一个二进制文件。

    文件格式：
    - 24 bytes: xyz_min(3)*float32 + xyz_max(3)*float32
    - 4 bytes: uint32 length
    - length bytes: tmc3 输出的 bitstream
    """
    if not isinstance(xyz, torch.Tensor):
        xyz = torch.as_tensor(xyz)
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must have shape (N,3), got {tuple(xyz.shape)}")

    xyz_np = xyz.detach().cpu().numpy().astype(np.float32)
    voxelized, xyz_min, xyz_max = _voxelize(xyz_np)
    voxelized = _sorted_voxels(voxelized)

    os.makedirs(os.path.dirname(out_bin_path), exist_ok=True)
    with TemporaryDirectory() as tmp:
        ply_path = os.path.join(tmp, "voxelized_xyz.ply")
        write_ply_geo_ascii(voxelized, ply_path)

        gpcc_stream = os.path.join(tmp, "gpcc.bin")
        gpcc_encode(gpcc_codec_path, ply_path, gpcc_stream)

        with open(out_bin_path, "wb") as f:
            head = np.array([xyz_min, xyz_max], dtype=np.float32)
            f.write(head.tobytes())  # 24 bytes
            _write_binary_blob(f, gpcc_stream)


def gpcc_decompress_xyz_from_file(bin_path: str, gpcc_codec_path: str, device: str | torch.device = "cuda") -> torch.Tensor:
    """从 gpcc_compress_xyz_to_file 输出的二进制文件解压出 (N,3) 的 xyz。"""
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"GPCC xyz bitstream not found: {bin_path}")

    with TemporaryDirectory() as tmp:
        with open(bin_path, "rb") as f:
            head = np.frombuffer(f.read(24), dtype=np.float32)
            xyz_min, xyz_max = head[:3], head[3:]

            gpcc_stream = os.path.join(tmp, "gpcc.bin")
            _read_binary_blob(gpcc_stream, f)

        recon_ply = os.path.join(tmp, "recon.ply")
        gpcc_decode(gpcc_codec_path, gpcc_stream, recon_ply)
        voxelized = read_ply_geo_bin(recon_ply).astype(np.float32)
        voxelized = _sorted_voxels(voxelized)

        xyz = _devoxelize(voxelized, xyz_min, xyz_max)
        xyz_t = torch.from_numpy(xyz).to(device=device)
        return xyz_t


def gpcc_roundtrip_xyz(xyz: torch.Tensor, out_bin_path: str, gpcc_codec_path: str, device: str | torch.device = "cuda") -> torch.Tensor:
    """对 xyz 做一次 GPCC 压缩+解压，返回解压后的 xyz。"""
    gpcc_compress_xyz_to_file(xyz, out_bin_path, gpcc_codec_path)
    return gpcc_decompress_xyz_from_file(out_bin_path, gpcc_codec_path, device=device)
