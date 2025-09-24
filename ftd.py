"""Compute the Fréchet trajectory distance (FTD) for a single video pair."""

import argparse
import random
from pathlib import Path
from typing import Iterable, Sequence

import torch
import decord
import numpy as np
import torchvision.transforms as T
from PIL import Image
from frechetdist import frdist


MAX_FRAMES = 49
DEFAULT_COTRACKER_ROOT = Path("/home/ubuntu/co-tracker")
VALID_MASK_SUFFIXES: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp")


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_frame_count(video_path: str) -> int:
    """Return the number of frames in the video located at ``video_path``."""
    video_reader = decord.VideoReader(video_path)
    return len(video_reader)


def _frame_indices(start: int, end: int, max_num_frames: int, frame_sample_step: int | None) -> list[int]:
    """Return the frame indices sampled between ``start`` and ``end`` inclusive."""
    if end <= start:
        return [start]
    if end - start <= max_num_frames:
        return list(range(start, end))
    step = frame_sample_step or max(1, (end - start) // max_num_frames)
    return list(range(start, end, step))


def get_video_frames(
    video_path: str,
    width: int,
    height: int,
    skip_frames_start: int,
    skip_frames_end: int,
    max_num_frames: int,
    frame_sample_step: int | None = None,
) -> torch.FloatTensor:
    """Load frames from a video and return a normalized tensor shaped [T, C, H, W]."""
    with decord.bridge.use_torch():
        video_reader = decord.VideoReader(uri=video_path, width=width, height=height)
        video_num_frames = len(video_reader)
        start_frame = min(skip_frames_start, video_num_frames)
        end_frame = max(0, video_num_frames - skip_frames_end)

        indices = _frame_indices(start_frame, end_frame, max_num_frames, frame_sample_step)
        frames = video_reader.get_batch(indices=indices)
        frames = frames[: max_num_frames].float()

        selected_num_frames = frames.size(0)
        remainder = (3 + selected_num_frames) % 4
        if remainder != 0:
            frames = frames[:-remainder]
        assert frames.size(0) % 4 == 1

        transform = T.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)
        frames = torch.stack(tuple(map(transform, frames)), dim=0)

        return frames.permute(0, 3, 1, 2).contiguous()


def sample_points_from_mask(mask: np.ndarray, n_in: int, n_out: int, device: torch.device) -> torch.Tensor:
    """Sample ``n_in`` foreground and ``n_out`` background points from ``mask``."""
    binm = mask > 0
    ys_in, xs_in = np.where(binm)
    ys_out, xs_out = np.where(~binm)
    in_pts = random.sample(list(zip(ys_in, xs_in)), min(len(ys_in), n_in)) if n_in else []
    out_pts = random.sample(list(zip(ys_out, xs_out)), min(len(ys_out), n_out)) if n_out else []
    pts = in_pts + out_pts
    arr = np.array([[0.0, x, y] for (y, x) in pts], dtype=np.float32)
    return torch.from_numpy(arr).to(device)


def sample_points_inside_mask(mask: np.ndarray, n_in: int, device: torch.device) -> torch.Tensor:
    """Sample ``n_in`` foreground points from ``mask`` and return them as a tensor."""
    binm = mask > 0
    ys_in, xs_in = np.where(binm)
    in_pts = random.sample(list(zip(ys_in, xs_in)), min(len(ys_in), n_in)) if n_in else []
    arr = np.array([[0.0, x, y] for (y, x) in in_pts], dtype=np.float32)
    return torch.from_numpy(arr).to(device)


def fill_and_drop(track: torch.Tensor, vis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill occluded points using nearest visible neighbor and drop invisible tracks."""
    filled = track.clone()
    N, F, _ = filled.shape

    for t in range(1, F):
        visible_idx = torch.where(vis[:, t])[0]
        if len(visible_idx):
            inv_idx = torch.where(~vis[:, t])[0]
            if len(inv_idx):
                ref_pos = filled[inv_idx, t - 1]
                dmat = torch.cdist(ref_pos, filled[visible_idx, t])
                nearest = torch.argmin(dmat, dim=1)
                filled[inv_idx, t] = filled[visible_idx[nearest], t]
        else:
            filled[:, t] = filled[:, t - 1]

    dropped = []
    for i in range(N):
        invis_frames = (~vis[i]).nonzero(as_tuple=False).flatten()
        if invis_frames.numel():
            first_inv = invis_frames[0].item()
            if first_inv + 1 < vis.shape[1] and not vis[i, first_inv + 1 :].any():
                dropped.append(i)

    return filled, torch.tensor(dropped, dtype=torch.long, device=track.device)


def compute_ftd(
    model,
    device: torch.device,
    reference: torch.Tensor,
    target: torch.Tensor,
    mask: np.ndarray,
    *,
    n_points: int,
    use_fg_mask_only: bool,
) -> float:
    """Return the Fréchet trajectory distance between ``reference`` and ``target`` videos."""
    ref = reference.unsqueeze(0).to(device)
    tgt = target.unsqueeze(0).to(device)

    _, _, _, H, W = ref.shape

    print("Using mask with shape:", mask.shape, "and", np.count_nonzero(mask), "foreground pixels.")

    if use_fg_mask_only:
        queries = sample_points_inside_mask(mask, n_points, device)
    else:
        fg_points = max(1, n_points // 2)
        bg_points = max(0, n_points - fg_points)
        queries = sample_points_from_mask(mask, fg_points, bg_points, device)

    if queries.size(0) == 0:
        raise ValueError("Mask sampling produced zero query points. Check the mask contents.")

    queries = queries.unsqueeze(0)

    tracks: list[torch.Tensor] = []
    dropped_indices: list[set[int]] = []
    for vid in (ref, tgt):
        vid = (vid * 0.5 + 0.5) * 255
        pts, vis = model(vid, queries=queries)
        pts, drop = fill_and_drop(pts[0], vis[0])
        pts[:, :, 0] = pts[:, :, 0] / W
        pts[:, :, 1] = pts[:, :, 1] / H
        tracks.append(pts)
        dropped_indices.append(set(int(idx) for idx in drop.cpu().tolist()))

    dists: list[float] = []
    for i in range(tracks[0].shape[1]):
        if i in dropped_indices[0] or i in dropped_indices[1]:
            continue

        fd = frdist(tracks[0][:, i, :].cpu().numpy(), tracks[1][:, i, :].cpu().numpy())
        if not np.isnan(fd):
            dists.append(fd**2)

    if not dists:
        raise RuntimeError("No valid trajectories were produced. Check masks or videos.")

    return float(np.sqrt(np.mean(dists)))


def load_mask(mask_dir: Path, width: int, height: int) -> np.ndarray:
    """Load the first mask image from ``mask_dir`` and resize it to match the video."""
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    mask_files = sorted(p for p in mask_dir.iterdir() if p.suffix.lower() in VALID_MASK_SUFFIXES)
    if not mask_files:
        raise FileNotFoundError(f"No mask images found in {mask_dir}")

    mask = Image.open(mask_files[0]).convert("L").resize((width, height))
    return np.array(mask, dtype=np.uint8)


def _select_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_cotracker(cotracker_root: Path | None, device: torch.device):
    if cotracker_root and cotracker_root.exists():
        model = torch.hub.load(str(cotracker_root), "cotracker3_offline", source="local")
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

    # Move the tracker to the same device as the video tensors to avoid device mismatches.
    return model.to(device).eval()


def compute_pair_ftd(
    reference_video: Path,
    target_video: Path,
    mask_dir: Path,
    *,
    width: int,
    height: int,
    max_frames: int,
    skip_start: int,
    skip_end: int,
    frame_sample_step: int | None,
    model,
    device: torch.device,
    n_points: int,
    use_fg_mask_only: bool,
) -> float:
    ref_count = get_frame_count(str(reference_video))
    tgt_count = get_frame_count(str(target_video))

    def remaining_frames(total: int) -> int:
        return max(0, total - skip_start - skip_end)

    available_frames = min(remaining_frames(ref_count), remaining_frames(tgt_count))
    if available_frames <= 0:
        raise ValueError("Insufficient frames after applying skip configuration.")

    max_num_frames = min(available_frames, max_frames)

    reference_frames = get_video_frames(
        str(reference_video),
        width=width,
        height=height,
        skip_frames_start=skip_start,
        skip_frames_end=skip_end,
        max_num_frames=max_num_frames,
        frame_sample_step=frame_sample_step,
    )
    target_frames = get_video_frames(
        str(target_video),
        width=width,
        height=height,
        skip_frames_start=skip_start,
        skip_frames_end=skip_end,
        max_num_frames=max_num_frames,
        frame_sample_step=frame_sample_step,
    )

    print(f"Reference frames shape: {reference_frames.shape}")
    print(f"Target frames shape: {target_frames.shape}")

    mask = load_mask(mask_dir, target_frames.shape[3], target_frames.shape[2])

    return compute_ftd(
        model,
        device,
        reference_frames,
        target_frames,
        mask,
        n_points=n_points,
        use_fg_mask_only=use_fg_mask_only,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute the FTD for a single video pair.")
    parser.add_argument("--reference_video", type=Path, help="Path to the reference (ground truth) video.")
    parser.add_argument("--target_video", type=Path, help="Path to the generated or target video.")
    parser.add_argument("--mask_path", type=Path, help="Directory containing mask images for the object.")
    parser.add_argument("--num_points", type=int, default=100, help="Number of query points used for tracking.")
    parser.add_argument("--use_foreground_only", action="store_true", help="Sample query points only from the foreground mask.")
    parser.add_argument("--width", type=int, default=832, help="Frame width used when decoding videos.")
    parser.add_argument("--height", type=int, default=480, help="Frame height used when decoding videos.")
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES, help="Maximum frames evaluated per video.")
    parser.add_argument("--skip_start", type=int, default=0, help="Frames skipped at the start of each video.")
    parser.add_argument("--skip_end", type=int, default=0, help="Frames skipped from the end of each video.")
    parser.add_argument("--frame_sample_step", type=int, default=None, help="Stride used when sampling frames.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for point sampling.")
    parser.add_argument("--device", type=str, default=None, help="Torch device string (defaults to auto-detect).")
    parser.add_argument(
        "--cotracker-root",
        type=Path,
        default=DEFAULT_COTRACKER_ROOT,
        help="Path to a local CoTracker repository clone. Falls back to the online repo if missing.",
    )
    args = parser.parse_args(args=argv)

    if args.num_points <= 0:
        parser.error("--num-points must be greater than zero.")
    if args.max_frames <= 0:
        parser.error("--max-frames must be greater than zero.")
    if args.frame_sample_step is not None and args.frame_sample_step <= 0:
        parser.error("--frame-sample-step must be greater than zero when provided.")

    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    device = _select_device(args.device)
    print("Using device:", device)

    model = _load_cotracker(args.cotracker_root.expanduser(), device)

    ftd = compute_pair_ftd(
        reference_video=args.reference_video.expanduser(),
        target_video=args.target_video.expanduser(),
        mask_dir=args.mask_path.expanduser(),
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
        skip_start=args.skip_start,
        skip_end=args.skip_end,
        frame_sample_step=args.frame_sample_step,
        model=model,
        device=device,
        n_points=args.num_points,
        use_fg_mask_only=args.use_foreground_only,
    )

    print(f"FTD: {ftd:.6f}")


if __name__ == "__main__":
    main()
