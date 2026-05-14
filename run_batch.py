import argparse
import csv
import json
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class Config:
    base_dir: Path = Path(
        r"C:\Users\mahes\OneDrive\Desktop\Projects\Projects_Personal\Summer Internship\Run_02"
    )
    project_file: str = "VNR_Flood_Model.prj"
    template_u_suffix: str = "02"
    output_dir: Path = Path(r"C:\Users\mahes\OneDrive\Desktop\Projects\dataset")


CFG = Config()


def model_stem() -> str:
    return Path(CFG.project_file).stem


def suffix_tag(number: int) -> str:
    return f"{number:02d}"


def u_file_for(suffix: str) -> Path:
    return CFG.base_dir / f"{model_stem()}.u{suffix}"


def project_path() -> Path:
    return CFG.base_dir / CFG.project_file


def validate_paths(template_suffix: str) -> None:
    required_paths = [
        project_path(),
        u_file_for(template_suffix),
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def get_scenario_library() -> dict:
    return {
        "light": [0, 5, 10, 15, 10, 5, 0],
        "moderate": [0, 20, 50, 80, 50, 20, 0],
        "extreme": [0, 50, 150, 300, 200, 100, 0],
        "burst": [0, 0, 200, 250, 100, 20, 0],
    }


def interpolate_pattern(values: Sequence[float], target_length: int) -> List[float]:
    if target_length <= 0:
        return []
    if len(values) == target_length:
        return list(values)
    if len(values) == 1:
        return [float(values[0])] * target_length
    src_x = np.linspace(0, len(values) - 1, len(values))
    dst_x = np.linspace(0, len(values) - 1, target_length)
    return [max(0.0, float(v)) for v in np.interp(dst_x, src_x, values)]


def generate_rain(num_points: int) -> tuple[List[float], str]:
    scenarios = get_scenario_library()
    scenario_name = random.choice(list(scenarios.keys()))
    base = interpolate_pattern(scenarios[scenario_name], num_points)
    noisy = [max(0.0, v + random.uniform(-8.0, 8.0)) for v in base]
    return noisy, scenario_name


def _is_hydro_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    for token in stripped.split():
        try:
            float(token)
        except ValueError:
            return False
    return True


def _looks_like_indexed_hydrograph(tokenized: Sequence[Sequence[str]]) -> bool:
    if not tokenized or any(len(parts) != 2 for parts in tokenized):
        return False

    first_col: List[float] = []
    for parts in tokenized:
        try:
            first_col.append(float(parts[0]))
        except ValueError:
            return False

    if len(first_col) < 2:
        return True

    return all(first_col[i + 1] > first_col[i] for i in range(len(first_col) - 1))


def _format_hydro_lines(rain_values: Sequence[float], template_lines: Sequence[str]) -> List[str]:
    tokenized = [line.strip().split() for line in template_lines]
    if not tokenized:
        raise RuntimeError("Hydrograph section is empty.")

    out_lines: List[str] = []

    if _looks_like_indexed_hydrograph(tokenized):
        final_rain = [max(0, int(round(v))) for v in interpolate_pattern(rain_values, len(tokenized))]
        for idx, parts in enumerate(tokenized):
            lead_ws_match = re.match(r"^\s*", template_lines[idx])
            lead_ws = lead_ws_match.group(0) if lead_ws_match else ""
            out_lines.append(f"{lead_ws}{parts[0]} {final_rain[idx]}\n")
        return out_lines

    token_counts = [len(parts) for parts in tokenized]
    total_values = sum(token_counts)
    final_values = [max(0, int(round(v))) for v in interpolate_pattern(rain_values, total_values)]

    cursor = 0
    for idx, count in enumerate(token_counts):
        lead_ws_match = re.match(r"^\s*", template_lines[idx])
        lead_ws = lead_ws_match.group(0) if lead_ws_match else ""
        chunk = final_values[cursor : cursor + count]
        cursor += count
        out_lines.append(f"{lead_ws}{' '.join(str(v) for v in chunk)}\n")

    return out_lines


def create_u_file_from_template(template_suffix: str, run_suffix: str, rain: Sequence[float]) -> Path:
    template_u = u_file_for(template_suffix)
    new_u = u_file_for(run_suffix)
    shutil.copy(template_u, new_u)

    lines = new_u.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)

    out: List[str] = []
    i = 0
    replaced = False
    flow_title = f"AutoFlow_{run_suffix}"

    while i < len(lines):
        line = lines[i]
        if line.startswith("Flow Title="):
            out.append(f"Flow Title={flow_title}\n")
        else:
            out.append(line)

        if "Precipitation Hydrograph=" not in line:
            i += 1
            continue

        i += 1
        hydro_lines: List[str] = []
        while i < len(lines) and _is_hydro_line(lines[i]):
            hydro_lines.append(lines[i])
            i += 1

        if not hydro_lines:
            raise ValueError("Template hydrograph section has no numeric values.")

        out.extend(_format_hydro_lines(rain, hydro_lines))
        replaced = True

    if not replaced:
        raise RuntimeError("No 'Precipitation Hydrograph=' block found in template flow file.")

    new_u.write_text("".join(out), encoding="utf-8")
    return new_u


def register_unsteady_file_in_project(run_suffix: str) -> None:
    prj = project_path()
    flow_tag = f"u{run_suffix}"

    lines = prj.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    if any(line.strip() == f"Unsteady File={flow_tag}" for line in lines):
        return

    insert_idx = next(
        (i for i, line in enumerate(lines) if line.startswith("Y Axis Title=")),
        len(lines),
    )
    updated = lines[:insert_idx] + [f"Unsteady File={flow_tag}\n"] + lines[insert_idx:]
    prj.write_text("".join(updated), encoding="utf-8")


def write_run_metadata(
    run_dir: Path,
    run_suffix: str,
    template_suffix: str,
    scenario: str,
    rain: Sequence[float],
    u_file: Path,
) -> None:
    payload = {
        "run_suffix": run_suffix,
        "template_suffix": template_suffix,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scenario": scenario,
        "rain_mm_per_timestep": [round(float(v), 3) for v in rain],
        "u_file": str(u_file),
        "flow_title": f"AutoFlow_{run_suffix}",
        "status": "generated",
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_manifest_row(
    manifest_file: Path,
    run_suffix: str,
    scenario: str,
    status: str,
    message: str,
) -> None:
    write_header = not manifest_file.exists()
    with manifest_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run_suffix", "scenario", "status", "message", "time"])
        writer.writerow([
            run_suffix,
            scenario,
            status,
            message,
            datetime.now().isoformat(timespec="seconds"),
        ])


def run_batch(num_runs: int, start_suffix: int, register_project: bool, template_suffix: str) -> None:
    validate_paths(template_suffix)
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = CFG.output_dir / "unsteady_generation_manifest.csv"

    print(f"Project: {project_path()}")
    print(f"Template flow: {u_file_for(template_suffix)}")
    print(f"Output dir: {CFG.output_dir}")

    for idx in range(num_runs):
        suffix = suffix_tag(start_suffix + idx)
        run_dir = CFG.output_dir / f"run_{suffix}"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            print(f"\n=== Generate u{suffix} ===")
            rain, scenario = generate_rain(num_points=24)
            u_file = create_u_file_from_template(template_suffix, suffix, rain)
            if register_project:
                register_unsteady_file_in_project(suffix)

            write_run_metadata(run_dir, suffix, template_suffix, scenario, rain, u_file)
            append_manifest_row(manifest, suffix, scenario, "ok", "generated")
            print(f"Generated: {u_file}")

        except Exception as exc:
            append_manifest_row(manifest, suffix, "unknown", "failed", str(exc))
            print(f"u{suffix} failed: {exc}")

    print("\nGeneration finished.")
    print(f"Manifest: {manifest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate only HEC-RAS unsteady flow (.uXX) files for manual simulation runs."
    )
    parser.add_argument(
        "--num-runs",
        "--num-rums",
        dest="num_runs",
        type=int,
        default=10,
        help="Number of unsteady flow files to generate.",
    )
    parser.add_argument(
        "--start-suffix",
        type=int,
        default=31,
        help="First suffix number (e.g., 31 -> u31).",
    )
    parser.add_argument(
        "--template-suffix",
        type=int,
        default=1,
        help="Template unsteady suffix to copy from (e.g., 1 uses u01).",
    )
    parser.add_argument(
        "--no-project-register",
        action="store_true",
        help="Do not append Unsteady File=uXX entries to the .prj file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.template_suffix < 0 or args.template_suffix > 99:
        raise ValueError("--template-suffix must be between 0 and 99")

    run_batch(
        num_runs=args.num_runs,
        start_suffix=args.start_suffix,
        register_project=not args.no_project_register,
        template_suffix=suffix_tag(args.template_suffix),
    )