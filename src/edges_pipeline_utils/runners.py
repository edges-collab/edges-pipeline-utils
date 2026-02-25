"""Notebook execution utilities for EDGES pipelines."""

import subprocess as sbp
from pathlib import Path

import jupyter_client
import papermill as pm
import toml
import yaml

k_manager = jupyter_client.kernelspec.KernelSpecManager()
avail_kernels = k_manager.find_kernel_specs()

here = Path(__file__).parent


def run_notebook(
    notebook: Path,
    kernel: str,
    formats: tuple[str] = (),
    ipynb: bool = True,
    output_dir: Path = Path(),
    convert_args: str = "",
    cfgfile: Path | None = None,
    basename: str | None = None,
    **kwargs,
) -> Path:
    """Execute a Jupyter notebook with papermill."""
    if basename is None:
        basename = notebook.stem

    if cfgfile is not None:
        if cfgfile.suffix == ".toml":
            params = toml.load(cfgfile)
        elif cfgfile.suffix == ".yaml":
            with open(cfgfile) as fl:
                params = yaml.safe_load(fl)
        else:
            raise ValueError(f"Unkown extension on --toml input: {cfgfile}")
        if params is None:
            params = {}
    else:
        params = {}

    infer = pm.inspect_notebook(notebook)
    tps = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "Path": Path,
        None: None,
    }

    new_kw = {}
    for k, v in kwargs.items():
        if k in infer:
            tp = infer[k]["inferred_type_name"]
            try:
                new_kw[k] = tps[tp](v)
            except Exception:
                new_kw[k] = v
        else:
            raise ValueError(f"Unknown parameter {k}")

    params |= new_kw

    output_path = Path(output_dir) / f"{basename}.ipynb"

    pm.execute_notebook(
        str(notebook),
        output_path=output_path,
        kernel_name=kernel,
        parameters=params,
    )

    for fmt in formats:
        sbp.run(
            [
                "jupyter",
                "nbconvert",
                "--output",
                f"{basename}.{fmt}",
                "--output-dir",
                str(output_path.parent),
                "--to",
                fmt,
                convert_args,
                str(output_path),
            ],
            check=True,
        )

    if not ipynb:
        output_path.unlink()

    return output_path
