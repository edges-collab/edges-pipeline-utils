from pathlib import Path
from .runners import run_notebook, notebook_choices
import typer
import jupyter_client
from typing import Optional
from pygsdata import GSData
import h5py
from edges.const import KNOWN_TELESCOPES
from read_acq.gsdata import read_acq_to_gsdata, fast_lst_setter

k_manager = jupyter_client.kernelspec.KernelSpecManager()
avail_kernels = k_manager.find_kernel_specs()
here = Path(__file__).parent



app = typer.Typer()

@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def run(
    ctx: typer.Context, 
    notebook: notebook_choices,
    kernel: str,
    formats: list[str] = ("html",),
    ipynb: bool = True,
    nbout: Path = Path('.'),
    convert_args: str = "",
    cfgfile: Optional[Path] = None,
    basename: Optional[str] = None,
):
    kwargs = {}
    for i, arg in enumerate(ctx.args[::2]):
        kwargs[arg.replace("--","")] = ctx.args[2*i + 1]

    run_notebook(
        notebook=notebook,
        kernel=kernel,
        formats=formats,
        ipynb=ipynb,
        output_dir=nbout,
        convert_args=convert_args,
        cfgfile=cfgfile,
        basename=basename,
        **kwargs
    )


@app.command()
def get_ydays(
    first_year: int, 
    first_day: int, 
    last_year: int, 
    last_day: int,
    print_index: bool = False,
):
    from pathlib import Path
    from datetime import datetime as dt, timedelta

    datadir = Path("/data5/edges/data/2014_February_Boolardy/mro/low/")
    
    first = dt(year=first_year, month=1, day=1) + timedelta(days=first_day-1)
    last = dt(year=last_year, month=1, day=1) + timedelta(days=last_day-1)
    
    day=first
    index = 0
    while day <= last:
        tt = day.timetuple()
        y,d = tt.tm_year, tt.tm_yday
        files_to_load = sorted((datadir / str(y)).glob(f"{y}_{d:>03}_*.acq"))
        if files_to_load:
            if print_index:
                print(f"{index:03}: {y}-{d:03} [{', '.join(x.name for x in files_to_load)}]")
            else:
                print(f"{y}-{d:03}")
            index += 1
        day += timedelta(days=1)

@app.command()
def gather(files: list[Path], outfile: Path):
    """Gather all the individual .finalspec. files into one GSH5 file."""
    fldict = {}
    for fl in files:
        with h5py.File(fl, 'r') as _fl:
            fldict[fl] = _fl['metadata']['times'][0]
            
    day_files = sorted(files, key=lambda pth: fldict[pth])
    data = GSData.from_file(day_files, concat_axis='time')
    data.write_gsh5(outfile)
    
@app.command()
def convert(
    year: int,
    day: int,
    outdir: Path,
    telescope: str = 'edges-low-alan',
    lst_setter: Literal['fast', 'astropy'] = 'fast',
    
):
    """Convert ACQ files to GSH5 files, one per day.
    
    Note that the `telescope` and `lst_setter` options have defaults 
    matching Alan's pipeline.
    """
    obsname = f"{year}-{day:>03}"
    files_to_load = sorted((Path(datadir) / str(year)).glob(f"{year}_{day:03}_*.acq"))
    data = read_acq_to_gsdata(files_to_load, telescope=KNOWN_TELESCOPES[telescope], name=obsname, lst_setter=fast_lst_setter)
    
    outfile = outdir / f'{telescope}_lst{lst_setter}' / f"{obsname}.gsh5"
    if not outfile.exists():
        outfile.mkdir(parents=True, exists_ok=True)
        
    data.write_gsh5(outfile)
    print(f"Wrote {outfile}")
    
if __name__ == "__main__":
    app()
