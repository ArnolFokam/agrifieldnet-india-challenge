import os
import subprocess
from typing import List, Union

import fire
from dotenv import load_dotenv

from aic.helpers import get_dir, generate_random_string


load_dotenv()

ROOT_DIR = os.getenv('ROOT_DIR')
CONDA_ENV_NAME = os.getenv('CONDA_ENV_NAME')
CONDA_HOME = os.getenv('CONDA_HOME')
SLURM_LOG_DIR = os.getenv('SLURM_LOG_DIR')
SLURM_DIR = os.getenv('SLURM_DIR')

scripts = {
    "train": "main_training",
}


def main(
        experiment: str,
        yaml_sweep_file: str,
        exclude: Union[str, List[str]] = None,
        include: Union[str, List[str]] = None,
        partition_name: str = 'stampede',
        max_runs: int = 20,
        use_slurm: bool = True,
):
    """This creates a slurm file and runs it
        Args:
            partition_name (str): Partition to run the code on
            experiment (str): experiment to run
            exclude (str): nodes to exclude
            include (str): nodes to include
            yaml_sweep_file (str): file path containing the parameter to sweep through
            max_runs (int): maximum number of python call torun an experiment bet bash file ran
            use_slurm (bool): is this script submitted to slurm. If yes? add '_slurm' ssuffix to prevent commit and
                              call 'sbatch' else call 'bash'
        """
        
    
    idx = generate_random_string()
    
    # Create Slurm File
    def get_bash_text(bsh_cmd):
        return f'''#!/bin/bash
#SBATCH -p {partition_name}
#SBATCH -N 1
#SBATCH -t 72:00:00
{f"#SBATCH -x {exclude if isinstance(exclude, str) else ','.join(exclude)}" if isinstance(exclude, (str, list, tuple)) else ""} 
{f"#SBATCH -w {include if isinstance(include, str) else ','.join(include)}" if isinstance(include, (str, list, tuple)) else ""}
#SBATCH -J {experiment}
#SBATCH -o {get_dir(f"{ROOT_DIR}/{SLURM_LOG_DIR}", experiment, "outputs_slurm")}/{idx}.%N.%j.out
#SBATCH -e {get_dir(f"{ROOT_DIR}/{SLURM_LOG_DIR}", experiment, "errors_slurm")}/{idx}.%N.%j.err
{f"source ~/.bashrc && conda activate {CONDA_ENV_NAME}" if  use_slurm else ""}
cd {ROOT_DIR}
{bsh_cmd}
{"conda deactivate" if  use_slurm else ""}
'''

    directory = get_dir(f"{ROOT_DIR}/{SLURM_DIR}", experiment)
    suffix = "_slurm" if use_slurm else ""
    cmd = f"python scripts/{scripts[experiment]}.py --sweep_path {yaml_sweep_file} --sweep_count {max_runs}"
    fpath = os.path.join(directory, f'{idx}{suffix}.bash')
    with open(fpath, 'w+') as f:
        f.write(get_bash_text(cmd))

    # Run it
    if use_slurm:
        ans = subprocess.call(f'sbatch {fpath}'.split(" "))
    else:
        ans = subprocess.call(f"""
        source {CONDA_HOME}/etc/profile.d/conda.sh
        conda activate {CONDA_ENV_NAME}
        bash {fpath}
        """, shell=True, executable='/bin/bash')
    assert ans == 0
    print(f"Successfully called {fpath}")


if __name__ == "__main__":
    fire.Fire(main)