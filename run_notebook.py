import papermill as pm
from pathlib import Path
from time import time
from argparse import ArgumentParser
import os
import sys
from coolname import generate_slug

root = Path(__file__).parent
notebook_root = root / "notebooks"
if __name__ == "__main__":
    os.chdir(root)
    t = time()
    parser = ArgumentParser()
    parser.add_argument("--notebook", "-n", type=str, required=True)
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--model-path", type=str)
    parser.add_argument(
        "--remote",
        default=False,
        action="store_true",
        help="Use ndif remote execution. Check https://nnsight.net/status for available models",
    )
    parser.add_argument("--trust-remote-code", default=False, action="store_true")
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument(
        "--num-few-shot",
        "-nfs",
        type=int,
        default=5,
        help="Number of few shot examples",
    )
    parser.add_argument(
        "--use-tl",
        # "-utl",
        action="store_true",
        help="Use Transformer Lens",
    )
    parser.add_argument("--exp-id")
    args, unknown = parser.parse_known_args()
    kwargs = dict(vars(args))
    notebook = kwargs.pop("notebook")
    save_path = root / "results" / notebook
    save_path.mkdir(exist_ok=True, parents=True)
    source_notebook_path = notebook_root / f"{notebook}.ipynb"
    exp_id = str(int(time())) + "_" + (kwargs.get("exp_id", None) or generate_slug(2))
    target_notebook_path = save_path / (
        args.model.replace("/", "_") + f"_{exp_id}.ipynb"
    )
    kwargs["exp_id"] = exp_id
    print(f"Saving to {target_notebook_path}")
    kwargs["extra_args"] = unknown
    print(f"Running {notebook} with {kwargs}")

    try:
        pm.execute_notebook(
            source_notebook_path,
            target_notebook_path,
            parameters=kwargs,
            log_output=True,
            stdout_file=sys.stdout,
            stderr_file=sys.stderr,
        )
        print(f"Execution time: {time()-t:.2f}s")
    except (pm.PapermillExecutionError, KeyboardInterrupt) as e:
        print(e)
        if isinstance(e, pm.PapermillExecutionError):
            print("Error in notebook")
        delete = input(f"Delete notebook {target_notebook_path} ? (y/n)")
        if delete.lower() == "y":
            target_notebook_path.unlink()
        else:
            print(f"Notebook saved")
