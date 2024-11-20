import sys
import papermill as pm
from coolname import generate_slug

name = generate_slug(2)
print(name)
pm.execute_notebook(
    "patchscope_shifted_feature.ipynb",
    f"patchscope_shifted_feature_res_{name}.ipynb",
    log_output=True,
    stdout_file=sys.stdout,
    stderr_file=sys.stderr,
)
