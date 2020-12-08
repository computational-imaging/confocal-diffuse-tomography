import subprocess
import numpy as np
import os

processes = set()
max_processes = 10
vals = [(us_init, n) for us_init in np.linspace(200, 300, 11) for n in np.linspace(1.01, 1.23, 23)]

for val in vals:
    us_init = val[0]
    n = val[1]
    processes.add(subprocess.Popen(["python", "optimize.py", "--experiment_name", f"us_{us_init}_n_{n:.03f}",
                                    "--n_init", str(n), "--us_init", str(us_init)]))

    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update(
            [p for p in processes if p.poll() is not None])

# Check if all the child processes were closed
for p in processes:
    if p.poll() is None:
        p.wait()
