import os
import subprocess
import numpy as np

regs = np.logspace(1.5, 2.5, 9)
# regs = [0.00001,0.01,0.1,1,10,100,1000,10000]

for reg in regs:
    print(f"\n{'='*50}")
    print(f"Starting evaluation with REG = {reg:.4g}") # scientific notation formatting
    print(f"{'='*50}\n")

    env = os.environ.copy()
    env["REG"] = str(reg)

    subprocess.run(
        ["python", "debug.py"],
        env=env
    )
