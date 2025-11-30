# Carbon Deposition on Cu(111)

## Overview
Example of sequentially depositing C atoms onto a relaxed Cu(111) surface using GPUMD and NEP89 potential.

## Files
- relax/: NEP potential (`nep.txt`), input (`run.in`), relaxed substrate (`restart.xyz`), model.xyz
- create_deposit.py: driver script that adds C atoms each round and runs GPUMD
- deposit.py: lightweight GPUMD helper used by the script

## How to run
1) Relax the Cu(111) slab  
   cd relax && gpumd
   Produces `restart.xyz`.
2) Deposit carbon  
   python create_deposit.py  
   The script loops 20 rounds. Each round:
   - Loads previous `restart.xyz` (first round uses relax/restart.xyz)
   - Marks atoms below 5 Å as fixed group 0, the rest as group 1
   - Places 20 C atoms randomly over the surface at z=60 Å with velocity (0, 0, -0.01)
   - Runs GPUMD: 10,000 steps NVE (0.5 fs), then 50,000 steps NPT_SCR at 300 K (1 fs)
   - Writes outputs under `deposit/<round>/` (`dump.exyz`, `restart.xyz`, etc.)

## Adjustable parameters
- Fixed-layer thickness: `thickness` in create_deposit.py (default 5 Å)
- C atoms per round: 20 (edit in create_deposit.py)
- Injection height/velocity: z=60 Å, velocity=-0.01
- MD settings: edit the `run_in` list (time steps, ensemble, dump frequency, run length)
