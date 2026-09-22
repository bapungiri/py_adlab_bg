# server

Scripts and SLURM submission files meant to run on the compute server
(10.36.17.152), written/edited here so they're visible and reviewable in
this repo before anything executes remotely.

Workflow:
1. Write/edit a `.py` fitting script and matching `.slurm` submission file here.
2. Commit and push.
3. On the server: `cd ~/Codes/py_adlab_bg && git pull`, then `sbatch server/<job>.slurm`.
4. Results land in shared storage (`/mnt/pve/Homes/bapun/Data/results` or `GroupData`),
   readable from this machine over the existing SMB share without further server access.
