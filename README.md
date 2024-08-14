# 
1. It sets up a 16384 x 16384 lattice of spins (either +1 or -1).
2. It performs 1000 Monte Carlo steps, where each step attempts to flip every spin in the lattice.
3. The simulation uses the Metropolis algorithm to decide whether to accept or reject each spin flip.
4. It calculates and reports the energy of the system every 10 steps.


The large lattice size and the number of Monte Carlo steps should ensure that this simulation runs for at least 5 minutes on an A100 GPU

<br>
<center><img src="single-gpu-a100-vs-p100.png" width="500" alt="Description of the image"></center>
