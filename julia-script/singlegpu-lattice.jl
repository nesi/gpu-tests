using CUDA
using Printf

# Ising model parameters
const J = 1.0  # Interaction strength
const kb = 1.0 # Boltzmann constant
const T = 2.26 # Temperature (near critical point for 2D Ising model)

# Simulation parameters
const L = 16384  # Lattice size (16384 x 16384)
const MC_STEPS = 1000  # Number of Monte Carlo steps

# Xorshift random number generator
function xorshift32(state)
    state = xor(state, state << 13)
    state = xor(state, state >> 17)
    state = xor(state, state << 5)
    return state
end

# CUDA kernel for one Monte Carlo step
function mc_step_kernel!(spins, energy_changes, rand_states)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if idx <= L && idy <= L
        i = (idy - 1) * L + idx
        
        # Update random state
        rand_states[i] = xorshift32(rand_states[i])
        rand_float = (rand_states[i] >> 1) * 2.3283064365386963e-10  # Convert to float between 0 and 1
        
        # Get neighboring spins (periodic boundary conditions)
        up = idy == 1 ? spins[L*L-L+idx] : spins[i-L]
        down = idy == L ? spins[idx] : spins[i+L]
        left = idx == 1 ? spins[i+L-1] : spins[i-1]
        right = idx == L ? spins[i-L+1] : spins[i+1]
        
        # Calculate energy change
        dE = 2 * J * spins[i] * (up + down + left + right)
        
        # Metropolis acceptance criterion
        if dE <= 0 || exp(-dE / (kb * T)) > rand_float
            spins[i] *= -1
            energy_changes[i] = dE
        else
            energy_changes[i] = 0.0f0
        end
    end
    
    return nothing
end

# Main simulation function
function run_ising_simulation()
    # Initialize spins randomly
    spins = CUDA.rand(Int8, L, L)
    spins = 2 .* spins .- 1
    d_spins = CuArray(spins)
    
    # Initialize energy changes array
    energy_changes = CUDA.zeros(Float32, L, L)
    
    # Initialize random number generator states
    rand_states = CUDA.rand(UInt32, L, L)
    
    # Calculate initial energy
    initial_energy = sum(J .* d_spins .* (circshift(d_spins, (0, 1)) .+ circshift(d_spins, (1, 0))))
    total_energy = [initial_energy]
    
    # Set up kernel launch configuration
    threads = (16, 16)
    blocks = (ceil(Int, L/threads[1]), ceil(Int, L/threads[2]))
    
    # Main simulation loop
    for step in 1:MC_STEPS
        @cuda threads=threads blocks=blocks mc_step_kernel!(d_spins, energy_changes, rand_states)
        CUDA.synchronize()
        
        # Sum up energy changes on CPU
        total_energy[1] += sum(Array(energy_changes))
        
        if step % 10 == 0
            @printf("Step %d: Energy per spin = %.6f\n", step, total_energy[1] / (L * L))
        end
    end
end

# Run the simulation
run_ising_simulation()
