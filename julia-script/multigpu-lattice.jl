using CUDA
using Printf
using Distributed

# Add worker processes if not already added
if nworkers() == 1
    addprocs(4)  # Add 4 worker processes, adjust as needed
end


# Load CUDA on all processes
@everywhere using CUDA

# Define a function to check CUDA functionality
@everywhere function check_cuda()
    if CUDA.functional()
        println("CUDA is functional on process $(myid())")
        # Try a simple CUDA operation
        x = CUDA.ones(10)
        CUDA.@cuda kernel(x)
        println("CUDA kernel launched successfully on process $(myid())")
    else
        println("CUDA is not functional on process $(myid())")
    end
end

# Define a simple CUDA kernel
@everywhere function kernel(x)
    i = threadIdx().x
    x[i] = 2 * x[i]
    return nothing
end

# Run the check on all worker processes
@sync begin
    for w in workers()
        @async remotecall_wait(check_cuda, w)
    end
end

println("All workers have performed CUDA checks.")


# Add worker processes for each GPU
num_gpus = length(CUDA.devices())
addprocs(num_gpus)

@everywhere begin
    using CUDA
    
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
        
        if idx <= size(spins, 1) && idy <= size(spins, 2)
            i = (idy - 1) * size(spins, 1) + idx
            
            # Update random state
            rand_states[i] = xorshift32(rand_states[i])
            rand_float = (rand_states[i] >> 1) * 2.3283064365386963e-10  # Convert to float between 0 and 1
            
            # Get neighboring spins (periodic boundary conditions)
            up = idy == 1 ? spins[end, idx] : spins[idy-1, idx]
            down = idy == size(spins, 2) ? spins[1, idx] : spins[idy+1, idx]
            left = idx == 1 ? spins[idy, end] : spins[idy, idx-1]
            right = idx == size(spins, 1) ? spins[idy, 1] : spins[idy, idx+1]
            
            # Calculate energy change
            dE = 2 * J * spins[idy, idx] * (up + down + left + right)
            
            # Metropolis acceptance criterion
            if dE <= 0 || exp(-dE / (kb * T)) > rand_float
                spins[idy, idx] *= -1
                energy_changes[idy, idx] = dE
            else
                energy_changes[idy, idx] = 0.0f0
            end
        end
        
        return nothing
    end

    # Function to run simulation on a single GPU
    function run_ising_simulation_gpu(gpu_id, sub_L)
        CUDA.device!(gpu_id)
        
        # Initialize spins randomly
        spins = CUDA.rand(Int8, sub_L, sub_L)
        spins = 2 .* spins .- 1
        
        # Initialize energy changes array
        energy_changes = CUDA.zeros(Float32, sub_L, sub_L)
        
        # Initialize random number generator states
        rand_states = CUDA.rand(UInt32, sub_L * sub_L)
        
        # Calculate initial energy
        initial_energy = sum(J .* spins .* (circshift(spins, (0, 1)) .+ circshift(spins, (1, 0))))
        total_energy = [initial_energy]
        
        # Set up kernel launch configuration
        threads = (16, 16)
        blocks = (ceil(Int, sub_L/threads[1]), ceil(Int, sub_L/threads[2]))
        
        # Main simulation loop
        for step in 1:MC_STEPS
            CUDA.@cuda threads=threads blocks=blocks mc_step_kernel!(spins, energy_changes, rand_states)
            CUDA.synchronize()
            
            # Sum up energy changes
            total_energy[1] += sum(Array(energy_changes))
            
            if step % 10 == 0
#                @printf("GPU %d - Step %d: Energy per spin = %.6f\n", gpu_id, step, total_energy[1] / (sub_L * sub_L))
                 println("GPU ", gpu_id, " - Step ", step, ": Energy per spin = ", round(total_energy[1] / (sub_L * sub_L), digits=6))
            end
        end
        
        return Array(spins), total_energy[1]
    end
end

# Main function to coordinate multi-GPU simulation
function run_multi_gpu_simulation()
    num_gpus = length(CUDA.devices())
    sub_L = div(L, num_gpus)  # Divide the lattice among GPUs
    
    # Run simulations on all GPUs
    results = pmap(1:num_gpus) do gpu_id
        run_ising_simulation_gpu(gpu_id, sub_L)
    end
    
    # Combine results
    full_lattice = vcat([result[1] for result in results]...)
    total_energy = sum([result[2] for result in results])
    
    println("Multi-GPU simulation completed")
    println("Final Energy per spin = ", round(total_energy / (L * L), digits=6))
#   @printf("Final Energy per spin = %.6f\n", total_energy / (L * L))
    
    return full_lattice, total_energy
end

# Run the multi-GPU simulation
run_multi_gpu_simulation()
