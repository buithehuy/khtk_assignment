"""
Leaky Integrate-and-Fire (LIF) Neuron Model Implementation

The LIF model is a simplified neuron model that captures key neuronal dynamics:
- Passive membrane properties (leaky integrator)
- Threshold mechanism for action potentials
- Reset mechanism after spike

This model is widely used in computational neuroscience due to its simplicity
and tractability while maintaining biological relevance.

The subthreshold dynamics follow:
    C_m * dV/dt = -g_L * (V - E_L) + I_syn + I_ext

When V reaches threshold, a spike is generated and V is reset.

Reference: Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014). 
Neuronal dynamics: From single neurons to networks and models of cognition. 
Cambridge University Press.
"""

import numpy as np
import matplotlib.pyplot as plt


class LeakyIntegrateAndFire:
    """
    Leaky Integrate-and-Fire neuron model.
    
    This model implements the LIF neuron with threshold and reset mechanism.
    Input current can be provided as a time-varying function.
    """
    
    # Membrane time constant (ms)
    tau_m = 10.0
    
    # Membrane capacitance (µF/cm²)
    C_m = 1.0
    
    # Leak conductance (mS/cm²)
    g_L = 0.1
    
    # Leak reversal potential (mV)
    E_L = -70.0
    
    # Spike threshold (mV)
    V_threshold = -50.0
    
    # Reset potential (mV) - potential after spike
    V_reset = -70.0
    
    # Refractory period (ms)
    t_ref = 2.0
    
    def __init__(self, V0=-70.0):
        """
        Initialize the LIF neuron.
        
        Parameters
        ----------
        V0 : float
            Initial membrane potential (mV). Default is -70 mV (resting).
        """
        self.V0 = V0
        self.last_spike_time = -np.inf  # Time of last spike
    
    def input_current(self, t):
        """
        Generate input current as a function of time.
        
        Uses a square wave pattern:
        - 15 µA/cm² from 10-40 ms
        - 30 µA/cm² from 60-90 ms
        - 0 µA/cm² otherwise
        
        Parameters
        ----------
        t : float or array-like
            Time (ms)
            
        Returns
        -------
        float or array-like
            Input current (µA/cm²)
        """
        t_arr = np.atleast_1d(t)
        I = np.zeros_like(t_arr, dtype=float)
        
        # Square wave 1: 15 µA/cm² from 10-40 ms
        I[(t_arr >= 10) & (t_arr < 40)] = 15.0
        
        # Square wave 2: 30 µA/cm² from 60-90 ms
        I[(t_arr >= 60) & (t_arr < 90)] = 30.0
        
        return I if isinstance(t, np.ndarray) else I[0]
    
    def lif_derivative(self, V, I_ext, spike_occurred):
        """
        Calculate the derivative of membrane potential.
        
        Parameters
        ----------
        V : float
            Current membrane potential (mV)
        I_ext : float
            External input current (µA/cm²)
        spike_occurred : bool
            Whether a spike has just occurred
            
        Returns
        -------
        float
            Derivative dV/dt (mV/ms)
        """
        if spike_occurred:
            # After spike, return to reset potential immediately
            return 0.0
        
        # Standard LIF equation: dV/dt = [-(V - E_L) + R * I_ext] / tau_m
        # where R = 1/g_L is the membrane resistance
        R_m = 1.0 / self.g_L
        
        dV_dt = (-(V - self.E_L) + R_m * I_ext) / self.tau_m
        
        return dV_dt
    
    def simulate(self, t_span, dt=0.01):
        """
        Simulate the LIF neuron.
        
        Parameters
        ----------
        t_span : tuple
            Time span (t_start, t_end) in milliseconds
        dt : float
            Integration time step (ms). Default is 0.01 ms.
            
        Returns
        -------
        tuple
            (t, V, spike_times)
            - t: Time array (ms)
            - V: Membrane potential (mV)
            - spike_times: Times of spike events (ms)
        """
        # Time vector
        t = np.arange(t_span[0], t_span[1], dt)
        n_steps = len(t)
        
        # Initialize state variables
        V = np.zeros(n_steps)
        V[0] = self.V0
        
        spike_times = []
        in_refractory = False
        
        # Integrate using Euler method
        for i in range(n_steps - 1):
            current_time = t[i]
            V_current = V[i]
            
            # Check if in refractory period
            if current_time < (self.last_spike_time + self.t_ref):
                in_refractory = True
            else:
                in_refractory = False
            
            # Get input current
            I_ext = self.input_current(current_time)
            
            # Update membrane potential
            dV_dt = self.lif_derivative(V_current, I_ext, in_refractory)
            V[i + 1] = V_current + dV_dt * dt
            
            # Check for threshold crossing
            if V[i + 1] >= self.V_threshold and not in_refractory:
                spike_times.append(current_time + dt)
                V[i + 1] = self.V_reset
                self.last_spike_time = current_time + dt
        
        return t, V, np.array(spike_times)
    
    def plot_results(self, t, V, spike_times, save_path=None):
        """
        Plot simulation results.
        
        Parameters
        ----------
        t : array-like
            Time array (ms)
        V : array-like
            Membrane potential (mV)
        spike_times : array-like
            Times of spike events (ms)
        save_path : str, optional
            Path to save the figure. If None, does not save.
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot membrane potential
        ax.plot(t, V, 'b-', linewidth=2, label='Membrane Potential V(t)')
        
        # Mark threshold
        ax.axhline(y=self.V_threshold, color='r', linestyle='--', 
                  linewidth=2, label=f'Threshold ({self.V_threshold} mV)')
        
        # Mark reset potential
        ax.axhline(y=self.V_reset, color='g', linestyle='--', 
                  linewidth=1.5, alpha=0.5, label=f'Reset ({self.V_reset} mV)')
        
        # Mark resting potential
        ax.axhline(y=self.E_L, color='orange', linestyle='--', 
                  linewidth=1.5, alpha=0.5, label=f'Resting ({self.E_L} mV)')
        
        # Mark spike times with red dots
        if len(spike_times) > 0:
            V_at_spikes = np.ones_like(spike_times) * self.V_threshold
            ax.plot(spike_times, V_at_spikes, 'r*', markersize=15, 
                   label=f'Spikes (n={len(spike_times)})')
        
        # Add shaded regions for input currents
        I_input = np.array([self.input_current(time) for time in t])
        ax.fill_between(t, -90, -80, where=(I_input > 0), alpha=0.2, color='gray',
                       label='Input Current On')
        
        ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Membrane Potential (mV)', fontsize=12, fontweight='bold')
        ax.set_title('Leaky Integrate-and-Fire Neuron Model', 
                    fontsize=14, fontweight='bold')
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(-95, 0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig, ax


def main():
    """Main function to run LIF simulation."""
    
    print("=" * 70)
    print("LEAKY INTEGRATE-AND-FIRE (LIF) NEURON MODEL SIMULATION")
    print("=" * 70)
    
    # Create LIF neuron
    lif = LeakyIntegrateAndFire(V0=-70.0)
    
    print(f"\nNeuron Parameters:")
    print(f"  Membrane time constant (τ_m): {lif.tau_m} ms")
    print(f"  Leak reversal potential (E_L): {lif.E_L} mV")
    print(f"  Spike threshold (V_th): {lif.V_threshold} mV")
    print(f"  Reset potential (V_reset): {lif.V_reset} mV")
    print(f"  Refractory period: {lif.t_ref} ms")
    print(f"  Initial potential: {lif.V0} mV")
    
    print(f"\nInput Protocol:")
    print(f"  Square wave 1: 15 µA/cm² from 10-40 ms")
    print(f"  Square wave 2: 30 µA/cm² from 60-90 ms")
    
    # Simulate for 100 ms
    print(f"\nSimulating for 100 ms...")
    t, V, spike_times = lif.simulate((0, 100), dt=0.01)
    
    print(f"Simulation complete!")
    print(f"\nResults:")
    print(f"  Number of spikes: {len(spike_times)}")
    if len(spike_times) > 0:
        print(f"  Spike times (ms): {np.array2string(spike_times, precision=2)}")
        print(f"  Mean inter-spike interval: {np.mean(np.diff(spike_times)):.2f} ms")
    print(f"  Minimum potential: {np.min(V):.2f} mV")
    print(f"  Maximum potential: {np.max(V):.2f} mV")
    
    # Plot results
    lif.plot_results(t, V, spike_times, save_path='lif_results.png')
    plt.show()


if __name__ == "__main__":
    main()
