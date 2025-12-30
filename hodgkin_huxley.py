"""
Hodgkin-Huxley Neuron Model Implementation

This module implements the complete Hodgkin-Huxley equations that describe
the dynamics of action potentials in the squid giant axon.

The model uses voltage-dependent ion channels (sodium and potassium) and
a leak channel to simulate realistic neuronal membrane dynamics.

Reference: Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative 
description of membrane current and its application to conduction and 
excitation in nerve. Journal of Physiology, 117(4), 500-544.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class HodgkinHuxley:
    """
    Complete Hodgkin-Huxley neuron model implementation.
    
    Parameters are based on the squid giant axon at 6.3°C and are adjusted
    to biologically realistic values.
    """
    
    # Membrane capacitance (µF/cm²)
    C_m = 1.0
    
    # Leak conductance (mS/cm²)
    g_L = 0.3
    
    # Sodium conductance (mS/cm²)
    g_Na = 120.0
    
    # Potassium conductance (mS/cm²)
    g_K = 36.0
    
    # Reversal potentials (mV)
    E_L = -54.387  # Leak reversal potential
    E_Na = 60.0    # Sodium reversal potential
    E_K = -77.0    # Potassium reversal potential
    
    def __init__(self, I_ext=20.0, V0=-65.0):
        """
        Initialize the Hodgkin-Huxley model.
        
        Parameters
        ----------
        I_ext : float
            External input current (µA/cm²). Default is 20 µA/cm².
        V0 : float
            Initial membrane potential (mV). Default is -65 mV.
        """
        self.I_ext = I_ext
        self.V0 = V0
        
    def alpha_m(self, V):
        """
        Alpha rate constant for sodium activation gate (m).
        
        Determines the opening rate of sodium channels as a function of
        membrane potential.
        
        Parameters
        ----------
        V : float
            Membrane potential (mV)
            
        Returns
        -------
        float
            Alpha_m rate constant (1/ms)
        """
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    
    def beta_m(self, V):
        """
        Beta rate constant for sodium activation gate (m).
        
        Determines the closing rate of sodium channels.
        """
        return 4.0 * np.exp(-(V + 65.0) / 18.0)
    
    def alpha_h(self, V):
        """
        Alpha rate constant for sodium inactivation gate (h).
        
        Determines the rate of sodium channel inactivation (block).
        """
        return 0.07 * np.exp(-(V + 65.0) / 20.0)
    
    def beta_h(self, V):
        """
        Beta rate constant for sodium inactivation gate (h).
        
        Determines the rate of recovery from inactivation.
        """
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    def alpha_n(self, V):
        """
        Alpha rate constant for potassium activation gate (n).
        
        Determines the opening rate of potassium channels.
        """
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    
    def beta_n(self, V):
        """
        Beta rate constant for potassium activation gate (n).
        
        Determines the closing rate of potassium channels.
        """
        return 0.125 * np.exp(-(V + 65.0) / 80.0)
    
    def m_infinity(self, V):
        """
        Steady-state value of sodium activation (m) at given potential.
        """
        alpha = self.alpha_m(V)
        beta = self.beta_m(V)
        return alpha / (alpha + beta)
    
    def h_infinity(self, V):
        """
        Steady-state value of sodium inactivation (h) at given potential.
        """
        alpha = self.alpha_h(V)
        beta = self.beta_h(V)
        return alpha / (alpha + beta)
    
    def n_infinity(self, V):
        """
        Steady-state value of potassium activation (n) at given potential.
        """
        alpha = self.alpha_n(V)
        beta = self.beta_n(V)
        return alpha / (alpha + beta)
    
    def get_steady_state(self, V):
        """
        Get steady-state gating variables at resting potential.
        
        Returns
        -------
        tuple
            (m_init, h_init, n_init) steady-state gating variables
        """
        m0 = self.m_infinity(V)
        h0 = self.h_infinity(V)
        n0 = self.n_infinity(V)
        return m0, h0, n0
    
    def hodgkin_huxley_eqs(self, state, t):
        """
        Hodgkin-Huxley differential equations.
        
        The system consists of 4 coupled ODEs:
        - dV/dt: Membrane potential dynamics
        - dm/dt: Sodium activation gating variable
        - dh/dt: Sodium inactivation gating variable
        - dn/dt: Potassium activation gating variable
        
        Parameters
        ----------
        state : array-like
            Current state [V, m, h, n]
        t : float
            Current time (not used in equations)
            
        Returns
        -------
        list
            Time derivatives [dV/dt, dm/dt, dh/dt, dn/dt]
        """
        V, m, h, n = state
        
        # Currents
        I_Na = self.g_Na * (m ** 3) * h * (V - self.E_Na)
        I_K = self.g_K * (n ** 4) * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)
        
        # Membrane potential equation
        # C_m * dV/dt = I_ext - I_Na - I_K - I_L
        dV_dt = (self.I_ext - I_Na - I_K - I_L) / self.C_m
        
        # Gating variable equations
        # dx/dt = alpha_x(V) * (1 - x) - beta_x(V) * x
        
        alpha_m = self.alpha_m(V)
        beta_m = self.beta_m(V)
        dm_dt = alpha_m * (1 - m) - beta_m * m
        
        alpha_h = self.alpha_h(V)
        beta_h = self.beta_h(V)
        dh_dt = alpha_h * (1 - h) - beta_h * h
        
        alpha_n = self.alpha_n(V)
        beta_n = self.beta_n(V)
        dn_dt = alpha_n * (1 - n) - beta_n * n
        
        return [dV_dt, dm_dt, dh_dt, dn_dt]
    
    def simulate(self, t_span, dt=0.01):
        """
        Simulate the Hodgkin-Huxley model.
        
        Parameters
        ----------
        t_span : tuple
            Time span (t_start, t_end) in milliseconds
        dt : float
            Integration time step (ms). Default is 0.01 ms.
            
        Returns
        -------
        tuple
            (t, V, m, h, n, I_Na, I_K, I_L)
            - t: Time array (ms)
            - V: Membrane potential (mV)
            - m, h, n: Gating variables
            - I_Na: Sodium current (µA/cm²)
            - I_K: Potassium current (µA/cm²)
            - I_L: Leak current (µA/cm²)
        """
        # Time vector
        t = np.arange(t_span[0], t_span[1], dt)
        
        # Initial conditions
        m0, h0, n0 = self.get_steady_state(self.V0)
        initial_state = [self.V0, m0, h0, n0]
        
        # Integrate ODEs
        solution = odeint(self.hodgkin_huxley_eqs, initial_state, t)
        
        V = solution[:, 0]
        m = solution[:, 1]
        h = solution[:, 2]
        n = solution[:, 3]
        
        # Calculate currents
        I_Na = self.g_Na * (m ** 3) * h * (V - self.E_Na)
        I_K = self.g_K * (n ** 4) * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)
        
        return t, V, m, h, n, I_Na, I_K, I_L
    
    def plot_results(self, t, V, I_Na, I_K, save_path=None):
        """
        Plot simulation results.
        
        Parameters
        ----------
        t : array-like
            Time array (ms)
        V : array-like
            Membrane potential (mV)
        I_Na : array-like
            Sodium current (µA/cm²)
        I_K : array-like
            Potassium current (µA/cm²)
        save_path : str, optional
            Path to save the figure. If None, does not save.
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot membrane potential
        axes[0].plot(t, V, 'b-', linewidth=2, label='Action Potential')
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0].axhline(y=-70, color='r', linestyle='--', alpha=0.3, label='Resting Potential')
        axes[0].set_ylabel('Membrane Potential (mV)', fontsize=12, fontweight='bold')
        axes[0].set_title('Hodgkin-Huxley Neuron Model: Action Potential', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=10)
        
        # Plot sodium current
        axes[1].plot(t, I_Na, 'r-', linewidth=2, label='I_Na')
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1].set_ylabel('Sodium Current I_Na (µA/cm²)', fontsize=12, fontweight='bold')
        axes[1].set_title('Inward Sodium Current (Depolarization Phase)', 
                         fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10)
        
        # Plot potassium current
        axes[2].plot(t, I_K, 'g-', linewidth=2, label='I_K')
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[2].set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Potassium Current I_K (µA/cm²)', fontsize=12, fontweight='bold')
        axes[2].set_title('Outward Potassium Current (Repolarization Phase)', 
                         fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig, axes


def main():
    """Main function to run Hodgkin-Huxley simulation."""
    
    print("=" * 70)
    print("HODGKIN-HUXLEY NEURON MODEL SIMULATION")
    print("=" * 70)
    
    # Create model instance with 20 µA/cm² step input
    hh = HodgkinHuxley(I_ext=20.0, V0=-65.0)
    
    print(f"\nParameters:")
    print(f"  External input current: {hh.I_ext} µA/cm²")
    print(f"  Initial membrane potential: {hh.V0} mV")
    print(f"  Membrane capacitance: {hh.C_m} µF/cm²")
    print(f"  Sodium conductance (g_Na): {hh.g_Na} mS/cm²")
    print(f"  Potassium conductance (g_K): {hh.g_K} mS/cm²")
    print(f"  Leak conductance (g_L): {hh.g_L} mS/cm²")
    
    # Simulate for 100 ms
    print(f"\nSimulating for 100 ms...")
    t, V, m, h, n, I_Na, I_K, I_L = hh.simulate((0, 100), dt=0.01)
    
    print(f"Simulation complete!")
    print(f"\nResults:")
    print(f"  Maximum membrane potential: {np.max(V):.2f} mV")
    print(f"  Minimum membrane potential: {np.min(V):.2f} mV")
    print(f"  Peak sodium current: {np.max(I_Na):.2f} µA/cm²")
    print(f"  Peak potassium current: {np.max(I_K):.2f} µA/cm²")
    
    # Plot results
    hh.plot_results(t, V, I_Na, I_K, 
                   save_path='hodgkin_huxley_results.png')
    plt.show()


if __name__ == "__main__":
    main()
