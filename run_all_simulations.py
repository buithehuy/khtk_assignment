"""
Computational Neuroscience Assignment: Complete Project Runner

This script runs all three neural models and generates comprehensive results:
1. Hodgkin-Huxley neuron model with action potential simulation
2. Leaky Integrate-and-Fire neuron model with threshold dynamics
3. Echo State Network for time series prediction

Simply execute this script to run all simulations and generate plots.
"""

import sys
import os
import matplotlib.pyplot as plt

# Import all models
from hodgkin_huxley import HodgkinHuxley
from lif_model import LeakyIntegrateAndFire
from echo_state_network import (
    EchoStateNetworkPredictor, 
    mackey_glass_series,
    plot_esn_results,
    calculate_metrics
)


def run_all_simulations():
    """Run all three neural models."""
    
    print("\n" + "=" * 80)
    print(" " * 15 + "COMPUTATIONAL NEUROSCIENCE ASSIGNMENT")
    print(" " * 10 + "Multi-Model Neural Simulation and Prediction")
    print("=" * 80)
    
    # ========================================================================
    # 1. HODGKIN-HUXLEY MODEL
    # ========================================================================
    print("\n\n" + "▶" * 40)
    print("PART 1: HODGKIN-HUXLEY NEURON MODEL")
    print("▶" * 40 + "\n")
    
    print("Simulating the full Hodgkin-Huxley equations...")
    print("-" * 80)
    
    hh = HodgkinHuxley(I_ext=20.0, V0=-65.0)
    
    print(f"\nModel Configuration:")
    print(f"  • External input current: {hh.I_ext} µA/cm²")
    print(f"  • Initial membrane potential: {hh.V0} mV")
    print(f"  • Membrane capacitance: {hh.C_m} µF/cm²")
    print(f"  • Sodium conductance (g_Na): {hh.g_Na} mS/cm²")
    print(f"  • Potassium conductance (g_K): {hh.g_K} mS/cm²")
    print(f"  • Leak conductance (g_L): {hh.g_L} mS/cm²")
    
    # Simulate
    t, V, m, h, n, I_Na, I_K, I_L = hh.simulate((0, 100), dt=0.01)
    
    print(f"\nSimulation Results:")
    print(f"  ✓ Simulation complete (100 ms)")
    print(f"  • Peak membrane potential: {np.max(V):.2f} mV")
    print(f"  • Resting membrane potential: {np.min(V):.2f} mV")
    print(f"  • Peak inward Na+ current: {np.max(I_Na):.2f} µA/cm²")
    print(f"  • Peak outward K+ current: {np.max(I_K):.2f} µA/cm²")
    print(f"  • Action potential amplitude: {np.max(V) - np.min(V):.2f} mV")
    
    # Plot
    print(f"\n  Generating plots...")
    hh.plot_results(t, V, I_Na, I_K, save_path='01_hodgkin_huxley_results.png')
    
    print(f"  ✓ Plot saved: 01_hodgkin_huxley_results.png")
    
    # ========================================================================
    # 2. LEAKY INTEGRATE-AND-FIRE MODEL
    # ========================================================================
    print("\n\n" + "▶" * 40)
    print("PART 2: LEAKY INTEGRATE-AND-FIRE NEURON MODEL")
    print("▶" * 40 + "\n")
    
    print("Simulating the LIF neuron with threshold-reset dynamics...")
    print("-" * 80)
    
    lif = LeakyIntegrateAndFire(V0=-70.0)
    
    print(f"\nNeuron Configuration:")
    print(f"  • Membrane time constant: {lif.tau_m} ms")
    print(f"  • Leak reversal potential (E_L): {lif.E_L} mV")
    print(f"  • Spike threshold (V_th): {lif.V_threshold} mV")
    print(f"  • Reset potential (V_reset): {lif.V_reset} mV")
    print(f"  • Refractory period: {lif.t_ref} ms")
    
    print(f"\nInput Protocol:")
    print(f"  • Square wave 1: 15 µA/cm² from 10-40 ms")
    print(f"  • Square wave 2: 30 µA/cm² from 60-90 ms")
    
    # Simulate
    t_lif, V_lif, spike_times = lif.simulate((0, 100), dt=0.01)
    
    print(f"\nSimulation Results:")
    print(f"  ✓ Simulation complete (100 ms)")
    print(f"  • Number of spikes: {len(spike_times)}")
    if len(spike_times) > 0:
        print(f"  • Spike times (ms): {np.array2string(spike_times[:10], precision=2)}" + 
              ("..." if len(spike_times) > 10 else ""))
        if len(spike_times) > 1:
            print(f"  • Mean inter-spike interval: {np.mean(np.diff(spike_times)):.2f} ms")
    print(f"  • Membrane potential range: {np.min(V_lif):.2f} to {np.max(V_lif):.2f} mV")
    
    # Plot
    print(f"\n  Generating plots...")
    lif.plot_results(t_lif, V_lif, spike_times, save_path='02_lif_results.png')
    
    print(f"  ✓ Plot saved: 02_lif_results.png")
    
    # ========================================================================
    # 3. ECHO STATE NETWORK
    # ========================================================================
    print("\n\n" + "▶" * 40)
    print("PART 3: ECHO STATE NETWORK FOR TIME SERIES PREDICTION")
    print("▶" * 40 + "\n")
    
    print("Setting up Echo State Network for Mackey-Glass prediction...")
    print("-" * 80)
    
    # Generate Mackey-Glass
    print(f"\n1. Generating Mackey-Glass chaotic time series...")
    data = mackey_glass_series(n_samples=2000, tau=17)
    print(f"   ✓ Generated {len(data)} samples")
    print(f"   • Range: [{np.min(data):.4f}, {np.max(data):.4f}]")
    print(f"   • Mean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")
    
    # Create and train predictor
    print(f"\n2. Creating Echo State Network...")
    predictor = EchoStateNetworkPredictor(
        reservoir_dim=300,
        prediction_horizons=[10, 100]
    )
    print(f"   ✓ ESN created with 300 reservoir neurons")
    
    print(f"\n3. Training ESN models...")
    print(f"   (Training on 1600 samples, testing on 400 samples)")
    predictor.train(data, train_ratio=0.8)
    print(f"   ✓ Training complete")
    
    # Generate predictions
    print(f"\n4. Generating predictions on test set...")
    n_train = int(len(data) * 0.8)
    data_test = data[n_train:]
    
    predictions = {}
    metrics_dict = {}
    
    for horizon in predictor.prediction_horizons:
        esn = predictor.esns[horizon]
        esn.reset_state()
        u_test = data_test[:-horizon].reshape(-1, 1)
        pred = esn.predict(u_test, activation='tanh').flatten()
        
        # Ground truth
        y_test = data[n_train + horizon:n_train + horizon + len(pred)]
        
        predictions[horizon] = pred
        metrics_dict[horizon] = calculate_metrics(y_test, pred)
    
    print(f"   ✓ Predictions generated")
    
    # Results
    print(f"\n5. Prediction Performance:")
    print(f"   " + "-" * 76)
    for horizon in sorted(metrics_dict.keys()):
        m = metrics_dict[horizon]
        print(f"\n   ┌─ Horizon: {horizon} steps ahead")
        print(f"   │  RMSE:  {m['RMSE']:.6f}")
        print(f"   │  NRMSE: {m['NRMSE']:.4f}")
        print(f"   │  MAE:   {m['MAE']:.6f}")
        print(f"   └─ MSE:   {m['MSE']:.6f}")
    
    # Plot
    print(f"\n6. Generating plots...")
    plot_esn_results(data, 0.8, predictions, save_path='03_esn_results.png')
    print(f"   ✓ Plot saved: 03_esn_results.png")
    
    # Performance comparison
    print(f"\n7. Analysis:")
    print(f"   " + "-" * 76)
    near_rmse = metrics_dict[10]['RMSE']
    far_rmse = metrics_dict[100]['RMSE']
    degradation = (far_rmse - near_rmse) / near_rmse * 100
    
    print(f"\n   Prediction Accuracy Comparison:")
    print(f"   • Near-future (t+10):  RMSE = {near_rmse:.6f}")
    print(f"   • Far-future (t+100):  RMSE = {far_rmse:.6f}")
    print(f"   • Performance degradation: {degradation:.1f}%")
    print(f"\n   Interpretation:")
    print(f"   • The chaotic nature of Mackey-Glass limits prediction horizon")
    print(f"   • ESN captures short-term dynamics effectively (t+10)")
    print(f"   • Longer predictions (t+100) show higher uncertainty")
    print(f"   • This demonstrates the Lyapunov instability of chaotic systems")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("SIMULATION SUMMARY")
    print("=" * 80)
    
    print("\n✓ All simulations completed successfully!\n")
    
    print("Generated Files:")
    print("  1. 01_hodgkin_huxley_results.png  - Action potential dynamics")
    print("  2. 02_lif_results.png             - Spike train and threshold behavior")
    print("  3. 03_esn_results.png             - Time series predictions")
    
    print("\nProject Structure:")
    print("  • hodgkin_huxley.py      - HH model implementation")
    print("  • lif_model.py            - LIF model implementation")
    print("  • echo_state_network.py   - ESN implementation")
    print("  • run_all_simulations.py  - This runner script")
    print("  • README.md               - Complete documentation")
    
    print("\n" + "=" * 80)
    print("See README.md for detailed explanation of models and results.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Import numpy for calculations
    import numpy as np
    
    # Run all simulations
    run_all_simulations()
    
    # Show plots
    plt.show()
