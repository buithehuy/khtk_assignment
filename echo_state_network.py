"""
Echo State Network (Reservoir Computing) for Time Series Prediction

Echo State Networks are a type of recurrent neural network that perform
computation through a high-dimensional dynamical system (the reservoir).
The network dynamics are governed by a sparsely connected, randomly initialized
recurrent layer. Only the output weights are trained using linear regression.

Key advantages:
- Fast training (linear regression instead of backpropagation)
- Rich nonlinear dynamics from untrained reservoir
- Effective for temporal processing and time series prediction

This implementation demonstrates ESN for predicting near-future and far-future
values of the chaotic Mackey-Glass time series.

Reference: Jaeger, H. (2001). The "echo state" approach to analysing and 
training recurrent neural networks. GMD Technical Report 148.
"""

import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


def mackey_glass_series(n_samples=2000, tau=17, n=10, beta=0.2, gamma=0.1, dt=0.1):
    """
    Generate Mackey-Glass time series.
    
    The Mackey-Glass equation is a time-delayed differential equation that
    exhibits chaotic behavior. It's commonly used as a benchmark for evaluating
    time series prediction algorithms.
    
    The equation is:
        dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    tau : int
        Time delay parameter. Default is 17 (chaotic regime)
    n : float
        Nonlinearity parameter. Default is 10.
    beta : float
        Input scaling. Default is 0.2.
    gamma : float
        Decay rate. Default is 0.1.
    dt : float
        Integration time step. Default is 0.1.
        
    Returns
    -------
    np.ndarray
        Mackey-Glass time series
    """
    # Initial condition
    history_len = tau
    x = np.zeros(n_samples)
    x[:history_len] = 0.5
    
    for i in range(history_len, n_samples):
        x_tau = x[i - tau]
        dx_dt = beta * x_tau / (1.0 + x_tau ** n) - gamma * x[i - 1]
        x[i] = x[i - 1] + dt * dx_dt
    
    return x


class EchoStateNetwork:
    """
    Echo State Network (Reservoir Computer) implementation.
    
    The ESN consists of:
    1. Input layer: Projects input to reservoir
    2. Reservoir: Sparsely connected, randomly initialized recurrent network
    3. Output layer: Linear combination of reservoir states (trained via regression)
    """
    
    def __init__(self, input_dim=1, reservoir_dim=300, output_dim=1, 
                 spectral_radius=0.9, sparsity=0.95, input_scale=0.5, 
                 regularization=1e-6, seed=42):
        """
        Initialize the Echo State Network.
        
        Parameters
        ----------
        input_dim : int
            Dimension of input (default: 1 for univariate time series)
        reservoir_dim : int
            Number of neurons in reservoir (default: 300)
        output_dim : int
            Dimension of output (default: 1)
        spectral_radius : float
            Spectral radius of recurrent weight matrix (default: 0.9)
            Controls the stability and memory of the reservoir
        sparsity : float
            Sparsity of recurrent connections (default: 0.95)
            0.95 means 95% of connections are zero
        input_scale : float
            Scaling of input weights (default: 0.5)
        regularization : float
            L2 regularization coefficient for Ridge regression (default: 1e-6)
        seed : int
            Random seed for reproducibility
        """
        np.random.seed(seed)
        
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scale = input_scale
        self.regularization = regularization
        
        # Initialize input weights (from input to reservoir)
        self.W_in = np.random.randn(reservoir_dim, input_dim) * input_scale
        
        # Initialize recurrent weights (reservoir connections)
        W = np.random.randn(reservoir_dim, reservoir_dim)
        # Apply sparsity mask
        mask = np.random.binomial(1, 1 - sparsity, (reservoir_dim, reservoir_dim))
        W = W * mask
        
        # Scale by spectral radius
        eigenvalues = np.linalg.eigvals(W)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        W = W * (spectral_radius / max_eigenvalue)
        
        self.W = W
        
        # Output weights (trained via regression)
        self.W_out = None
        
        # Reservoir state
        self.x = np.zeros(reservoir_dim)
        
        # Training data storage
        self.X_train = None  # Reservoir states
        self.y_train = None  # Target outputs
    
    def step(self, u, activation='tanh'):
        """
        Perform one time step of the reservoir dynamics.
        
        Parameters
        ----------
        u : np.ndarray
            Input vector (shape: input_dim,)
        activation : str
            Activation function ('tanh' or 'relu')
            
        Returns
        -------
        np.ndarray
            Reservoir state after update
        """
        # Reservoir update: x(t+1) = f(W_in * u(t) + W * x(t))
        if activation == 'tanh':
            self.x = np.tanh(np.dot(self.W_in, u) + np.dot(self.W, self.x))
        elif activation == 'relu':
            self.x = np.maximum(0, np.dot(self.W_in, u) + np.dot(self.W, self.x))
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        return self.x.copy()
    
    def train(self, u_train, y_train, discard=100, activation='tanh'):
        """
        Train the output weights of the ESN.
        
        Parameters
        ----------
        u_train : np.ndarray
            Input training data (shape: n_samples, input_dim)
        y_train : np.ndarray
            Target training data (shape: n_samples, output_dim)
        discard : int
            Number of initial samples to discard (transient period)
        activation : str
            Activation function to use
        """
        n_samples = len(u_train)
        
        # Reset reservoir
        self.x = np.zeros(self.reservoir_dim)
        
        # Collect reservoir states
        X_train = np.zeros((n_samples - discard, self.reservoir_dim))
        
        # Run training data through reservoir
        for i in range(n_samples):
            self.step(u_train[i], activation=activation)
            if i >= discard:
                X_train[i - discard, :] = self.x
        
        # Train output layer using Ridge regression
        ridge = Ridge(alpha=self.regularization)
        ridge.fit(X_train, y_train[discard:])
        
        self.W_out = ridge.coef_.T  # Shape: (reservoir_dim, output_dim)
        self.bias = ridge.intercept_
        
        self.X_train = X_train
        self.y_train = y_train[discard:]
    
    def predict(self, u_test, activation='tanh'):
        """
        Predict using trained ESN.
        
        Parameters
        ----------
        u_test : np.ndarray
            Test input (shape: n_samples, input_dim)
        activation : str
            Activation function to use
            
        Returns
        -------
        np.ndarray
            Predictions (shape: n_samples, output_dim)
        """
        if self.W_out is None:
            raise ValueError("Model must be trained first!")
        
        n_samples = len(u_test)
        predictions = np.zeros((n_samples, self.output_dim))
        
        for i in range(n_samples):
            self.step(u_test[i], activation=activation)
            predictions[i] = np.dot(self.x, self.W_out) + self.bias
        
        return predictions
    
    def reset_state(self):
        """Reset reservoir state to zero."""
        self.x = np.zeros(self.reservoir_dim)


class EchoStateNetworkPredictor:
    """
    Wrapper class for ESN-based time series prediction.
    
    Handles training and testing for predicting near-future and far-future values.
    """
    
    def __init__(self, reservoir_dim=300, prediction_horizons=[1, 10, 100]):
        """
        Initialize the predictor.
        
        Parameters
        ----------
        reservoir_dim : int
            Number of neurons in the reservoir
        prediction_horizons : list
            List of prediction steps ahead to train on
        """
        self.reservoir_dim = reservoir_dim
        self.prediction_horizons = prediction_horizons
        self.esns = {}  # Dictionary to store ESN for each horizon
    
    def prepare_data(self, data, horizon):
        """
        Prepare training data for a specific prediction horizon.
        
        For horizon=10: predict x(t+10) from x(t)
        For horizon=100: predict x(t+100) from x(t)
        
        Parameters
        ----------
        data : np.ndarray
            Time series data
        horizon : int
            Prediction steps ahead
            
        Returns
        -------
        tuple
            (u_train, y_train) where u_train is input and y_train is target
        """
        n = len(data) - horizon
        u_train = data[:n].reshape(-1, 1)
        y_train = data[horizon:horizon + n].reshape(-1, 1)
        
        return u_train, y_train
    
    def train(self, data, train_ratio=0.8):
        """
        Train ESN models for all prediction horizons.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data
        train_ratio : float
            Fraction of data to use for training
        """
        n_train = int(len(data) * train_ratio)
        data_train = data[:n_train]
        
        for horizon in self.prediction_horizons:
            print(f"  Training ESN for horizon={horizon}...")
            
            # Prepare data
            u_train, y_train = self.prepare_data(data_train, horizon)
            
            # Create and train ESN
            esn = EchoStateNetwork(
                input_dim=1,
                reservoir_dim=self.reservoir_dim,
                output_dim=1,
                spectral_radius=0.9,
                sparsity=0.95,
                input_scale=0.5,
                regularization=1e-6
            )
            
            esn.train(u_train, y_train, discard=300, activation='tanh')
            self.esns[horizon] = esn
    
    def predict(self, data, train_ratio=0.8, horizon=None):
        """
        Generate predictions for specified horizon.
        
        Parameters
        ----------
        data : np.ndarray
            Full time series data
        train_ratio : float
            Fraction used for training
        horizon : int
            Prediction horizon. If None, uses all trained horizons.
            
        Returns
        -------
        dict
            Dictionary mapping horizon -> predictions
        """
        n_train = int(len(data) * train_ratio)
        data_test = data[n_train:]
        
        predictions = {}
        
        horizons = [horizon] if horizon is not None else self.prediction_horizons
        
        for h in horizons:
            if h not in self.esns:
                print(f"Warning: ESN for horizon={h} not trained!")
                continue
            
            # Prepare test data
            u_test = data_test[:-h].reshape(-1, 1) if h < len(data_test) else data_test.reshape(-1, 1)
            
            # Reset and predict
            esn = self.esns[h]
            esn.reset_state()
            pred = esn.predict(u_test, activation='tanh')
            
            predictions[h] = pred.flatten()
        
        return predictions


def plot_esn_results(data, train_ratio, predictions_dict, save_path=None):
    """
    Plot ESN prediction results.
    
    Parameters
    ----------
    data : np.ndarray
        Original time series
    train_ratio : float
        Training data ratio
    predictions_dict : dict
        Dictionary of {horizon: predictions}
    save_path : str, optional
        Path to save figure
    """
    n_train = int(len(data) * train_ratio)
    test_start = n_train
    
    # Adjust figure height based on number of subplots
    n_subplots = len(predictions_dict)
    fig, axes = plt.subplots(n_subplots, 1, 
                             figsize=(14, 3.5 * n_subplots))
    if len(predictions_dict) == 1:
        axes = [axes]
    
    for idx, (horizon, predictions) in enumerate(sorted(predictions_dict.items())):
        ax = axes[idx]
        
        # Ground truth
        ax.plot(data[test_start:], 'b-', linewidth=2, alpha=0.7, label='Ground Truth')
        
        # Predictions
        ax.plot(predictions, 'r--', linewidth=2, alpha=0.7, label='ESN Prediction')
        
        # Shaded region for training data
        ax.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(0, ax.get_ylim()[1] * 0.95, 'Train | Test', 
               fontsize=10, ha='left', va='top')
        
        # ax.set_xlabel('Time Steps (Test Set)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mackey-Glass Value', fontsize=11, fontweight='bold')
        ax.set_title(f'ESN Prediction: {horizon} Steps Ahead', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def calculate_metrics(y_true, y_pred):
    """
    Calculate prediction error metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    dict
        Dictionary of metrics (MSE, RMSE, MAE, NRMSE)
    """
    # Ensure same length
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Normalized RMSE
    y_std = np.std(y_true)
    if y_std > 0:
        nrmse = rmse / y_std
    else:
        nrmse = float('inf')
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'NRMSE': nrmse
    }


def main():
    """Main function to run ESN demonstration."""
    
    print("=" * 70)
    print("ECHO STATE NETWORK: MACKEY-GLASS TIME SERIES PREDICTION")
    print("=" * 70)
    
    # Generate Mackey-Glass time series
    print("\n1. Generating Mackey-Glass time series...")
    data = mackey_glass_series(n_samples=2000, tau=17)
    print(f"   Generated {len(data)} samples")
    print(f"   Min value: {np.min(data):.4f}, Max value: {np.max(data):.4f}")
    
    # Create predictor
    print("\n2. Creating Echo State Network...")
    predictor = EchoStateNetworkPredictor(
        reservoir_dim=300,
        prediction_horizons=[10, 100]
    )
    
    # Train ESN
    print("\n3. Training ESN models...")
    print("   (This may take a moment...)")
    predictor.train(data, train_ratio=0.8)
    
    # Generate predictions
    print("\n4. Generating predictions...")
    n_train = int(len(data) * 0.8)
    data_test = data[n_train:]
    
    predictions = {}
    metrics = {}
    
    for horizon in predictor.prediction_horizons:
        esn = predictor.esns[horizon]
        esn.reset_state()
        u_test = data_test[:-horizon].reshape(-1, 1)
        pred = esn.predict(u_test, activation='tanh').flatten()
        
        # Ground truth for comparison
        y_test = data[n_train + horizon:n_train + horizon + len(pred)]
        
        predictions[horizon] = pred
        metrics[horizon] = calculate_metrics(y_test, pred)
    
    # Print results
    print("\n5. Prediction Performance:")
    print("-" * 70)
    for horizon in sorted(metrics.keys()):
        m = metrics[horizon]
        print(f"\n   Horizon: {horizon} steps ahead")
        print(f"     MSE:   {m['MSE']:.6f}")
        print(f"     RMSE:  {m['RMSE']:.6f}")
        print(f"     MAE:   {m['MAE']:.6f}")
        print(f"     NRMSE: {m['NRMSE']:.6f}")
    
    print("\n" + "=" * 70)
    print("Key Observations:")
    print(f"  • Near-future (horizon=10) prediction is more accurate")
    print(f"  • Far-future (horizon=100) prediction shows degradation")
    print(f"  • This demonstrates the chaotic nature of Mackey-Glass series")
    print("=" * 70)
    
    # Plot results
    plot_esn_results(data, 0.8, predictions, save_path='esn_results.png')
    plt.show()


if __name__ == "__main__":
    main()
