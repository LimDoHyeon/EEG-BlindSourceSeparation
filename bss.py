import numpy as np
import mne
from scipy.linalg import svd, pinv


def sobi(X, num_components=10, lag=1):
    """
    Simple SOBI(Second order blind Identification) implementation.
    X: Input signal matrix (channels x samples)
    num_components: Number of components to extract
    lag: Time lag for autocorrelation estimation
    """
    # Step 1: Estimate autocorrelation matrices
    p, n = X.shape
    R0 = np.dot(X, X.T) / n
    R1 = np.dot(X[:, :-lag], X[:, lag:].T) / (n - lag)

    # Step 2: Joint diagonalization
    U, S, V = svd(R0)
    K = np.dot(np.sqrt(pinv(S)), U.T)
    M = np.dot(K, R1).dot(K.T)
    U1, S1, V1 = svd(M)
    A = np.dot(U1.T, K)

    # Return the estimated mixing matrix and source signals
    S_est = np.dot(pinv(A), X)
    return A, S_est[:num_components, :]


def identify_eog_components(S, rate=250):
    """
    Identify EOG components based on frequency content.
    S: Source signals (components x samples)
    rate: Sampling rate
    """
    from scipy.signal import welch
    freqs, psd = welch(S, fs=rate, nperseg=512)
    # Simple criteria: EOG components often have higher power in low frequencies
    eog_indices = np.argmax(psd[:, freqs < 4], axis=1)
    return eog_indices


def remove_artifacts(X, eog_inds):
    """
    Remove identified artifacts from the signal.
    X: Original signal matrix (channels x samples)
    eog_inds: Indices of components identified as EOG artifacts
    """
    A, S_est = sobi(X)
    S_clean = np.array(S_est)
    S_clean[eog_inds, :] = 0  # Remove EOG components
    X_clean = np.dot(A, S_clean)  # Reconstruct the signal without EOG components
    return X_clean


"""
# Example usage:
# Load your EEG data into a NumPy array (channels x samples)
# For demonstration, let's create a random array
X = np.random.randn(21, 1000)  # Example EEG data with 21 channels and 1000 samples

# Apply SOBI and identify EOG components
A, S_est = sobi(X)
eog_inds = identify_eog_components(S_est)

# Remove EOG artifacts
X_clean = remove_artifacts(X, eog_inds)

# Note: This is a very simplified implementation and mainly for demonstration.
# For real applications, consider using specialized libraries like MNE-Python for EEG processing.
"""