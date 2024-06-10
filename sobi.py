import numpy as np
from scipy.linalg import eigh, inv

class SOBI:
    """
    Second Order Blind Identification (SOBI) 알고리즘을 이용한 Blind Source Separation

    Args:
    n_sources (int): 추정할 소스의 수

    Attributes:
    mixing_ (numpy.ndarray): 추정된 혼합 행렬

    Methods:
    fit_transform(X): 주어진 혼합 신호를 이용하여 소스 신호를 추정하고 반환합니다.

    References:
    Belouchrani, A., Abed-Meraim, K., Cardoso, J.-F., & Moulines, E. (1997).
    A blind source separation technique using second-order statistics.
    IEEE Transactions on signal processing, 45(2), 434-444.
    """
    def __init__(self, n_sources):
        self.n_sources = n_sources
        self.mixing_ = None

    def fit_transform(self, X):
        T, N = X.shape
        M = np.mean(X, axis=1, keepdims=True)
        X_centered = X - M

        # Covariance matrix at zero lag
        R0 = np.dot(X_centered, X_centered.T) / N

        # Whitening
        D, E = eigh(R0)
        D = np.diag(D)
        Q = np.dot(E, np.sqrt(inv(D)).dot(E.T))

        X_whitened = np.dot(Q, X_centered)

        L = 100  # number of lags
        R = np.zeros((self.n_sources, self.n_sources, L))
        for k in range(L):
            X_lag = X_whitened[:, k:N]  # k 열부터 끝까지
            X_lead = X_whitened[:, :N - k]  # 시작부터 N-k 열까지
            R[:, :, k] = np.dot(X_lag, X_lead.T) / (N - k)

        """
        # Covariance matrices at multiple lags
        L = 100  # number of lags
        R = np.zeros((self.n_sources, self.n_sources, L))
        for k in range(L):
            R[:, :, k] = np.dot(X_whitened[:, k:], X_whitened[:, :-k - 1].T) / (N - k)
        """

        # Joint diagonalization
        B = Q
        for k in range(L):
            B = np.dot(B, self._joint_diag(R[:, :, k]))

        self.mixing_ = inv(B).dot(Q)
        S = np.dot(B.T, X_centered)
        return S

    def _joint_diag(self, A, eps=1e-6, max_iter=1000):
        """
        Approximate joint diagonalization of a set of matrices.
        """
        n = A.shape[0]  # 행렬 A의 크기 n을 가져옴
        B = np.eye(n)  # 초기 변환 행렬을 단위행렬로 설정
        for i in range(max_iter):
            for p in range(n - 1):
                for q in range(p + 1, n):
                    App = A[p, p]; Aqq = A[q, q]; Apq = A[p, q]
                    phi = 0.5 * np.arctan2(2 * Apq, Aqq - App)  # 회전 각도 계산
                    c = np.cos(phi)
                    s = np.sin(phi)
                    J = np.eye(n)
                    J[p, p] = c; J[q, q] = c; J[p, q] = s; J[q, p] = -s  # 회전 행렬 (Jacobi rotation)
                    A = J.T.dot(A).dot(J)
                    B = B.dot(J)
            off_diag = np.sum(np.abs(np.triu(A, 1)))
            if off_diag < eps:
                break
        return B
