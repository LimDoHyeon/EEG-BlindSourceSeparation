"""
This is blind source separation method using Independent Component Analysis (ICA).
It aims to separate EMG, EOG from EEG signals during spoken speech.
"""

import numpy as np
import mne
import sobi
import Utility.util as util


# SOBI를 이용한 혼합 행렬 추정 모듈  -> 검증 완료
def estimate_mixing_matrix(eeg_data):
    """
    SOBI 알고리즘을 사용하여 혼합 행렬을 추정합니다.

    Parameters:
    eeg_data (numpy.ndarray): EEG 데이터, shape (채널 수, 샘플 수)

    Returns:
    sources (numpy.ndarray): 추정된 소스 신호, shape (채널 수, 샘플 수)
    mixing_matrix (numpy.ndarray): 추정된 혼합 행렬, shape (채널 수, 채널 수)
    """
    sobi = SOBI(n_sources=eeg_data.shape[0])
    sources = sobi.fit_transform(eeg_data)
    mixing_matrix = sobi.mixing_

    return sources, mixing_matrix

# Sevcik's algorithm을 이용한 프랙탈 차원 계산 모듈 -> 일단 검증 완료
def calculate_fractal_dimension(signal):
    N = len(signal)  # 신호의 샘플 수
    x = np.arange(N)  # 파형의 샘플 인덱스
    y = signal  # 파형의 샘플 값

    # 프랙탈 차원 계산 위한 x와 y 변형
    x = x / np.max(x)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))

    # 각 점들 사이의 거리 계산 - 이렇게 계산해도 되는지는 모르겠음
    L = sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

    # 프랙탈 차원 계산
    fractal_dimension = 1 + np.log(L) / np.log(2 * (N - 1))
    return fractal_dimension

# 각 신호를 일정한 길이의 프레임으로 나누어 프랙탈 차원을 계산하고, 그 평균을 구함 -> 검증 완료
def calculate_mean_fractal_dimension(eeg_data, frame_length_ratio=0.1):
    fds = []
    frame_length = int(eeg_data.shape[1] * frame_length_ratio)
    for channel in range(eeg_data.shape[0]):
        channel_fds = []
        for start in range(0, eeg_data.shape[1], frame_length):
            end = start + frame_length
            if end > eeg_data.shape[1]:
                break
            frame = eeg_data[channel, start:end]
            fd = calculate_fractal_dimension(frame)
            channel_fds.append(fd)
        mean_fd = np.mean(channel_fds)
        fds.append(mean_fd)
    return np.array(fds)


# EOG 성분 식별 모듈 -> 검증 완료
def identify_eog_components(fractal_dimensions):
    sorted_indices = np.argsort(fractal_dimensions)  # 프랙탈 차원 값 기준으로 '인덱스'들을 오름차순 정렬
    sorted_fds = fractal_dimensions[sorted_indices]  # sorted_indices에서 정렬한 인덱스 번호 기준으로 실제 '값' 정렬
    k = 1  # Default to 1 if no suitable k is found
    for i in range(1, len(sorted_fds) // 2):
        if sorted_fds[i] - sorted_fds[i - 1] < sorted_fds[i + 1] - sorted_fds[i]:
            k = i
            break
    return sorted_indices[:k]  # Identify as EOG components s(1)(t),s(2)(t),...,s(k)(t)

# 공간 필터링 및 EEG 신호 복원 모듈 -> 검증 완료
def reconstruct_clean_eeg(eeg_data, mixing_matrix, eog_components):
    A_EEG = np.delete(mixing_matrix, eog_components, axis=1)  # EOG 성분을 제거한 혼합 행렬
    S_EEG = np.delete(eeg_data, eog_components, axis=0)  # EOG 성분을 제거한 소스 신호
    A_EEG_pinv = np.linalg.pinv(A_EEG)  # A_EEG의 무어-펜로스 의사역행렬을 계산
    clean_eeg = np.dot(A_EEG_pinv, S_EEG)  # 깨끗한 EEG 신호를 복원
    return clean_eeg


# 전체 프로세스 통합 모듈
def main(file_path):
    filtered_eeg_data= util.load_eeg_data_and_filtering(file_path)  # Data load
    sources, mixing_matrix = estimate_mixing_matrix(filtered_eeg_data)  # SOBI
    fractal_dimensions = calculate_mean_fractal_dimension(sources)  # Sevcik's algorithm
    eog_components = identify_eog_components(fractal_dimensions)  # EOG component identification
    clean_eeg = reconstruct_clean_eeg(filtered_eeg_data, mixing_matrix, eog_components)  # Clean EEG(pseudo-inverse)
    return clean_eeg, info  # info?

# 테스트용 메인 함수
if __name__ == "__main__":
    file_path = "your_directory.mff"
    clean_eeg, info = main(file_path)
    print("EEG 아티팩트 제거 완료")

    # mne.RawArray를 사용하여 결과 저장
    clean_raw = mne.io.RawArray(clean_eeg, info)
    clean_raw.save("bss_eeg.fif", overwrite=True)
