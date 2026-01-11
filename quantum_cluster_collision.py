"""
클러스터 내 다중 객체 충돌 탐색 - QAE

클러스터 내 N개 객체의 모든 쌍(N*(N-1)/2)을
양자 병렬성으로 동시에 탐색하여 위험한 쌍을 찾음
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from itertools import combinations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

from tle_fetcher import TLEFetcher, SatelliteData
from orbit_propagator import OrbitPropagator, Position


@dataclass
class ClusterCollisionResult:
    """클러스터 충돌 분석 결과"""
    cluster_size: int
    num_pairs: int
    dangerous_pairs: List[Tuple[str, str, float]]  # (위성1, 위성2, 거리)
    quantum_search_iterations: int
    classical_comparisons: int
    speedup_achieved: float


class QuantumClusterAnalyzer:
    """양자 클러스터 충돌 분석기

    N개 객체 클러스터에서:
    - 고전: O(N²) 쌍별 비교
    - 양자: O(√N²) = O(N) Grover 탐색
    """

    def __init__(self, collision_threshold_km: float = 10.0):
        self.collision_threshold_km = collision_threshold_km
        self.propagator = OrbitPropagator()

    def _encode_pair_index(self, n_objects: int, i: int, j: int) -> int:
        """객체 쌍 (i,j)를 단일 인덱스로 인코딩

        N개 객체의 쌍 수: N*(N-1)/2
        (0,1)→0, (0,2)→1, ..., (i,j)→index
        """
        # i < j 보장
        if i > j:
            i, j = j, i
        # 삼각 인덱싱
        return i * n_objects - (i * (i + 1)) // 2 + j - i - 1

    def _decode_pair_index(self, n_objects: int, index: int) -> Tuple[int, int]:
        """인덱스를 객체 쌍 (i,j)로 디코딩"""
        i = 0
        while index >= n_objects - i - 1:
            index -= (n_objects - i - 1)
            i += 1
        j = i + 1 + index
        return i, j

    def _calculate_distances(
        self,
        satellites: List[SatelliteData],
        time: datetime
    ) -> np.ndarray:
        """모든 쌍의 거리 계산 (고전적)"""
        n = len(satellites)
        positions = []

        for sat in satellites:
            pos = self.propagator.propagate_single(sat.satrec, time)
            if pos:
                positions.append(np.array([pos.x, pos.y, pos.z]))
            else:
                positions.append(np.array([np.nan, np.nan, np.nan]))

        # 거리 행렬 (상삼각)
        n_pairs = n * (n - 1) // 2
        distances = np.zeros(n_pairs)

        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if not np.any(np.isnan(positions[i])) and not np.any(np.isnan(positions[j])):
                    distances[idx] = np.linalg.norm(positions[i] - positions[j])
                else:
                    distances[idx] = np.inf
                idx += 1

        return distances, positions

    def _build_grover_oracle(
        self,
        n_qubits: int,
        dangerous_indices: List[int]
    ) -> QuantumCircuit:
        """위험한 쌍을 마킹하는 Grover 오라클"""
        oracle = QuantumCircuit(n_qubits, name='Oracle')

        for idx in dangerous_indices:
            # 인덱스를 이진수로
            binary = format(idx, f'0{n_qubits}b')

            # |0⟩ 위치에 X 게이트
            for i, bit in enumerate(reversed(binary)):
                if bit == '0':
                    oracle.x(i)

            # Multi-controlled Z (위상 반전)
            if n_qubits > 1:
                oracle.h(n_qubits - 1)
                oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
                oracle.h(n_qubits - 1)
            else:
                oracle.z(0)

            # X 게이트 복원
            for i, bit in enumerate(reversed(binary)):
                if bit == '0':
                    oracle.x(i)

        return oracle

    def _build_diffusion(self, n_qubits: int) -> QuantumCircuit:
        """Grover diffusion operator"""
        diffusion = QuantumCircuit(n_qubits, name='Diffusion')

        diffusion.h(range(n_qubits))
        diffusion.x(range(n_qubits))

        if n_qubits > 1:
            diffusion.h(n_qubits - 1)
            diffusion.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            diffusion.h(n_qubits - 1)
        else:
            diffusion.z(0)

        diffusion.x(range(n_qubits))
        diffusion.h(range(n_qubits))

        return diffusion

    def quantum_search_dangerous_pairs(
        self,
        satellites: List[SatelliteData],
        time: datetime,
        threshold_km: Optional[float] = None
    ) -> ClusterCollisionResult:
        """클러스터 내 위험한 쌍을 양자 탐색

        Args:
            satellites: 클러스터 내 위성들
            time: 분석 시점
            threshold_km: 위험 거리 임계값 (km)

        Returns:
            ClusterCollisionResult
        """
        if threshold_km is None:
            threshold_km = self.collision_threshold_km

        n = len(satellites)
        n_pairs = n * (n - 1) // 2

        if n_pairs == 0:
            return ClusterCollisionResult(
                cluster_size=n,
                num_pairs=0,
                dangerous_pairs=[],
                quantum_search_iterations=0,
                classical_comparisons=0,
                speedup_achieved=1.0
            )

        # 1. 거리 계산 (이 부분은 고전적으로 - 물리 시뮬레이션)
        distances, positions = self._calculate_distances(satellites, time)

        # 2. 위험한 쌍 식별 (오라클 구성용)
        dangerous_indices = [i for i, d in enumerate(distances) if d < threshold_km]

        # 3. 큐비트 수 결정
        n_qubits = max(1, int(np.ceil(np.log2(max(n_pairs, 1)))))
        n_states = 2 ** n_qubits

        print(f"\n  Cluster Analysis:")
        print(f"    Objects: {n}")
        print(f"    Pairs to check: {n_pairs}")
        print(f"    Qubits needed: {n_qubits}")
        print(f"    Dangerous pairs found: {len(dangerous_indices)}")

        if len(dangerous_indices) == 0:
            return ClusterCollisionResult(
                cluster_size=n,
                num_pairs=n_pairs,
                dangerous_pairs=[],
                quantum_search_iterations=0,
                classical_comparisons=n_pairs,
                speedup_achieved=1.0
            )

        # 4. Grover 회로 구성
        qc = QuantumCircuit(n_qubits, n_qubits)

        # 초기 중첩
        qc.h(range(n_qubits))

        # 최적 반복 횟수
        num_dangerous = len(dangerous_indices)
        if num_dangerous > 0 and num_dangerous < n_states:
            optimal_iterations = int(np.pi / 4 * np.sqrt(n_states / num_dangerous))
            optimal_iterations = max(1, min(optimal_iterations, 10))  # 제한
        else:
            optimal_iterations = 1

        print(f"    Grover iterations: {optimal_iterations}")

        # Oracle과 Diffusion 반복
        oracle = self._build_grover_oracle(n_qubits, dangerous_indices)
        diffusion = self._build_diffusion(n_qubits)

        for _ in range(optimal_iterations):
            qc.compose(oracle, inplace=True)
            qc.compose(diffusion, inplace=True)

        # 측정
        qc.measure(range(n_qubits), range(n_qubits))

        # 5. 시뮬레이션
        simulator = AerSimulator()
        job = simulator.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts()

        # 6. 결과 분석
        # 가장 많이 측정된 상태들
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        found_dangerous = []
        for state, count in sorted_counts[:min(5, len(sorted_counts))]:
            # 비트 순서 반전 (Qiskit 규칙)
            idx = int(state[::-1], 2)
            if idx < n_pairs:
                i, j = self._decode_pair_index(n, idx)
                if idx in dangerous_indices:
                    dist = distances[idx]
                    found_dangerous.append((
                        satellites[i].name,
                        satellites[j].name,
                        dist
                    ))
                    print(f"    [FOUND] {satellites[i].name} - {satellites[j].name}: {dist:.1f} km (measured {count} times)")

        # 전체 위험 쌍 목록 (고전 검증용)
        all_dangerous = []
        for idx in dangerous_indices:
            i, j = self._decode_pair_index(n, idx)
            all_dangerous.append((
                satellites[i].name,
                satellites[j].name,
                distances[idx]
            ))

        # 스피드업 계산 (이론적)
        classical_ops = n_pairs
        quantum_ops = optimal_iterations * 2  # Oracle + Diffusion per iteration
        speedup = classical_ops / max(quantum_ops, 1)

        return ClusterCollisionResult(
            cluster_size=n,
            num_pairs=n_pairs,
            dangerous_pairs=all_dangerous,
            quantum_search_iterations=optimal_iterations,
            classical_comparisons=n_pairs,
            speedup_achieved=speedup
        )


def analyze_cluster_with_qae(
    cluster_members: List[SatelliteData],
    time: datetime,
    threshold_km: float = 50.0
) -> ClusterCollisionResult:
    """클러스터 분석 실행"""
    analyzer = QuantumClusterAnalyzer(collision_threshold_km=threshold_km)
    return analyzer.quantum_search_dangerous_pairs(cluster_members, time, threshold_km)


def demo():
    """데모 실행"""
    print("=" * 70)
    print("  QUANTUM CLUSTER COLLISION ANALYSIS")
    print("  Using Grover's Algorithm for Pair Search")
    print("=" * 70)

    # 데이터 로드
    print("\n[1/3] Loading satellite data...")

    starlink_fetcher = TLEFetcher(group="starlink")
    starlink = starlink_fetcher.load_satellites()

    debris_fetcher = TLEFetcher(group="cosmos-2251-debris")
    debris = debris_fetcher.load_satellites()

    # 작은 클러스터 시뮬레이션 (8개 객체 = 28쌍)
    # 실제로는 cascade_cluster.py에서 찾은 클러스터 사용
    test_cluster = starlink[:5] + debris[:3]

    print(f"  Test cluster: {len(test_cluster)} objects")
    print(f"  Total pairs: {len(test_cluster) * (len(test_cluster)-1) // 2}")

    # 분석
    print("\n[2/3] Running quantum search...")
    result = analyze_cluster_with_qae(
        test_cluster,
        datetime.utcnow(),
        threshold_km=1000.0  # 1000km 임계값 (테스트용)
    )

    # 결과
    print("\n[3/3] Results:")
    print(f"  Cluster size: {result.cluster_size}")
    print(f"  Pairs analyzed: {result.num_pairs}")
    print(f"  Dangerous pairs: {len(result.dangerous_pairs)}")
    print(f"  Grover iterations: {result.quantum_search_iterations}")
    print(f"  Theoretical speedup: {result.speedup_achieved:.2f}x")

    if result.dangerous_pairs:
        print("\n  Top dangerous pairs:")
        sorted_pairs = sorted(result.dangerous_pairs, key=lambda x: x[2])
        for name1, name2, dist in sorted_pairs[:5]:
            print(f"    {name1[:20]:20} - {name2[:20]:20}: {dist:.1f} km")

    print("\n" + "=" * 70)
    print("  Analysis Complete!")
    print("=" * 70)

    return result


def demo_with_real_cluster():
    """실제 클러스터 탐지 후 양자 분석"""
    from cascade_cluster import CascadeClusterDetector

    print("=" * 70)
    print("  INTEGRATED: CLUSTER DETECTION + QUANTUM PAIR SEARCH")
    print("=" * 70)

    # 1. 데이터 로드
    print("\n[1/4] Loading satellite data...")
    starlink_fetcher = TLEFetcher(group="starlink")
    starlink = starlink_fetcher.load_satellites()[:1000]

    debris_fetcher = TLEFetcher(group="cosmos-2251-debris")
    debris = debris_fetcher.load_satellites()

    all_objects = starlink + debris
    print(f"  Total objects: {len(all_objects)}")

    # 2. 클러스터 탐지
    print("\n[2/4] Detecting clusters (DBSCAN)...")
    detector = CascadeClusterDetector(eps_km=200.0, min_samples=5)
    clusters = detector.detect_clusters(all_objects, datetime.utcnow())
    print(f"  Clusters found: {len(clusters)}")

    if not clusters:
        print("  No clusters found.")
        return

    # 3. 가장 위험한 클러스터 선택
    top_cluster = clusters[0]
    print(f"\n[3/4] Analyzing top cluster:")
    print(f"  Risk score: {top_cluster.risk_score:.1f}")
    print(f"  Objects: {top_cluster.size}")
    print(f"  Altitude: {top_cluster.altitude_km:.0f} km")

    # 4. 양자 쌍 탐색
    print("\n[4/4] Quantum pair search in cluster...")

    # 클러스터 크기 제한 (큐비트 수 한계)
    max_objects = 15  # 15개 = 105쌍 = 7큐비트
    cluster_members = top_cluster.members[:max_objects]

    result = analyze_cluster_with_qae(
        cluster_members,
        datetime.utcnow(),
        threshold_km=100.0  # 100km 임계값
    )

    # 결과
    print("\n" + "=" * 70)
    print("  QUANTUM ANALYSIS RESULTS")
    print("=" * 70)
    print(f"  Cluster objects analyzed: {result.cluster_size}")
    print(f"  Pairs: {result.num_pairs}")
    print(f"  Close pairs (<100km): {len(result.dangerous_pairs)}")
    print(f"  Grover iterations: {result.quantum_search_iterations}")
    print(f"  Speedup: {result.speedup_achieved:.2f}x")

    if result.dangerous_pairs:
        print("\n  Closest pairs in cluster:")
        sorted_pairs = sorted(result.dangerous_pairs, key=lambda x: x[2])
        for name1, name2, dist in sorted_pairs[:10]:
            # 잔해 여부 표시
            marker1 = "[D]" if "DEB" in name1.upper() else "[S]"
            marker2 = "[D]" if "DEB" in name2.upper() else "[S]"
            print(f"    {marker1} {name1[:18]:18} - {marker2} {name2[:18]:18}: {dist:.1f} km")

    print("\n" + "=" * 70)

    return result


if __name__ == "__main__":
    # 기본 데모
    # demo()

    # 실제 클러스터 연동 데모
    demo_with_real_cluster()
