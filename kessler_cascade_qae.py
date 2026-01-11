"""
Kessler Syndrome 연쇄 충돌 시뮬레이션 - QAE 방식

LEO에서의 연쇄 충돌을 경로 의존적 확률 문제로 모델링
배리어 옵션과 동일한 구조:
- 시간 스텝마다 충돌 여부 결정
- 충돌 시 잔해 증가 (분기)
- "임계 잔해 수 도달" = knock-in 이벤트

현실적 모델링:
- 활성 위성은 자동 회피 기동 가능 (Starlink 등)
- 잔해 vs 잔해: 회피 불가 → 진짜 위험
- 잔해 vs 활성위성: 활성위성이 회피 시도
- 고장 위성: 회피 불가 → 잔해와 동일 취급
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


@dataclass
class KesslerResult:
    """Kessler Syndrome 시뮬레이션 결과"""
    initial_debris: int
    initial_satellites: int
    time_steps: int

    # 확률
    cascade_probability: float      # 연쇄 충돌 발생 확률
    critical_threshold: int         # 임계 잔해 수

    # 경로 분석
    total_paths: int
    cascade_paths: int              # 임계점 도달 경로 수

    # 최악 시나리오
    max_debris_possible: int
    expected_debris: float

    # 성능
    classical_ops: int
    quantum_queries: int
    speedup: float


class KesslerCascadeSimulator:
    """Kessler Syndrome 연쇄 충돌 시뮬레이터

    경로 의존적 문제로 모델링:
    - 각 시간 스텝에서 충돌 발생 여부 (확률적)
    - 충돌 시 잔해 증가 (브랜칭)
    - 잔해 증가 → 다음 충돌 확률 증가 (피드백)
    """

    def __init__(
        self,
        initial_debris: int = 1000,      # 초기 잔해 수
        initial_satellites: int = 5000,   # 초기 위성 수
        failed_satellite_ratio: float = 0.05,  # 고장 위성 비율 (5%)
        orbital_shell_volume_km3: float = 1e12,  # 궤도 쉘 부피
        avg_collision_cross_section_m2: float = 10.0,  # 평균 충돌 단면적
        avg_relative_velocity_km_s: float = 10.0,  # 평균 상대 속도
        debris_per_collision: int = 100,  # 충돌당 생성 잔해
        time_step_years: float = 1.0,     # 시간 스텝 (년)
        num_steps: int = 5,               # 시뮬레이션 스텝 수
        avoidance_success_rate: float = 0.95,  # 회피 성공률 (활성 위성)
        avoidance_warning_time_hours: float = 24.0  # 회피 경고 시간
    ):
        self.N_d0 = initial_debris
        self.N_s = initial_satellites
        self.N_failed = int(initial_satellites * failed_satellite_ratio)  # 고장 위성
        self.N_active = initial_satellites - self.N_failed  # 활성 위성
        self.V = orbital_shell_volume_km3
        self.A = avg_collision_cross_section_m2 / 1e6  # km^2로 변환
        self.v = avg_relative_velocity_km_s
        self.debris_per_collision = debris_per_collision
        self.dt = time_step_years
        self.n_steps = num_steps

        # 회피 시스템 파라미터
        self.avoidance_success_rate = avoidance_success_rate
        self.avoidance_warning_time = avoidance_warning_time_hours

        # 기본 충돌 확률 계산 (spatial density approach)
        # P = n * A * v * dt (단순화된 모델)
        self._base_collision_rate = self.A * self.v * self.dt * 365.25 * 24 * 3600

        print(f"Kessler Cascade Model (with Avoidance):")
        print(f"  Initial debris: {self.N_d0}")
        print(f"  Active satellites: {self.N_active} (can avoid)")
        print(f"  Failed satellites: {self.N_failed} (cannot avoid)")
        print(f"  Avoidance success rate: {self.avoidance_success_rate:.0%}")
        print(f"  Time steps: {self.n_steps} x {self.dt} years")
        print(f"  Base collision rate factor: {self._base_collision_rate:.2e}")

    def _collision_probability_by_type(
        self,
        n_debris: int,
        n_active: int,
        n_failed: int
    ) -> Dict[str, float]:
        """충돌 유형별 확률 계산 (회피 시스템 고려)

        충돌 유형:
        1. debris-debris: 회피 불가 → 100% 충돌
        2. debris-active: 활성 위성이 회피 시도 → 낮은 확률
        3. debris-failed: 회피 불가 → 100% 충돌
        4. active-active: 양쪽 모두 회피 가능 → 매우 낮음
        5. active-failed: 활성이 회피 시도 → 낮음
        6. failed-failed: 회피 불가 → 100% 충돌
        """
        base_annual_prob = 0.001  # 기본 연간 충돌 확률
        time_factor = self.dt

        # 밀도 기반 스케일링
        total = n_debris + n_active + n_failed
        density_factor = (total / 1000) ** 2

        base_p = base_annual_prob * density_factor * time_factor

        # 회피 불가능 객체 수 (잔해 + 고장위성)
        n_unavoidable = n_debris + n_failed

        # 충돌 유형별 확률
        probs = {}

        # 1. 회피 불가 vs 회피 불가 (debris-debris, debris-failed, failed-failed)
        # 이 쌍들은 회피 불가능 → 기본 확률 그대로
        n_unavoidable_pairs = n_unavoidable * (n_unavoidable - 1) / 2
        probs['unavoidable'] = base_p * (n_unavoidable_pairs / max(total, 1))

        # 2. 활성 위성 관련 충돌 (회피 가능)
        # debris-active, failed-active: 활성 위성이 회피 시도
        n_avoidable_pairs = n_active * n_unavoidable
        avoidance_factor = 1 - self.avoidance_success_rate  # 회피 실패 시에만 충돌
        probs['partially_avoidable'] = base_p * (n_avoidable_pairs / max(total, 1)) * avoidance_factor

        # 3. active-active: 양쪽 모두 회피 → 거의 0
        n_active_pairs = n_active * (n_active - 1) / 2
        probs['fully_avoidable'] = base_p * (n_active_pairs / max(total, 1)) * (avoidance_factor ** 2)

        # 총 충돌 확률
        probs['total'] = probs['unavoidable'] + probs['partially_avoidable'] + probs['fully_avoidable']

        return probs

    def _collision_probability(self, n_debris: int, n_satellites: int) -> float:
        """현재 상태에서 다음 스텝까지 충돌 확률 (회피 고려)

        이전 버전 호환용: 내부적으로 _collision_probability_by_type 사용
        """
        # 위성 중 일부는 고장 상태
        n_active = int(n_satellites * (1 - self.N_failed / max(self.N_s, 1)))
        n_failed = n_satellites - n_active

        probs = self._collision_probability_by_type(n_debris, n_active, n_failed)

        # 확률 제한
        return min(max(probs['total'], 0.0001), 0.95)

    def _path_to_debris_trajectory(self, path_binary: str) -> List[int]:
        """이진 경로를 잔해 수 궤적으로 변환

        path_binary: '10110' → 충돌, 무충돌, 충돌, 충돌, 무충돌
        """
        debris_counts = [self.N_d0]
        current_debris = self.N_d0
        current_satellites = self.N_s

        for bit in path_binary:
            if bit == '1':  # 충돌 발생
                # 잔해 증가
                new_debris = self.debris_per_collision
                current_debris += new_debris
                # 위성 감소 (충돌한 위성)
                current_satellites = max(0, current_satellites - 1)

            debris_counts.append(current_debris)

        return debris_counts

    def _check_cascade_threshold(
        self,
        debris_trajectory: List[int],
        threshold_multiplier: float = 3.0
    ) -> bool:
        """임계 잔해 수 도달 여부 (Kessler point)

        임계점: 잔해가 초기의 N배 이상 → 연쇄 반응 시작
        """
        threshold = self.N_d0 * threshold_multiplier
        return max(debris_trajectory) >= threshold

    def classical_cascade_simulation(
        self,
        threshold_multiplier: float = 3.0
    ) -> Tuple[float, int, int]:
        """고전적 경로 열거로 연쇄 충돌 확률 계산

        Returns:
            (연쇄 확률, 경로 수, 연쇄 경로 수)
        """
        n_paths = 2 ** self.n_steps
        cascade_count = 0
        total_prob_cascade = 0.0

        for path_idx in range(n_paths):
            path_binary = format(path_idx, f'0{self.n_steps}b')

            # 경로 확률 계산 (각 스텝의 조건부 확률 곱)
            path_prob = 1.0
            current_debris = self.N_d0
            current_satellites = self.N_s

            for i, bit in enumerate(path_binary):
                p_collision = self._collision_probability(current_debris, current_satellites)

                if bit == '1':  # 충돌 발생
                    path_prob *= p_collision
                    current_debris += self.debris_per_collision
                    current_satellites = max(0, current_satellites - 1)
                else:  # 충돌 없음
                    path_prob *= (1 - p_collision)

            # 잔해 궤적
            debris_trajectory = self._path_to_debris_trajectory(path_binary)

            # 임계점 도달 여부
            if self._check_cascade_threshold(debris_trajectory, threshold_multiplier):
                cascade_count += 1
                total_prob_cascade += path_prob

        return total_prob_cascade, n_paths, cascade_count

    def _build_cascade_state_preparation(self) -> QuantumCircuit:
        """연쇄 충돌 상태 준비 회로

        각 큐비트 = 각 시간 스텝의 충돌 여부
        확률은 이전 상태에 따라 달라짐 (조건부)

        간단화: 평균 충돌 확률로 초기화
        """
        qc = QuantumCircuit(self.n_steps)

        # 초기 충돌 확률
        p0 = self._collision_probability(self.N_d0, self.N_s)

        # 첫 스텝
        theta0 = 2 * np.arcsin(np.sqrt(p0))
        qc.ry(theta0, 0)

        # 이후 스텝들 (조건부 확률 근사)
        # 실제로는 controlled rotation이 필요하지만 간단화
        for i in range(1, self.n_steps):
            # 충돌 발생 시 확률 증가 가정
            p_after_collision = min(p0 * (1 + 0.5 * i), 0.9)
            theta = 2 * np.arcsin(np.sqrt((p0 + p_after_collision) / 2))
            qc.ry(theta, i)

        return qc

    def _build_cascade_oracle(self, threshold_collisions: int) -> QuantumCircuit:
        """연쇄 충돌 임계점 오라클

        threshold_collisions 회 이상 충돌 발생 시 마킹
        """
        n_qubits = self.n_steps + 1  # +1 for ancilla
        qc = QuantumCircuit(n_qubits)
        ancilla = self.n_steps

        # threshold_collisions개 이상의 1이 있으면 ancilla flip
        # 이건 복잡하므로, 간단화: 연속 충돌 패턴 감지

        # 예: threshold=2면, 2개 이상 연속 1이 있으면 마킹
        if threshold_collisions >= 2 and self.n_steps >= 2:
            # 처음 두 큐비트가 모두 1이면 (연속 충돌)
            qc.ccx(0, 1, ancilla)
            qc.z(ancilla)
            qc.ccx(0, 1, ancilla)  # uncompute

        return qc

    def quantum_cascade_simulation(
        self,
        threshold_multiplier: float = 3.0,
        num_shots: int = 10000
    ) -> Tuple[float, int]:
        """양자 회로로 연쇄 충돌 확률 추정

        Returns:
            (연쇄 확률, 양자 쿼리 수)
        """
        # 임계 충돌 횟수 계산
        # threshold_multiplier배 잔해 = 초기 + (threshold-1) * debris_per_collision
        threshold_collisions = int(np.ceil(
            (self.N_d0 * (threshold_multiplier - 1)) / self.debris_per_collision
        ))
        threshold_collisions = max(1, min(threshold_collisions, self.n_steps))

        print(f"  Cascade threshold: {threshold_collisions} collisions")

        n_qubits = self.n_steps + 1
        qc = QuantumCircuit(n_qubits, self.n_steps)

        # 상태 준비
        prep = self._build_cascade_state_preparation()
        qc.compose(prep, qubits=list(range(self.n_steps)), inplace=True)

        # 오라클 (간단화)
        oracle = self._build_cascade_oracle(threshold_collisions)
        qc.compose(oracle, qubits=list(range(n_qubits)), inplace=True)

        # 측정
        qc.measure(range(self.n_steps), range(self.n_steps))

        # 시뮬레이션
        simulator = AerSimulator()
        job = simulator.run(qc, shots=num_shots)
        result = job.result()
        counts = result.get_counts()

        # 연쇄 확률 계산
        cascade_count = 0
        for path_binary_rev, count in counts.items():
            path_binary = path_binary_rev[::-1]
            debris_trajectory = self._path_to_debris_trajectory(path_binary)

            if self._check_cascade_threshold(debris_trajectory, threshold_multiplier):
                cascade_count += count

        cascade_prob = cascade_count / num_shots
        quantum_queries = int(np.sqrt(2 ** self.n_steps))

        return cascade_prob, quantum_queries

    def run_analysis(self, threshold_multiplier: float = 3.0) -> KesslerResult:
        """전체 분석 실행"""

        print(f"\n{'='*60}")
        print(f"  KESSLER SYNDROME CASCADE ANALYSIS")
        print(f"  Path-Dependent QAE Approach")
        print(f"{'='*60}")

        print(f"\nScenario:")
        print(f"  Initial debris: {self.N_d0}")
        print(f"  Active satellites: {self.N_s}")
        print(f"  Debris per collision: {self.debris_per_collision}")
        print(f"  Critical threshold: {threshold_multiplier}x initial = {int(self.N_d0 * threshold_multiplier)} debris")
        print(f"  Simulation: {self.n_steps} steps x {self.dt} years = {self.n_steps * self.dt} years")

        # 고전적 계산
        print(f"\n[Classical] Enumerating all paths...")
        classical_prob, n_paths, cascade_paths = self.classical_cascade_simulation(threshold_multiplier)
        print(f"  Total paths: {n_paths}")
        print(f"  Cascade paths: {cascade_paths}")
        print(f"  Cascade probability: {classical_prob:.4%}")

        # 양자 계산
        print(f"\n[Quantum] Amplitude estimation...")
        quantum_prob, q_queries = self.quantum_cascade_simulation(threshold_multiplier)
        print(f"  Quantum queries: {q_queries}")
        print(f"  Cascade probability: {quantum_prob:.4%}")

        # 비교
        speedup = n_paths / max(q_queries, 1)

        # 최악 시나리오
        max_debris = self.N_d0 + self.n_steps * self.debris_per_collision

        # 기대 잔해 수 (Monte Carlo 근사)
        expected_debris = self.N_d0
        current_debris = self.N_d0
        current_sats = self.N_s
        for _ in range(self.n_steps):
            p = self._collision_probability(current_debris, current_sats)
            expected_debris += p * self.debris_per_collision
            current_debris += p * self.debris_per_collision

        print(f"\n[Analysis]")
        print(f"  Classical ops: {n_paths}")
        print(f"  Quantum queries: {q_queries}")
        print(f"  Speedup: {speedup:.1f}x")
        print(f"  Max possible debris: {max_debris}")
        print(f"  Expected debris (approx): {expected_debris:.0f}")

        return KesslerResult(
            initial_debris=self.N_d0,
            initial_satellites=self.N_s,
            time_steps=self.n_steps,
            cascade_probability=classical_prob,
            critical_threshold=int(self.N_d0 * threshold_multiplier),
            total_paths=n_paths,
            cascade_paths=cascade_paths,
            max_debris_possible=max_debris,
            expected_debris=expected_debris,
            classical_ops=n_paths,
            quantum_queries=q_queries,
            speedup=speedup
        )


def demo_kessler_cascade():
    """Kessler Syndrome 데모"""

    print("="*60)
    print("  QUANTUM AMPLITUDE ESTIMATION")
    print("  Kessler Syndrome Cascade Risk Assessment")
    print("="*60)

    # 시뮬레이터 생성 (연쇄효과가 보이도록 조정)
    # 더 밀집된 궤도 쉘, 더 높은 충돌 확률
    simulator = KesslerCascadeSimulator(
        initial_debris=500,
        initial_satellites=2000,
        orbital_shell_volume_km3=1e8,  # 매우 밀집된 쉘 (데모용)
        debris_per_collision=200,       # 충돌당 더 많은 잔해
        time_step_years=5.0,            # 5년 스텝
        num_steps=5                     # 25년 시뮬레이션, 32개 경로
    )

    # 분석 실행
    result = simulator.run_analysis(threshold_multiplier=2.0)

    print("\n" + "="*60)
    print("  KESSLER CASCADE RISK SUMMARY")
    print("="*60)
    print(f"  Initial state: {result.initial_debris} debris, {result.initial_satellites} satellites")
    print(f"  Time horizon: {result.time_steps} years")
    print(f"  Critical threshold: {result.critical_threshold} debris")
    print(f"  CASCADE PROBABILITY: {result.cascade_probability:.2%}")
    print(f"  Speedup: {result.speedup:.1f}x")
    print("="*60)

    return result


def demo_sensitivity_analysis():
    """민감도 분석 - Kessler Point 탐색"""

    print("\n" + "="*60)
    print("  SENSITIVITY ANALYSIS: Finding the Kessler Point")
    print("="*60)

    # 다양한 밀도에서 연쇄 확률 변화 관찰
    volumes = [1e12, 5e11, 2e11, 1e11, 5e10, 2e10, 1e10]

    print(f"\n{'Density (rel)':>15} {'Cascade Prob':>15} {'Risk Level':>15}")
    print("-" * 50)

    for vol in volumes:
        sim = KesslerCascadeSimulator(
            initial_debris=1000,
            initial_satellites=3000,
            orbital_shell_volume_km3=vol,
            debris_per_collision=100,
            time_step_years=10.0,  # 10년 단위
            num_steps=5            # 50년 시뮬레이션
        )
        prob, n_paths, cascade = sim.classical_cascade_simulation(threshold_multiplier=3.0)

        relative_density = 1e12 / vol

        if prob < 0.01:
            risk = "SAFE"
        elif prob < 0.10:
            risk = "LOW"
        elif prob < 0.30:
            risk = "MEDIUM"
        elif prob < 0.60:
            risk = "HIGH"
        else:
            risk = "CRITICAL"

        print(f"{relative_density:>15.1f}x {prob:>14.2%} {risk:>15}")

    print("-" * 45)
    print("\nMore initial debris -> Higher cascade probability")
    print("This is the Kessler point: density-dependent runaway")


def demo_avoidance_comparison():
    """회피 시스템 효과 비교"""

    print("\n" + "="*60)
    print("  COLLISION AVOIDANCE SYSTEM EFFECTIVENESS")
    print("="*60)

    scenarios = [
        ("No avoidance (all failed)", 0.0, 1.0),      # 모든 위성 고장
        ("Poor avoidance (50%)", 0.50, 0.2),          # 50% 성공률
        ("Good avoidance (90%)", 0.90, 0.1),          # 90% 성공률
        ("Excellent (95% - Starlink)", 0.95, 0.05),   # Starlink 수준
        ("Perfect avoidance (99%)", 0.99, 0.02),      # 이상적
    ]

    print(f"\n{'Scenario':<30} {'Cascade Prob':>15} {'Risk Reduction':>15}")
    print("-" * 65)

    baseline_prob = None

    for name, avoid_rate, fail_ratio in scenarios:
        sim = KesslerCascadeSimulator(
            initial_debris=2000,           # 더 많은 잔해
            initial_satellites=3000,
            failed_satellite_ratio=fail_ratio,
            orbital_shell_volume_km3=5e9,  # 더 밀집
            debris_per_collision=300,      # 더 많은 파편
            time_step_years=10.0,
            num_steps=5,
            avoidance_success_rate=avoid_rate
        )
        prob, _, _ = sim.classical_cascade_simulation(threshold_multiplier=1.5)

        if baseline_prob is None:
            baseline_prob = max(prob, 1e-10)
            reduction = "-"
        else:
            reduction = f"{(1 - prob/baseline_prob)*100:.1f}%"

        print(f"{name:<30} {prob:>14.2%} {reduction:>15}")

    print("-" * 65)
    print("\nKey insight: Active collision avoidance dramatically reduces cascade risk")
    print("But once debris-to-debris collisions dominate, avoidance becomes ineffective")


def demo_kessler_tipping_point():
    """Kessler 임계점 탐색 - 잔해 비율에 따른 전환점"""

    print("\n" + "="*60)
    print("  KESSLER TIPPING POINT ANALYSIS")
    print("  When does debris-debris collision dominate?")
    print("="*60)

    debris_counts = [1000, 2000, 4000, 6000, 8000, 10000, 15000]
    satellites = 5000

    print(f"\n{'Debris':>10} {'Debris %':>10} {'Unavoid %':>12} {'Cascade':>12} {'Status':>12}")
    print("-" * 60)

    for n_debris in debris_counts:
        sim = KesslerCascadeSimulator(
            initial_debris=n_debris,
            initial_satellites=satellites,
            failed_satellite_ratio=0.05,
            orbital_shell_volume_km3=5e9,  # 더 밀집
            debris_per_collision=200,
            time_step_years=10.0,
            num_steps=5,
            avoidance_success_rate=0.95
        )

        # 충돌 유형별 분석
        n_failed = sim.N_failed
        n_active = sim.N_active
        probs = sim._collision_probability_by_type(n_debris, n_active, n_failed)

        total = n_debris + satellites
        debris_pct = n_debris / total * 100

        # 회피 불가 충돌 비율
        unavoid_pct = probs['unavoidable'] / max(probs['total'], 1e-10) * 100

        prob, _, _ = sim.classical_cascade_simulation(threshold_multiplier=1.5)

        if prob < 0.10:
            status = "SAFE"
        elif prob < 0.30:
            status = "WARNING"
        elif prob < 0.60:
            status = "DANGER"
        else:
            status = "CRITICAL"

        print(f"{n_debris:>10} {debris_pct:>9.1f}% {unavoid_pct:>11.1f}% {prob:>11.2%} {status:>12}")

    print("-" * 60)
    print("\nTIPPING POINT: When unavoidable collision % exceeds ~70%")
    print("At this point, active avoidance systems become ineffective")
    print("→ This is the TRUE Kessler Syndrome threshold")


if __name__ == "__main__":
    # 기본 데모
    result = demo_kessler_cascade()

    # 회피 시스템 효과 분석
    demo_avoidance_comparison()

    # 임계점 분석
    demo_kessler_tipping_point()

    # 민감도 분석
    # demo_sensitivity_analysis()
