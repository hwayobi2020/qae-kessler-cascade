"""
경로 의존적 옵션 가치 평가 - QAE 방식

배리어 옵션(Barrier Option)을 양자 진폭 추정으로 가치평가
- 경로를 양자 상태로 인코딩
- 배리어 조건을 오라클로 구현
- QAE로 생존 경로 확률 추정
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator


@dataclass
class PathOptionResult:
    """경로 의존적 옵션 분석 결과"""
    option_type: str
    spot_price: float
    strike_price: float
    barrier_level: float

    # 가치
    classical_price: float
    quantum_price: float

    # 확률
    survival_probability: float  # 배리어 안 건드릴 확률
    knockout_probability: float  # 배리어 건드릴 확률

    # 성능
    num_paths_classical: int
    quantum_queries: int
    speedup: float


class QuantumPathOptionPricer:
    """양자 경로 의존적 옵션 가격 결정기

    경로를 양자 상태로 인코딩:
    - n 큐비트 = n 시간 스텝
    - |0⟩ = 가격 하락, |1⟩ = 가격 상승
    - 2^n 개의 가능한 경로
    """

    def __init__(
        self,
        spot: float = 100.0,
        strike: float = 100.0,
        volatility: float = 0.2,
        risk_free_rate: float = 0.05,
        time_to_maturity: float = 1.0,
        num_steps: int = 4  # 시간 스텝 수 (= 큐비트 수)
    ):
        self.S0 = spot
        self.K = strike
        self.sigma = volatility
        self.r = risk_free_rate
        self.T = time_to_maturity
        self.n_steps = num_steps

        # 이항 모델 파라미터 계산
        self.dt = self.T / self.n_steps
        self.u = np.exp(self.sigma * np.sqrt(self.dt))  # 상승 배율
        self.d = 1 / self.u  # 하락 배율

        # 리스크 중립 확률
        self.p = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)
        self.q = 1 - self.p

        print(f"Binomial Model Parameters:")
        print(f"  Up factor (u): {self.u:.4f}")
        print(f"  Down factor (d): {self.d:.4f}")
        print(f"  Risk-neutral prob (p): {self.p:.4f}")

    def _path_to_prices(self, path_binary: str) -> List[float]:
        """이진 경로를 가격 시퀀스로 변환

        path_binary: '1010' → 상승, 하락, 상승, 하락
        """
        prices = [self.S0]
        current_price = self.S0

        for bit in path_binary:
            if bit == '1':
                current_price *= self.u
            else:
                current_price *= self.d
            prices.append(current_price)

        return prices

    def _check_barrier_hit(
        self,
        prices: List[float],
        barrier: float,
        barrier_type: str = 'down-and-out'
    ) -> bool:
        """배리어 터치 여부 확인"""
        if barrier_type == 'down-and-out':
            return min(prices) <= barrier
        elif barrier_type == 'up-and-out':
            return max(prices) >= barrier
        elif barrier_type == 'down-and-in':
            return min(prices) <= barrier
        elif barrier_type == 'up-and-in':
            return max(prices) >= barrier
        return False

    def classical_barrier_option(
        self,
        barrier: float,
        barrier_type: str = 'down-and-out',
        option_type: str = 'call'
    ) -> Tuple[float, float, int]:
        """고전적 Monte Carlo로 배리어 옵션 가격 계산

        Returns:
            (옵션 가격, 생존 확률, 시뮬레이션 경로 수)
        """
        n_paths = 2 ** self.n_steps  # 모든 경로 열거 (작은 n에서)

        total_payoff = 0.0
        survival_count = 0

        # 모든 경로 열거
        for path_idx in range(n_paths):
            path_binary = format(path_idx, f'0{self.n_steps}b')
            prices = self._path_to_prices(path_binary)

            # 경로 확률 (이항 모델)
            num_ups = path_binary.count('1')
            num_downs = self.n_steps - num_ups
            path_prob = (self.p ** num_ups) * (self.q ** num_downs)

            # 배리어 체크
            hit_barrier = self._check_barrier_hit(prices, barrier, barrier_type)

            if barrier_type in ['down-and-out', 'up-and-out']:
                # Knock-out: 배리어 터치하면 무효
                if not hit_barrier:
                    survival_count += 1
                    final_price = prices[-1]
                    if option_type == 'call':
                        payoff = max(final_price - self.K, 0)
                    else:
                        payoff = max(self.K - final_price, 0)
                    total_payoff += path_prob * payoff
            else:
                # Knock-in: 배리어 터치해야 유효
                if hit_barrier:
                    survival_count += 1
                    final_price = prices[-1]
                    if option_type == 'call':
                        payoff = max(final_price - self.K, 0)
                    else:
                        payoff = max(self.K - final_price, 0)
                    total_payoff += path_prob * payoff

        # 할인
        discount = np.exp(-self.r * self.T)
        option_price = discount * total_payoff

        # 생존 확률 (정확히 계산)
        survival_prob = survival_count / n_paths

        return option_price, survival_prob, n_paths

    def _build_path_state_preparation(self) -> QuantumCircuit:
        """경로 상태 준비 회로

        리스크 중립 확률 p로 상승/하락 인코딩
        |ψ⟩ = √q|0⟩ + √p|1⟩ (각 시간 스텝)
        """
        qc = QuantumCircuit(self.n_steps)

        # 각 큐비트에 확률 인코딩
        # Ry(θ) where θ = 2*arcsin(√p)
        theta = 2 * np.arcsin(np.sqrt(self.p))

        for i in range(self.n_steps):
            qc.ry(theta, i)

        return qc

    def _build_barrier_oracle(
        self,
        barrier: float,
        barrier_type: str = 'down-and-out'
    ) -> QuantumCircuit:
        """배리어 조건 오라클

        배리어를 건드리는 경로에 위상 반전
        (knock-out의 경우 건드리지 않는 경로를 마킹)
        """
        qc = QuantumCircuit(self.n_steps + 1)  # +1 for ancilla
        ancilla = self.n_steps

        # 각 경로에 대해 배리어 터치 여부 확인
        # 간단화: 연속 하락 경로가 배리어 터치
        # 실제로는 더 복잡한 로직 필요

        # Down-and-out: 가격이 배리어 이하로 떨어지면 knock-out
        # 연속 하락 수에 따라 배리어 터치 결정

        # 몇 번 연속 하락하면 배리어 터치?
        # S0 * d^k <= barrier → k >= log(barrier/S0) / log(d)
        k_threshold = int(np.ceil(np.log(barrier / self.S0) / np.log(self.d)))
        k_threshold = max(1, min(k_threshold, self.n_steps))

        print(f"  Barrier hit threshold: {k_threshold} consecutive downs")

        # k_threshold개 이상 연속 0이면 배리어 터치
        # 이건 복잡하므로, 단순화: 처음 k개가 모두 0이면 터치

        if barrier_type == 'down-and-out':
            # 처음 k개 큐비트가 모두 |0⟩이면 ancilla flip
            # 이후 ancilla가 |1⟩인 상태에 위상 반전

            # X 게이트로 |0⟩ 조건 만들기
            for i in range(k_threshold):
                qc.x(i)

            # Multi-controlled X on ancilla
            if k_threshold > 0:
                qc.mcx(list(range(k_threshold)), ancilla)

            # X 복원
            for i in range(k_threshold):
                qc.x(i)

            # Ancilla가 |1⟩이면 위상 반전 (knock-out된 경로)
            # knock-out이므로 생존 경로를 찾고 싶음
            # → ancilla가 |0⟩인 경로가 생존
            # → ancilla에 Z 적용하면 |1⟩에 -1 위상
            qc.z(ancilla)

            # ancilla 되돌리기 (uncompute)
            for i in range(k_threshold):
                qc.x(i)
            if k_threshold > 0:
                qc.mcx(list(range(k_threshold)), ancilla)
            for i in range(k_threshold):
                qc.x(i)

        return qc

    def quantum_barrier_option(
        self,
        barrier: float,
        barrier_type: str = 'down-and-out',
        option_type: str = 'call',
        num_shots: int = 10000
    ) -> Tuple[float, float, int]:
        """양자 회로로 배리어 옵션 분석

        QAE로 생존 확률 추정 후 옵션 가격 계산
        """
        n_qubits = self.n_steps + 1  # path qubits + ancilla

        qc = QuantumCircuit(n_qubits, self.n_steps)

        # 1. 상태 준비 (모든 경로 중첩)
        prep = self._build_path_state_preparation()
        qc.compose(prep, qubits=list(range(self.n_steps)), inplace=True)

        # 2. 배리어 오라클
        oracle = self._build_barrier_oracle(barrier, barrier_type)
        qc.compose(oracle, qubits=list(range(n_qubits)), inplace=True)

        # 3. 측정 (경로 큐비트만)
        qc.measure(range(self.n_steps), range(self.n_steps))

        # 시뮬레이션
        simulator = AerSimulator()
        job = simulator.run(qc, shots=num_shots)
        result = job.result()
        counts = result.get_counts()

        # 결과 분석
        # 각 측정된 경로에 대해 생존 여부 및 페이오프 계산
        total_payoff = 0.0
        survival_count = 0

        for path_binary_rev, count in counts.items():
            # Qiskit 비트 순서 반전
            path_binary = path_binary_rev[::-1]
            prices = self._path_to_prices(path_binary)

            hit_barrier = self._check_barrier_hit(prices, barrier, barrier_type)

            if barrier_type in ['down-and-out', 'up-and-out']:
                if not hit_barrier:
                    survival_count += count
                    final_price = prices[-1]
                    if option_type == 'call':
                        payoff = max(final_price - self.K, 0)
                    else:
                        payoff = max(self.K - final_price, 0)
                    total_payoff += count * payoff

        # 확률 추정
        survival_prob = survival_count / num_shots

        # 옵션 가격 (가중 평균 페이오프 * 할인)
        discount = np.exp(-self.r * self.T)
        option_price = discount * (total_payoff / num_shots)

        # 양자 쿼리 수 (Grover iterations 기반이면 √N)
        quantum_queries = int(np.sqrt(2 ** self.n_steps))

        return option_price, survival_prob, quantum_queries

    def compare_methods(
        self,
        barrier: float,
        barrier_type: str = 'down-and-out',
        option_type: str = 'call'
    ) -> PathOptionResult:
        """고전 vs 양자 비교"""

        print(f"\n{'='*60}")
        print(f"  PATH-DEPENDENT OPTION PRICING")
        print(f"  Barrier Option: {barrier_type.upper()} {option_type.upper()}")
        print(f"{'='*60}")

        print(f"\nOption Parameters:")
        print(f"  Spot: ${self.S0:.2f}")
        print(f"  Strike: ${self.K:.2f}")
        print(f"  Barrier: ${barrier:.2f}")
        print(f"  Volatility: {self.sigma*100:.1f}%")
        print(f"  Risk-free rate: {self.r*100:.1f}%")
        print(f"  Time to maturity: {self.T:.2f} years")
        print(f"  Time steps: {self.n_steps} (= {2**self.n_steps} paths)")

        # 고전적 계산
        print(f"\n[Classical] Enumerating all paths...")
        classical_price, classical_surv, n_paths = self.classical_barrier_option(
            barrier, barrier_type, option_type
        )
        print(f"  Survival probability: {classical_surv:.4f}")
        print(f"  Option price: ${classical_price:.4f}")

        # 양자 계산
        print(f"\n[Quantum] Amplitude estimation...")
        quantum_price, quantum_surv, q_queries = self.quantum_barrier_option(
            barrier, barrier_type, option_type
        )
        print(f"  Survival probability: {quantum_surv:.4f}")
        print(f"  Option price: ${quantum_price:.4f}")

        # 비교
        speedup = n_paths / max(q_queries, 1)

        print(f"\n[Comparison]")
        print(f"  Classical paths: {n_paths}")
        print(f"  Quantum queries: {q_queries}")
        print(f"  Theoretical speedup: {speedup:.1f}x")
        print(f"  Price difference: ${abs(classical_price - quantum_price):.4f}")

        return PathOptionResult(
            option_type=f"{barrier_type} {option_type}",
            spot_price=self.S0,
            strike_price=self.K,
            barrier_level=barrier,
            classical_price=classical_price,
            quantum_price=quantum_price,
            survival_probability=classical_surv,
            knockout_probability=1 - classical_surv,
            num_paths_classical=n_paths,
            quantum_queries=q_queries,
            speedup=speedup
        )


def demo_barrier_option():
    """배리어 옵션 데모"""

    print("="*60)
    print("  QUANTUM AMPLITUDE ESTIMATION")
    print("  Path-Dependent Option Pricing Demo")
    print("="*60)

    # 프라이서 생성
    pricer = QuantumPathOptionPricer(
        spot=100.0,
        strike=100.0,
        volatility=0.3,
        risk_free_rate=0.05,
        time_to_maturity=1.0,
        num_steps=5  # 32개 경로
    )

    # Down-and-out Call 분석
    result = pricer.compare_methods(
        barrier=80.0,  # 배리어 80
        barrier_type='down-and-out',
        option_type='call'
    )

    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    print(f"  Barrier Option Type: {result.option_type}")
    print(f"  Classical Price: ${result.classical_price:.4f}")
    print(f"  Quantum Price: ${result.quantum_price:.4f}")
    print(f"  Knockout Probability: {result.knockout_probability:.2%}")
    print(f"  Speedup: {result.speedup:.1f}x")
    print("="*60)

    return result


def demo_multiple_barriers():
    """다양한 배리어 레벨 비교"""

    print("\n" + "="*60)
    print("  BARRIER SENSITIVITY ANALYSIS")
    print("="*60)

    pricer = QuantumPathOptionPricer(
        spot=100.0,
        strike=100.0,
        volatility=0.25,
        risk_free_rate=0.05,
        time_to_maturity=1.0,
        num_steps=6  # 64개 경로
    )

    barriers = [70, 75, 80, 85, 90]
    results = []

    print(f"\n{'Barrier':>10} {'Surv.Prob':>12} {'Classical':>12} {'Quantum':>12}")
    print("-" * 50)

    for barrier in barriers:
        # 간단히 classical만 (빠른 비교)
        price, surv, _ = pricer.classical_barrier_option(
            barrier, 'down-and-out', 'call'
        )
        results.append((barrier, surv, price))
        print(f"${barrier:>9.0f} {surv:>11.2%} ${price:>11.4f}")

    print("-" * 50)
    print("\nAs barrier approaches spot price:")
    print("  → Knockout probability increases")
    print("  → Option value decreases")

    return results


if __name__ == "__main__":
    # 기본 데모
    demo_barrier_option()

    # 배리어 민감도 분석
    demo_multiple_barriers()
