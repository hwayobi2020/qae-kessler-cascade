# QAE 기반 케슬러 증후군 연쇄 충돌 예측

Quantum Amplitude Estimation을 활용한 우주 파편 연쇄 충돌(케슬러 증후군) 확률 예측

## 핵심 아이디어

### 왜 QAE인가?

**QAE(Quantum Amplitude Estimation)**는 양자 컴퓨터로 확률을 추정하는 알고리즘입니다.

| 문제 유형 | 고전적 방법 | QAE | 승자 |
|----------|-----------|-----|-----|
| 단순 확률 추정 | O(1/√N) | O(1/N) + 오버헤드 | 고전적 |
| **경로 폭발 문제** | **O(2^n)** | **O(2^(n/2))** | **QAE** |

**경로 폭발 문제**에서 QAE가 압도적입니다:

| 시뮬레이션 스텝 | 고전적 연산 | QAE 쿼리 | 속도 향상 |
|---------------|-----------|---------|----------|
| 10 | 1,024 | 32 | 32배 |
| 20 | 1,048,576 | 1,024 | 1,024배 |
| 30 | 10억+ | 32,000 | 32,000배 |

### 왜 케슬러 증후군인가?

케슬러 증후군은 **경로 의존적 문제**입니다:

```
충돌 발생 → 파편 증가 → 충돌 확률 증가 → 또 충돌 → ...
```

각 단계의 결과가 다음 단계 확률에 영향 → **조건부 확률의 연쇄** → **경로 폭발**

이 구조는 금융의 **배리어 옵션**과 동일합니다:

| 배리어 옵션 | 케슬러 증후군 |
|-----------|-------------|
| 주가 | 파편 수 |
| 가격 변동 | 충돌 이벤트 |
| 배리어 레벨 | 임계 파편 수 |
| Knock-in | 연쇄 반응 시작 |
| 옵션 가치 | 연쇄 확률 |

## 스타링크와 충돌 회피

현대 위성(Starlink 등)은 자동 충돌 회피 시스템을 갖추고 있습니다:

- 회피 성공률: 약 95%
- 24시간 전 경고 시 기동 가능

**하지만 한계가 있습니다:**

| 충돌 유형 | 회피 가능? | 위험 수준 |
|----------|----------|----------|
| 활성 위성 ↔ 활성 위성 | O | 낮음 |
| 활성 위성 ↔ 파편 | O | 낮음 |
| **파편 ↔ 파편** | **X** | **높음** |
| **고장 위성 ↔ 파편** | **X** | **높음** |

> **핵심 통찰**: 케슬러 증후군은 위성-위성 충돌이 아닌, **파편-파편 충돌**에서 시작됩니다.
> 회피 불가능 충돌이 전체의 ~70%를 초과하면 임계점(Tipping Point)에 도달합니다.

## 프로젝트 구조

```
qae/
├── Core Modules
│   ├── tle_fetcher.py          # CelesTrak TLE 데이터 수집
│   ├── orbit_propagator.py     # SGP4 궤도 전파
│   ├── collision_detector.py   # KD-Tree 충돌 탐지
│   └── collision_probability.py # NASA 방식 Pc 계산
│
├── Quantum Modules
│   ├── quantum_collision.py    # 기본 QAE 충돌 확률
│   ├── qae_path_option.py      # 경로 의존 옵션 데모
│   ├── kessler_cascade_qae.py  # 케슬러 연쇄 QAE 시뮬레이션
│   └── quantum_cluster_collision.py # Grover 탐색 클러스터
│
├── Analysis
│   ├── cascade_cluster.py      # 공간 클러스터링
│   ├── analyze_debris.py       # 파편 분석
│   └── visualizer.py           # 시각화
│
└── Documentation
    ├── README.md               # 이 문서
    └── DISCUSSION.md           # 상세 논의 기록
```

## 설치

```bash
pip install -r requirements.txt
```

### 필수 패키지
- Python 3.8+
- qiskit >= 1.0
- qiskit-aer
- sgp4
- numpy, scipy
- requests
- matplotlib

## 사용법

### 케슬러 연쇄 분석 실행
```bash
python kessler_cascade_qae.py
```

### 경로 의존 옵션 데모
```bash
python qae_path_option.py
```

## 데이터 소스

| 데이터 | 출처 | 용도 |
|-------|-----|-----|
| TLE (궤도 요소) | [CelesTrak](https://celestrak.org) | 위성/파편 위치 |
| 파편 생성 모델 | NASA Breakup Model | Payoff 함수 |
| 충돌 단면적 | ESA 보고서 | 충돌 확률 계산 |

## 로드맵: v2.0 개선 계획

현재 버전은 **PoC(개념 증명)** 수준입니다. 다음 개선이 계획되어 있습니다:

### 1. Payoff Function 복잡도 상향

**현재 (v1.0):**
```
충돌 발생? → Yes/No (이진 분기)
```

**개선 (v2.0):**
```
충돌 에너지 E → 파편 생성 PDF: P(N_debris | E) → 양자 진폭 인코딩
```

- NASA Standard Breakup Model 적용
- 충돌 에너지 기반 파편 수 분포 (로그정규 분포)
- 양자 상태로 인코딩: `|ψ⟩ = Σ √P(n) |n⟩`

### 2. IQAE (Iterative QAE) 적용

**현재 문제:**
- 표준 QAE는 오라클이 복잡해지면 회로 깊이 폭발

**해결:**
- IQAE: 반복적 추정으로 회로 깊이 감소
- Qiskit `IterativeAmplitudeEstimation` 사용
- (옵션) Variational QAE for NISQ 장치

### 3. 실험 설계

```
시나리오:
- 파편: 1,000 ~ 10,000개
- 위성: 5,000개 (5% 고장)
- 회피 성공률: 95%
- 스텝: 10, 20, 30 (각 5년 단위)

비교:
- Classical Monte Carlo
- 표준 QAE
- IQAE
```

### 구현 로드맵

| 단계 | 작업 | 상태 |
|-----|-----|-----|
| 1 | NASA Breakup Model → PDF 구현 | 예정 |
| 2 | PDF → 양자 진폭 인코딩 회로 | 예정 |
| 3 | IQAE 통합 및 오라클 설계 | 예정 |
| 4 | 실험 및 비교 | 예정 |

## 현재 한계

- 이진 분기 모델 (단순화)
- 시뮬레이터 실험 (실제 양자 하드웨어 X)
- 3D 궤도 분포 단순화
- 단일 궤도 쉘 가정

## 참고 문헌

- Kessler, D.J. & Cour-Palais, B.G. (1978). "Collision Frequency of Artificial Satellites"
- Brassard, G., et al. (2002). "Quantum Amplitude Amplification and Estimation"
- Stamatopoulos, N., et al. (2020). "Option Pricing using Quantum Computers" - IBM
- ESA Space Debris Office Reports
- [CelesTrak](https://celestrak.org) for TLE data

## 라이선스

MIT License

## 기여

이 프로젝트는 QAE의 실용적 응용 분야를 탐색하는 과정에서 시작되었습니다.
"경로 의존적 문제 = QAE의 최적 적용 분야"라는 통찰이 핵심입니다.

Issues와 PR 환영합니다.
