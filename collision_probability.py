"""
Collision Probability Calculator - 충돌 확률(Pc) 계산
NASA/ESA 표준 방법론 기반
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime

from orbit_propagator import OrbitPropagator, Position


@dataclass
class CollisionProbability:
    """충돌 확률 분석 결과"""
    sat1_norad_id: int
    sat2_norad_id: int
    sat1_name: str
    sat2_name: str
    tca: datetime                    # Time of Closest Approach
    miss_distance_km: float          # 최소 거리
    relative_velocity_km_s: float    # 상대 속도
    collision_probability: float     # 충돌 확률 (Pc)
    combined_radius_m: float         # 결합 반경 (충돌 영역)
    is_crossing_orbit: bool          # 교차 궤도 여부
    risk_level: str                  # 위험 레벨

    def __repr__(self):
        return (
            f"{'='*60}\n"
            f"Collision Probability Analysis\n"
            f"{'='*60}\n"
            f"Satellites: {self.sat1_name} vs {self.sat2_name}\n"
            f"TCA: {self.tca.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"Miss Distance: {self.miss_distance_km*1000:.1f} m\n"
            f"Relative Velocity: {self.relative_velocity_km_s:.3f} km/s ({self.relative_velocity_km_s*1000:.1f} m/s)\n"
            f"Collision Probability (Pc): {self.collision_probability:.2e}\n"
            f"Combined Hard Body Radius: {self.combined_radius_m:.1f} m\n"
            f"Crossing Orbit: {'Yes' if self.is_crossing_orbit else 'No'}\n"
            f"Risk Level: {self.risk_level}\n"
            f"{'='*60}"
        )


class CollisionProbabilityCalculator:
    """충돌 확률 계산기

    NASA Conjunction Assessment 방법론 기반:
    - 2D Probability of Collision (Pc) 계산
    - 불확실성 타원체(covariance) 기반 확률
    """

    # 기본 위성 크기 (반경, 미터)
    DEFAULT_SATELLITE_RADIUS = 5.0  # 일반 위성
    STARLINK_RADIUS = 3.0           # Starlink
    DEBRIS_RADIUS = 0.5             # 우주 잔해 (평균)
    LARGE_DEBRIS_RADIUS = 2.0       # 대형 잔해

    # 위치 불확실성 (1-sigma, km) - TLE 기반 추정
    DEFAULT_POSITION_UNCERTAINTY = 1.0  # km
    DEBRIS_POSITION_UNCERTAINTY = 5.0   # km (잔해는 불확실성 높음)

    # 위험 레벨 임계값
    RED_THRESHOLD = 1e-4      # 빨간색: 매우 위험
    YELLOW_THRESHOLD = 1e-5   # 노란색: 주의 필요
    GREEN_THRESHOLD = 1e-7    # 녹색: 안전

    def __init__(self):
        self.propagator = OrbitPropagator()

    def calculate_pc_2d(
        self,
        miss_distance_m: float,
        combined_radius_m: float,
        sigma_r: float = 1000.0,  # 반경 방향 불확실성 (m)
        sigma_t: float = 1000.0   # 접선 방향 불확실성 (m)
    ) -> float:
        """2D 충돌 확률 계산 (Chan's formula 근사)

        Args:
            miss_distance_m: 최소 거리 (m)
            combined_radius_m: 결합 충돌 반경 (m)
            sigma_r: 반경 방향 불확실성 (m)
            sigma_t: 접선 방향 불확실성 (m)

        Returns:
            충돌 확률 (0~1)
        """
        # 불확실성 타원의 면적
        sigma_combined = np.sqrt(sigma_r**2 + sigma_t**2)

        # 충돌 단면적
        collision_area = np.pi * combined_radius_m**2

        # 불확실성 타원 면적
        uncertainty_area = np.pi * sigma_r * sigma_t

        if uncertainty_area == 0:
            return 0.0

        # 지수 감쇠 기반 확률 (miss distance가 클수록 확률 감소)
        exponent = -(miss_distance_m**2) / (2 * sigma_combined**2)

        # Pc 계산
        pc = (collision_area / uncertainty_area) * np.exp(exponent)

        return min(pc, 1.0)  # 최대 1

    def estimate_satellite_radius(self, name: str) -> float:
        """위성 이름으로 반경 추정 (m)"""
        name_upper = name.upper()

        if "STARLINK" in name_upper:
            return self.STARLINK_RADIUS
        elif "DEBRIS" in name_upper or "DEB" in name_upper:
            return self.DEBRIS_RADIUS
        elif "COSMOS" in name_upper and ("DEB" in name_upper or "DEBRIS" in name_upper):
            return self.LARGE_DEBRIS_RADIUS
        elif "IRIDIUM" in name_upper and "DEB" in name_upper:
            return self.LARGE_DEBRIS_RADIUS
        elif "R/B" in name_upper or "ROCKET" in name_upper:
            return self.LARGE_DEBRIS_RADIUS * 2  # 로켓 바디
        else:
            return self.DEFAULT_SATELLITE_RADIUS

    def estimate_uncertainty(self, name: str) -> float:
        """위성 이름으로 위치 불확실성 추정 (km)"""
        name_upper = name.upper()

        if "DEB" in name_upper or "DEBRIS" in name_upper:
            return self.DEBRIS_POSITION_UNCERTAINTY
        else:
            return self.DEFAULT_POSITION_UNCERTAINTY

    def is_crossing_orbit(
        self,
        relative_velocity_km_s: float,
        threshold_km_s: float = 1.0
    ) -> bool:
        """교차 궤도 여부 판단

        상대 속도가 높으면 교차 궤도일 가능성이 높음
        - 동일 궤도면: 상대 속도 < 0.5 km/s
        - 교차 궤도: 상대 속도 > 1.0 km/s (최대 ~15 km/s)
        """
        return relative_velocity_km_s > threshold_km_s

    def get_risk_level(self, pc: float) -> str:
        """위험 레벨 분류"""
        if pc >= self.RED_THRESHOLD:
            return "[RED] CRITICAL"
        elif pc >= self.YELLOW_THRESHOLD:
            return "[YEL] WARNING"
        elif pc >= self.GREEN_THRESHOLD:
            return "[GRN] MONITOR"
        else:
            return "[WHT] SAFE"

    def analyze_conjunction(
        self,
        sat1_name: str,
        sat2_name: str,
        sat1_norad_id: int,
        sat2_norad_id: int,
        tca: datetime,
        miss_distance_km: float,
        relative_velocity_km_s: float
    ) -> CollisionProbability:
        """근접 이벤트의 충돌 확률 분석

        Args:
            sat1_name, sat2_name: 위성 이름
            sat1_norad_id, sat2_norad_id: NORAD ID
            tca: 최근접 시각
            miss_distance_km: 최소 거리 (km)
            relative_velocity_km_s: 상대 속도 (km/s)

        Returns:
            CollisionProbability 객체
        """
        # 결합 반경 계산
        r1 = self.estimate_satellite_radius(sat1_name)
        r2 = self.estimate_satellite_radius(sat2_name)
        combined_radius_m = r1 + r2

        # 불확실성 추정
        u1 = self.estimate_uncertainty(sat1_name)
        u2 = self.estimate_uncertainty(sat2_name)
        combined_uncertainty_km = np.sqrt(u1**2 + u2**2)

        # 충돌 확률 계산
        pc = self.calculate_pc_2d(
            miss_distance_m=miss_distance_km * 1000,
            combined_radius_m=combined_radius_m,
            sigma_r=combined_uncertainty_km * 1000,
            sigma_t=combined_uncertainty_km * 1000
        )

        # 교차 궤도 판단
        crossing = self.is_crossing_orbit(relative_velocity_km_s)

        # 위험 레벨
        risk = self.get_risk_level(pc)

        return CollisionProbability(
            sat1_norad_id=sat1_norad_id,
            sat2_norad_id=sat2_norad_id,
            sat1_name=sat1_name,
            sat2_name=sat2_name,
            tca=tca,
            miss_distance_km=miss_distance_km,
            relative_velocity_km_s=relative_velocity_km_s,
            collision_probability=pc,
            combined_radius_m=combined_radius_m,
            is_crossing_orbit=crossing,
            risk_level=risk
        )


def filter_high_risk_events(
    events: list,  # List[ConjunctionEvent]
    calculator: CollisionProbabilityCalculator,
    min_relative_velocity: float = 1.0,  # km/s
    min_pc: float = 1e-10
) -> list:
    """고위험 이벤트 필터링

    Args:
        events: ConjunctionEvent 리스트
        calculator: CollisionProbabilityCalculator
        min_relative_velocity: 최소 상대 속도 (교차 궤도 필터)
        min_pc: 최소 충돌 확률

    Returns:
        CollisionProbability 리스트 (고위험만)
    """
    high_risk = []

    for event in events:
        # 교차 궤도 필터 (상대 속도 기준)
        if event.relative_velocity_km_s < min_relative_velocity:
            continue

        # 충돌 확률 계산
        cp = calculator.analyze_conjunction(
            sat1_name=event.sat1_name,
            sat2_name=event.sat2_name,
            sat1_norad_id=event.sat1_norad_id,
            sat2_norad_id=event.sat2_norad_id,
            tca=event.tca,
            miss_distance_km=event.min_distance_km,
            relative_velocity_km_s=event.relative_velocity_km_s
        )

        if cp.collision_probability >= min_pc:
            high_risk.append(cp)

    # 확률 순 정렬 (높은 순)
    high_risk.sort(key=lambda x: x.collision_probability, reverse=True)

    return high_risk


if __name__ == "__main__":
    # 테스트
    calc = CollisionProbabilityCalculator()

    # 예시: Starlink vs Debris 시나리오
    cp = calc.analyze_conjunction(
        sat1_name="STARLINK-1234",
        sat2_name="COSMOS 2251 DEB",
        sat1_norad_id=44000,
        sat2_norad_id=34000,
        tca=datetime.utcnow(),
        miss_distance_km=0.5,  # 500m
        relative_velocity_km_s=10.0  # 교차 궤도
    )

    print(cp)

    # 비교: 같은 거리, 같은 궤도면 (낮은 상대 속도)
    cp2 = calc.analyze_conjunction(
        sat1_name="STARLINK-1234",
        sat2_name="STARLINK-5678",
        sat1_norad_id=44000,
        sat2_norad_id=45000,
        tca=datetime.utcnow(),
        miss_distance_km=0.5,
        relative_velocity_km_s=0.1  # 같은 궤도면
    )

    print("\nComparison (same distance, same orbital plane):")
    print(cp2)
