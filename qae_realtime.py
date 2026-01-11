"""
실시간 TLE 데이터 + QAE 충돌 확률 분석

CelesTrak에서 실제 위성 데이터를 가져와서
Quantum Amplitude Estimation으로 희귀 충돌 확률 계산
"""

from datetime import datetime, timedelta
from typing import List, Tuple
import numpy as np

from tle_fetcher import TLEFetcher, SatelliteData
from collision_detector import CollisionDetector, ConjunctionEvent
from qae_collision_analysis import (
    RareCollisionAnalyzer,
    CollisionScenario,
    CovarianceMatrix,
    print_analysis_result
)


def estimate_covariance_from_tle(sat: SatelliteData) -> CovarianceMatrix:
    """TLE 데이터로부터 위치 불확실성 추정

    TLE 나이와 궤도 유형에 따라 불확실성 추정
    """
    name_upper = sat.name.upper()

    # 기본 불확실성 (km)
    if "STARLINK" in name_upper:
        # Starlink: 자주 업데이트, 높은 정밀도
        sigma_r = 0.1   # 100m
        sigma_i = 0.3   # 300m
        sigma_c = 0.05  # 50m
    elif "DEB" in name_upper or "DEBRIS" in name_upper:
        # 잔해: 업데이트 빈도 낮음, 불확실성 높음
        sigma_r = 1.0   # 1km
        sigma_i = 5.0   # 5km
        sigma_c = 0.5   # 500m
    elif "ISS" in name_upper or "ZARYA" in name_upper:
        # ISS: 매우 정밀
        sigma_r = 0.05  # 50m
        sigma_i = 0.1   # 100m
        sigma_c = 0.02  # 20m
    else:
        # 일반 위성
        sigma_r = 0.3   # 300m
        sigma_i = 1.0   # 1km
        sigma_c = 0.2   # 200m

    return CovarianceMatrix(sigma_r=sigma_r, sigma_i=sigma_i, sigma_c=sigma_c)


def estimate_combined_radius(sat1: SatelliteData, sat2: SatelliteData) -> float:
    """두 위성의 결합 Hard Body Radius 추정 (m)"""
    def get_radius(sat: SatelliteData) -> float:
        name = sat.name.upper()
        if "STARLINK" in name:
            return 3.0
        elif "ISS" in name or "ZARYA" in name or "NAUKA" in name:
            return 50.0
        elif "CSS" in name or "TIANHE" in name or "WENTIAN" in name:
            return 30.0
        elif "DEB" in name or "DEBRIS" in name:
            return 0.5
        elif "R/B" in name or "ROCKET" in name:
            return 5.0
        else:
            return 2.0

    return get_radius(sat1) + get_radius(sat2)


def run_qae_analysis_on_conjunctions(
    events: List[ConjunctionEvent],
    satellites: List[SatelliteData],
    top_n: int = 10
) -> List:
    """근접 이벤트에 대해 QAE 분석 실행

    Args:
        events: ConjunctionEvent 리스트
        satellites: 위성 데이터 리스트
        top_n: 분석할 상위 이벤트 수

    Returns:
        QAE 분석 결과 리스트
    """
    analyzer = RareCollisionAnalyzer()
    sat_dict = {sat.norad_id: sat for sat in satellites}
    results = []

    print("\n" + "=" * 70)
    print("  QAE COLLISION PROBABILITY ANALYSIS")
    print("  Real TLE Data + Quantum Amplitude Estimation")
    print("=" * 70)

    for i, event in enumerate(events[:top_n], 1):
        sat1 = sat_dict.get(event.sat1_norad_id)
        sat2 = sat_dict.get(event.sat2_norad_id)

        if not sat1 or not sat2:
            continue

        # 교차 궤도만 분석 (상대 속도 > 1 km/s)
        if event.relative_velocity_km_s < 1.0:
            continue

        # Covariance 추정
        cov1 = estimate_covariance_from_tle(sat1)
        cov2 = estimate_covariance_from_tle(sat2)

        # 결합 Covariance (RSS)
        combined_cov = CovarianceMatrix(
            sigma_r=np.sqrt(cov1.sigma_r**2 + cov2.sigma_r**2),
            sigma_i=np.sqrt(cov1.sigma_i**2 + cov2.sigma_i**2),
            sigma_c=np.sqrt(cov1.sigma_c**2 + cov2.sigma_c**2)
        )

        # 결합 반경
        combined_radius = estimate_combined_radius(sat1, sat2)

        # 시나리오 생성
        scenario = CollisionScenario(
            sat1_name=event.sat1_name,
            sat2_name=event.sat2_name,
            miss_distance_km=event.min_distance_km,
            relative_velocity_km_s=event.relative_velocity_km_s,
            combined_radius_m=combined_radius,
            covariance=combined_cov,
            tca=event.tca
        )

        print(f"\n[{i}/{min(top_n, len(events))}] Analyzing: {event.sat1_name} vs {event.sat2_name}")

        # QAE 분석 (Monte Carlo 샘플 줄여서 빠르게)
        result = analyzer.analyze_scenario(scenario, mc_samples=100000)
        results.append(result)

        # 간단 출력
        print(f"    Miss Distance: {event.min_distance_km*1000:.1f} m")
        print(f"    Relative Velocity: {event.relative_velocity_km_s:.2f} km/s")
        print(f"    Analytical Pc: {result.analytical_pc:.2e}")
        print(f"    Risk: {result.risk_level}")

    return results


def main():
    """실시간 TLE + QAE 분석 실행"""
    print("=" * 70)
    print("  REAL-TIME SATELLITE COLLISION ANALYSIS WITH QAE")
    print("=" * 70)

    # 1. 데이터 로드
    print("\n[1/3] Loading satellite data...")

    # Starlink + Debris
    starlink_fetcher = TLEFetcher(group="starlink")
    starlink = starlink_fetcher.load_satellites()

    debris_fetcher = TLEFetcher(group="cosmos-2251-debris")
    debris = debris_fetcher.load_satellites()

    all_sats = starlink[:1000] + debris  # Starlink 1000개만 (속도)
    print(f"    Total objects: {len(all_sats)}")

    # 2. 근접 이벤트 탐지
    print("\n[2/3] Detecting conjunctions...")
    detector = CollisionDetector(threshold_km=5.0)  # 5km 임계값

    events = detector.detect_conjunctions(
        all_sats,
        start_time=datetime.utcnow(),
        duration_hours=6.0
    )

    # 교차 궤도만 필터 (상대 속도 > 5 km/s)
    crossing_events = [e for e in events if e.relative_velocity_km_s > 5.0]
    print(f"    Crossing orbit events: {len(crossing_events)}")

    if not crossing_events:
        print("\n    No crossing orbit conjunctions found.")
        print("    This is actually good news - space is (relatively) safe right now!")
        return

    # 3. QAE 분석
    print("\n[3/3] Running QAE analysis on top events...")
    results = run_qae_analysis_on_conjunctions(
        crossing_events,
        all_sats,
        top_n=5
    )

    # 최종 요약
    print("\n" + "=" * 70)
    print("  SUMMARY: RARE COLLISION EVENTS")
    print("=" * 70)

    if results:
        red_events = [r for r in results if "RED" in r.risk_level]
        yellow_events = [r for r in results if "YEL" in r.risk_level]

        print(f"\n[RED] Critical (Pc > 1e-4):  {len(red_events)} events")
        print(f"[YEL] Warning (Pc > 1e-5):   {len(yellow_events)} events")

        if red_events:
            print("\n*** CRITICAL EVENTS ***")
            for r in red_events:
                print(f"  - {r.scenario.sat1_name} vs {r.scenario.sat2_name}")
                print(f"    Pc = {r.analytical_pc:.2e}, TCA = {r.scenario.tca}")

        # 가장 위험한 이벤트 상세
        if results:
            most_critical = max(results, key=lambda r: r.analytical_pc)
            print("\n" + "=" * 70)
            print("  MOST CRITICAL EVENT - DETAILED ANALYSIS")
            print_analysis_result(most_critical)

    print("\n" + "=" * 70)
    print("  Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
