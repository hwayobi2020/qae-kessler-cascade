"""
LEO vs Debris 충돌 위험 분석
CelesTrak 잔해 카탈로그와 활성 위성 간 교차 궤도 충돌 탐지
"""

import argparse
from datetime import datetime, timedelta
from typing import List, Tuple
import numpy as np

from tle_fetcher import TLEFetcher, SatelliteData
from orbit_propagator import OrbitPropagator
from collision_detector import CollisionDetector, ConjunctionEvent
from collision_probability import (
    CollisionProbabilityCalculator,
    CollisionProbability,
    filter_high_risk_events
)
from visualizer import Visualizer


def load_debris_catalog() -> List[SatelliteData]:
    """주요 우주 잔해 카탈로그 로드"""
    all_debris = []

    debris_groups = [
        "cosmos-2251-debris",  # 2009년 Cosmos-Iridium 충돌
        "iridium-33-debris",   # 2009년 Cosmos-Iridium 충돌
        "1999-025",            # 2007년 중국 ASAT 테스트 (Fengyun-1C)
    ]

    for group in debris_groups:
        try:
            fetcher = TLEFetcher(group=group)
            debris = fetcher.load_satellites()
            all_debris.extend(debris)
            print(f"  {group}: {len(debris)} objects")
        except Exception as e:
            print(f"  {group}: Failed to load - {e}")

    return all_debris


def load_leo_satellites(group: str = "active") -> List[SatelliteData]:
    """LEO 활성 위성 로드"""
    fetcher = TLEFetcher(group=group)
    satellites = fetcher.load_satellites()

    # LEO 필터링 (고도 2000km 이하 추정 - 평균 운동으로 판단)
    leo_satellites = []
    for sat in satellites:
        # 평균 운동(revs/day) > 11 이면 대략 LEO
        # TLE line2에서 평균 운동 추출
        try:
            mean_motion = float(sat.line2[52:63])
            if mean_motion > 11.0:  # LEO
                leo_satellites.append(sat)
        except:
            pass

    print(f"Filtered to {len(leo_satellites)} LEO satellites")
    return leo_satellites


def analyze_leo_vs_debris(
    leo_satellites: List[SatelliteData],
    debris: List[SatelliteData],
    duration_hours: float = 24.0,
    threshold_km: float = 10.0,
    min_relative_velocity: float = 5.0  # 교차 궤도 필터
) -> Tuple[List[ConjunctionEvent], List[CollisionProbability]]:
    """LEO 위성과 잔해 간 충돌 분석

    Args:
        leo_satellites: LEO 위성 리스트
        debris: 잔해 리스트
        duration_hours: 분석 기간
        threshold_km: 근접 임계 거리
        min_relative_velocity: 최소 상대 속도 (교차 궤도 필터)

    Returns:
        (근접 이벤트 리스트, 고위험 충돌 확률 리스트)
    """
    # 모든 객체 합치기
    all_objects = leo_satellites + debris

    print(f"\nAnalyzing {len(leo_satellites)} LEO satellites vs {len(debris)} debris objects")
    print(f"Total objects: {len(all_objects)}")

    # 충돌 탐지
    detector = CollisionDetector(threshold_km=threshold_km)
    start_time = datetime.utcnow()

    all_events = detector.detect_conjunctions(
        all_objects,
        start_time=start_time,
        duration_hours=duration_hours
    )

    # LEO vs Debris 이벤트만 필터링
    leo_ids = set(sat.norad_id for sat in leo_satellites)
    debris_ids = set(d.norad_id for d in debris)

    leo_debris_events = []
    for event in all_events:
        is_sat1_leo = event.sat1_norad_id in leo_ids
        is_sat2_leo = event.sat2_norad_id in leo_ids
        is_sat1_debris = event.sat1_norad_id in debris_ids
        is_sat2_debris = event.sat2_norad_id in debris_ids

        # LEO-Debris 쌍만
        if (is_sat1_leo and is_sat2_debris) or (is_sat1_debris and is_sat2_leo):
            leo_debris_events.append(event)

    print(f"Found {len(leo_debris_events)} LEO-Debris conjunction events")

    # 충돌 확률 계산 및 고위험 필터링
    calculator = CollisionProbabilityCalculator()
    high_risk = filter_high_risk_events(
        leo_debris_events,
        calculator,
        min_relative_velocity=min_relative_velocity,
        min_pc=1e-10
    )

    print(f"High-risk events (crossing orbit, Pc > 1e-10): {len(high_risk)}")

    return leo_debris_events, high_risk


def print_risk_summary(high_risk: List[CollisionProbability]):
    """고위험 이벤트 요약 출력"""
    if not high_risk:
        print("\n" + "=" * 60)
        print("No high-risk collision events detected!")
        print("=" * 60)
        return

    print("\n" + "=" * 70)
    print("  HIGH-RISK COLLISION EVENTS (LEO vs Debris)")
    print("=" * 70)

    # 위험 레벨별 분류
    red = [e for e in high_risk if "RED" in e.risk_level]
    yellow = [e for e in high_risk if "YELLOW" in e.risk_level]
    green = [e for e in high_risk if "GREEN" in e.risk_level]

    print(f"\nRisk Summary:")
    print(f"  [RED] Critical (Pc > 1e-4):    {len(red)} events")
    print(f"  [YEL] Warning (Pc > 1e-5):     {len(yellow)} events")
    print(f"  [GRN] Monitor (Pc > 1e-7):     {len(green)} events")

    print("\n" + "-" * 70)
    print(f"{'Rank':<5} {'Pc':<12} {'Distance':<12} {'Rel.Vel':<10} {'TCA':<20} {'Objects'}")
    print("-" * 70)

    for i, cp in enumerate(high_risk[:20], 1):
        obj_str = f"{cp.sat1_name[:15]} vs {cp.sat2_name[:15]}"
        print(
            f"{i:<5} "
            f"{cp.collision_probability:<12.2e} "
            f"{cp.miss_distance_km*1000:>8.1f} m  "
            f"{cp.relative_velocity_km_s:>6.2f} km/s "
            f"{cp.tca.strftime('%m-%d %H:%M'):<20} "
            f"{obj_str}"
        )

    # 가장 위험한 이벤트 상세
    if high_risk:
        print("\n" + "=" * 70)
        print("  MOST CRITICAL EVENT DETAILS")
        print("=" * 70)
        print(high_risk[0])


def main():
    parser = argparse.ArgumentParser(
        description="LEO vs Debris Collision Risk Analysis"
    )

    parser.add_argument(
        "--leo-group", "-l",
        default="starlink",
        help="LEO satellite group (default: starlink)"
    )

    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=24.0,
        help="Analysis duration in hours (default: 24)"
    )

    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=10.0,
        help="Conjunction threshold in km (default: 10)"
    )

    parser.add_argument(
        "--min-velocity", "-v",
        type=float,
        default=5.0,
        help="Minimum relative velocity for crossing orbit filter (default: 5 km/s)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  LEO vs Debris Collision Risk Analysis")
    print("  희귀 충돌 확률 시뮬레이션")
    print("=" * 60)
    print()

    # 1. 잔해 로드
    print("[1/4] Loading debris catalogs...")
    debris = load_debris_catalog()

    if not debris:
        print("No debris data available. Exiting.")
        return

    # 2. LEO 위성 로드
    print(f"\n[2/4] Loading LEO satellites ({args.leo_group})...")
    leo_satellites = load_leo_satellites(args.leo_group)

    if not leo_satellites:
        print("No LEO satellites loaded. Exiting.")
        return

    # 3. 충돌 분석
    print(f"\n[3/4] Analyzing collisions...")
    events, high_risk = analyze_leo_vs_debris(
        leo_satellites,
        debris,
        duration_hours=args.duration,
        threshold_km=args.threshold,
        min_relative_velocity=args.min_velocity
    )

    # 4. 결과 출력
    print("\n[4/4] Results")
    print_risk_summary(high_risk)

    print("\n" + "=" * 60)
    print("  Analysis Complete!")
    print("=" * 60)

    return events, high_risk


if __name__ == "__main__":
    main()
