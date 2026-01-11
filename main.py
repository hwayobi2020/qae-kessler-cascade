"""
QAE - Quasi-Analytical Ephemeris based Satellite Collision Prediction System
위성 충돌 예측 시스템 메인 스크립트
"""

import argparse
from datetime import datetime, timedelta
from typing import Optional
import sys

from tle_fetcher import TLEFetcher
from orbit_propagator import OrbitPropagator
from collision_detector import CollisionDetector, ConjunctionEvent
from visualizer import Visualizer


def print_header():
    """헤더 출력"""
    print("=" * 60)
    print("  QAE - Satellite Collision Prediction System")
    print("  CelesTrak TLE + SGP4 기반 위성 충돌 예측")
    print("=" * 60)
    print()


def print_events_summary(events: list, top_n: int = 20):
    """근접 이벤트 요약 출력"""
    print(f"\n{'='*60}")
    print(f"  Detection Results: {len(events)} conjunction events found")
    print(f"{'='*60}\n")

    if not events:
        print("No conjunction events detected within the threshold.")
        return

    print(f"Top {min(top_n, len(events))} closest approaches:\n")
    print(f"{'Rank':<5} {'Distance (km)':<15} {'TCA (UTC)':<22} {'Satellite 1':<20} {'Satellite 2':<20}")
    print("-" * 85)

    for i, event in enumerate(events[:top_n], 1):
        print(
            f"{i:<5} "
            f"{event.min_distance_km:<15.4f} "
            f"{event.tca.strftime('%Y-%m-%d %H:%M:%S'):<22} "
            f"{event.sat1_name[:18]:<20} "
            f"{event.sat2_name[:18]:<20}"
        )

    print()

    # 통계
    distances = [e.min_distance_km for e in events]
    print(f"Statistics:")
    print(f"  - Minimum distance: {min(distances):.4f} km")
    print(f"  - Maximum distance: {max(distances):.4f} km")
    print(f"  - Average distance: {sum(distances)/len(distances):.4f} km")

    # 위험 레벨별 분류
    critical = sum(1 for d in distances if d < 1.0)
    warning = sum(1 for d in distances if 1.0 <= d < 5.0)
    caution = sum(1 for d in distances if 5.0 <= d < 10.0)

    print(f"\nRisk Classification:")
    print(f"  - Critical (<1 km): {critical} events")
    print(f"  - Warning (1-5 km): {warning} events")
    print(f"  - Caution (5-10 km): {caution} events")


def run_full_analysis(
    group: str = "active",
    threshold_km: float = 10.0,
    duration_hours: float = 24.0,
    output_dir: Optional[str] = None,
    visualize: bool = True
):
    """전체 분석 실행

    Args:
        group: CelesTrak 위성 그룹
        threshold_km: 근접 임계 거리 (km)
        duration_hours: 분석 기간 (시간)
        output_dir: 결과 저장 디렉토리
        visualize: 시각화 여부
    """
    print_header()

    # 1. TLE 데이터 로드
    print("[1/4] Fetching TLE data from CelesTrak...")
    fetcher = TLEFetcher(group=group)

    try:
        satellites = fetcher.load_satellites()
    except Exception as e:
        print(f"Error fetching TLE data: {e}")
        return

    if not satellites:
        print("No satellites loaded. Exiting.")
        return

    print(f"      Loaded {len(satellites)} satellites\n")

    # 2. 충돌 탐지
    print(f"[2/4] Detecting conjunctions...")
    print(f"      Threshold: {threshold_km} km")
    print(f"      Duration: {duration_hours} hours")

    detector = CollisionDetector(threshold_km=threshold_km)
    start_time = datetime.utcnow()

    events = detector.detect_conjunctions(
        satellites,
        start_time=start_time,
        duration_hours=duration_hours
    )

    # 3. 결과 출력
    print("\n[3/4] Analysis complete")
    print_events_summary(events)

    # 4. 시각화
    if visualize and events:
        print("\n[4/4] Generating visualizations...")

        visualizer = Visualizer()

        # 타임라인
        timeline_path = f"{output_dir}/conjunction_timeline.png" if output_dir else None
        visualizer.plot_conjunction_timeline(
            events,
            title=f"Conjunction Events ({group}) - Next {duration_hours}h",
            save_path=timeline_path
        )

        # 히스토그램
        hist_path = f"{output_dir}/distance_histogram.png" if output_dir else None
        visualizer.plot_distance_histogram(
            events,
            save_path=hist_path
        )

        # 가장 가까운 이벤트 상세 분석
        if events:
            closest = events[0]
            sat1 = fetcher.get_satellite_by_norad_id(closest.sat1_norad_id)
            sat2 = fetcher.get_satellite_by_norad_id(closest.sat2_norad_id)

            if sat1 and sat2:
                detail_path = f"{output_dir}/closest_conjunction.png" if output_dir else None
                visualizer.plot_conjunction_detail(
                    closest, sat1, sat2,
                    save_path=detail_path
                )

    print("\n" + "=" * 60)
    print("  Analysis Complete!")
    print("=" * 60)

    return events


def run_target_analysis(
    target_name: str,
    group: str = "active",
    threshold_km: float = 50.0,
    duration_hours: float = 24.0
):
    """특정 위성 대상 분석

    Args:
        target_name: 대상 위성 이름 (부분 일치)
        group: CelesTrak 위성 그룹
        threshold_km: 근접 임계 거리 (km)
        duration_hours: 분석 기간 (시간)
    """
    print_header()

    # TLE 로드
    print(f"[1/3] Fetching TLE data...")
    fetcher = TLEFetcher(group=group)
    satellites = fetcher.load_satellites()

    if not satellites:
        print("No satellites loaded. Exiting.")
        return

    # 대상 위성 찾기
    targets = fetcher.get_satellite_by_name(target_name)
    if not targets:
        print(f"No satellite found matching '{target_name}'")
        return

    target = targets[0]
    print(f"      Target: {target.name} (NORAD ID: {target.norad_id})\n")

    # 충돌 탐지
    print(f"[2/3] Detecting conjunctions for {target.name}...")
    detector = CollisionDetector(threshold_km=threshold_km)

    events = detector.detect_for_target(
        target,
        satellites,
        start_time=datetime.utcnow(),
        duration_hours=duration_hours
    )

    # 결과 출력
    print("\n[3/3] Analysis complete")
    print_events_summary(events)

    return events


def main():
    parser = argparse.ArgumentParser(
        description="QAE - Satellite Collision Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # 기본 분석 (Active Satellites, 24시간)
  python main.py --group starlink          # Starlink 위성만 분석
  python main.py --threshold 5             # 5km 임계값으로 분석
  python main.py --duration 48             # 48시간 분석
  python main.py --target ISS              # ISS에 대한 분석
  python main.py --no-viz                  # 시각화 없이 분석
        """
    )

    parser.add_argument(
        "--group", "-g",
        type=str,
        default="active",
        choices=["active", "starlink", "oneweb", "stations", "weather", "geo"],
        help="CelesTrak satellite group (default: active)"
    )

    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=10.0,
        help="Conjunction threshold distance in km (default: 10.0)"
    )

    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=24.0,
        help="Analysis duration in hours (default: 24.0)"
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Analyze conjunctions for specific satellite (by name)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for plots"
    )

    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization"
    )

    args = parser.parse_args()

    try:
        if args.target:
            # 특정 위성 분석
            run_target_analysis(
                target_name=args.target,
                group=args.group,
                threshold_km=args.threshold,
                duration_hours=args.duration
            )
        else:
            # 전체 분석
            run_full_analysis(
                group=args.group,
                threshold_km=args.threshold,
                duration_hours=args.duration,
                output_dir=args.output,
                visualize=not args.no_viz
            )

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
