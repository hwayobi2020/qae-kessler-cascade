"""
Collision Detector - KD-Tree 기반 위성 충돌(근접) 탐지
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from dataclasses import dataclass

from tle_fetcher import SatelliteData
from orbit_propagator import OrbitPropagator, Position


@dataclass
class ConjunctionEvent:
    """근접 이벤트 (Conjunction)"""
    sat1_norad_id: int
    sat2_norad_id: int
    sat1_name: str
    sat2_name: str
    tca: datetime  # Time of Closest Approach
    min_distance_km: float  # 최소 거리 (km)
    relative_velocity_km_s: float  # 상대 속도 (km/s)

    def __repr__(self):
        return (
            f"Conjunction: {self.sat1_name} - {self.sat2_name}\n"
            f"  TCA: {self.tca}\n"
            f"  Min Distance: {self.min_distance_km:.3f} km\n"
            f"  Relative Velocity: {self.relative_velocity_km_s:.3f} km/s"
        )


class CollisionDetector:
    """위성 충돌 탐지 클래스"""

    def __init__(
        self,
        threshold_km: float = 10.0,
        coarse_step_minutes: float = 1.0,
        fine_step_seconds: float = 1.0
    ):
        """
        Args:
            threshold_km: 근접 판단 임계 거리 (km)
            coarse_step_minutes: 초기 탐색 시간 간격 (분)
            fine_step_seconds: 정밀 탐색 시간 간격 (초)
        """
        self.threshold_km = threshold_km
        self.coarse_step_minutes = coarse_step_minutes
        self.fine_step_seconds = fine_step_seconds
        self.propagator = OrbitPropagator()

    def find_close_pairs_at_time(
        self,
        positions: np.ndarray,
        sat_indices: List[int]
    ) -> List[Tuple[int, int, float]]:
        """특정 시점에서 근접한 위성 쌍 찾기 (KD-Tree 사용)

        Args:
            positions: (n_sats, 3) 형태의 위치 배열
            sat_indices: 위성 인덱스 리스트

        Returns:
            [(sat1_idx, sat2_idx, distance), ...] 형태의 리스트
        """
        # NaN 제거
        valid_mask = ~np.isnan(positions[:, 0])
        valid_positions = positions[valid_mask]
        valid_indices = np.array(sat_indices)[valid_mask]

        if len(valid_positions) < 2:
            return []

        # KD-Tree 구축
        tree = cKDTree(valid_positions)

        # 임계 거리 내 쌍 찾기
        pairs = tree.query_pairs(r=self.threshold_km)

        results = []
        for i, j in pairs:
            distance = np.linalg.norm(valid_positions[i] - valid_positions[j])
            results.append((valid_indices[i], valid_indices[j], distance))

        return results

    def detect_conjunctions(
        self,
        satellites: List[SatelliteData],
        start_time: datetime,
        duration_hours: float = 24.0
    ) -> List[ConjunctionEvent]:
        """시간 범위에서 모든 근접 이벤트 탐지

        Args:
            satellites: 위성 리스트
            start_time: 시작 시간
            duration_hours: 분석 기간 (시간)

        Returns:
            ConjunctionEvent 리스트
        """
        print(f"Detecting conjunctions for {len(satellites)} satellites...")
        print(f"Time range: {start_time} to {start_time + timedelta(hours=duration_hours)}")
        print(f"Threshold: {self.threshold_km} km")

        # 시간 범위 생성
        times = []
        current = start_time
        end_time = start_time + timedelta(hours=duration_hours)
        step = timedelta(minutes=self.coarse_step_minutes)

        while current <= end_time:
            times.append(current)
            current += step

        print(f"Analyzing {len(times)} time steps...")

        # 위성 인덱스 매핑
        sat_id_to_idx = {sat.norad_id: i for i, sat in enumerate(satellites)}
        sat_id_to_data = {sat.norad_id: sat for sat in satellites}

        # 전체 기간 위치 계산
        positions, velocities, valid_ids = self.propagator.propagate_to_array(
            satellites, times
        )

        # 각 시점에서 근접 쌍 탐지
        candidate_pairs = set()

        for time_idx in range(len(times)):
            pos_at_time = positions[time_idx]
            pairs = self.find_close_pairs_at_time(
                pos_at_time,
                list(range(len(satellites)))
            )

            for sat1_idx, sat2_idx, dist in pairs:
                pair_key = tuple(sorted([sat1_idx, sat2_idx]))
                candidate_pairs.add(pair_key)

        print(f"Found {len(candidate_pairs)} candidate pairs")

        # 각 후보 쌍에 대해 정밀 TCA 계산
        events = []

        for sat1_idx, sat2_idx in candidate_pairs:
            sat1 = satellites[sat1_idx]
            sat2 = satellites[sat2_idx]

            event = self._compute_tca(sat1, sat2, start_time, duration_hours)
            if event and event.min_distance_km <= self.threshold_km:
                events.append(event)

        # 거리 순 정렬
        events.sort(key=lambda e: e.min_distance_km)

        print(f"Confirmed {len(events)} conjunction events")
        return events

    def _compute_tca(
        self,
        sat1: SatelliteData,
        sat2: SatelliteData,
        start_time: datetime,
        duration_hours: float
    ) -> Optional[ConjunctionEvent]:
        """두 위성 간 TCA (Time of Closest Approach) 계산

        Args:
            sat1, sat2: 위성 데이터
            start_time: 시작 시간
            duration_hours: 분석 기간

        Returns:
            ConjunctionEvent 또는 None
        """
        # 1분 간격으로 초기 탐색
        times = []
        current = start_time
        end_time = start_time + timedelta(hours=duration_hours)
        step = timedelta(minutes=self.coarse_step_minutes)

        while current <= end_time:
            times.append(current)
            current += step

        min_dist = float('inf')
        min_time_idx = 0

        for i, t in enumerate(times):
            pos1 = self.propagator.propagate_single(sat1.satrec, t)
            pos2 = self.propagator.propagate_single(sat2.satrec, t)

            if pos1 and pos2:
                dist = np.linalg.norm(pos1.position_vector - pos2.position_vector)
                if dist < min_dist:
                    min_dist = dist
                    min_time_idx = i

        if min_dist > self.threshold_km * 2:  # 임계값의 2배보다 멀면 스킵
            return None

        # 정밀 TCA 탐색 (scipy.optimize 사용)
        t_start = times[max(0, min_time_idx - 1)]
        t_end = times[min(len(times) - 1, min_time_idx + 1)]

        def distance_at_time(minutes_from_start):
            t = t_start + timedelta(minutes=minutes_from_start)
            pos1 = self.propagator.propagate_single(sat1.satrec, t)
            pos2 = self.propagator.propagate_single(sat2.satrec, t)

            if pos1 and pos2:
                return np.linalg.norm(pos1.position_vector - pos2.position_vector)
            return float('inf')

        # 최소화
        duration_minutes = (t_end - t_start).total_seconds() / 60
        result = minimize_scalar(
            distance_at_time,
            bounds=(0, duration_minutes),
            method='bounded'
        )

        tca = t_start + timedelta(minutes=result.x)
        min_distance = result.fun

        # TCA에서 상대 속도 계산
        pos1 = self.propagator.propagate_single(sat1.satrec, tca)
        pos2 = self.propagator.propagate_single(sat2.satrec, tca)

        if pos1 and pos2:
            rel_velocity = np.linalg.norm(pos1.velocity_vector - pos2.velocity_vector)
        else:
            rel_velocity = 0.0

        return ConjunctionEvent(
            sat1_norad_id=sat1.norad_id,
            sat2_norad_id=sat2.norad_id,
            sat1_name=sat1.name,
            sat2_name=sat2.name,
            tca=tca,
            min_distance_km=min_distance,
            relative_velocity_km_s=rel_velocity
        )

    def detect_for_target(
        self,
        target_sat: SatelliteData,
        other_satellites: List[SatelliteData],
        start_time: datetime,
        duration_hours: float = 24.0
    ) -> List[ConjunctionEvent]:
        """특정 위성에 대한 근접 이벤트만 탐지

        Args:
            target_sat: 대상 위성
            other_satellites: 다른 위성들
            start_time: 시작 시간
            duration_hours: 분석 기간

        Returns:
            ConjunctionEvent 리스트
        """
        print(f"Detecting conjunctions for {target_sat.name}...")

        events = []

        for other_sat in other_satellites:
            if other_sat.norad_id == target_sat.norad_id:
                continue

            event = self._compute_tca(target_sat, other_sat, start_time, duration_hours)
            if event and event.min_distance_km <= self.threshold_km:
                events.append(event)

        events.sort(key=lambda e: e.min_distance_km)
        return events


if __name__ == "__main__":
    from tle_fetcher import TLEFetcher

    # 테스트 (우주 정거장들만으로 간단 테스트)
    fetcher = TLEFetcher(group="stations")
    satellites = fetcher.load_satellites()

    if len(satellites) >= 2:
        detector = CollisionDetector(threshold_km=100.0)  # 테스트용 넓은 임계값
        now = datetime.utcnow()

        events = detector.detect_conjunctions(
            satellites,
            start_time=now,
            duration_hours=1.0  # 1시간만 테스트
        )

        print(f"\n=== Conjunction Events ===")
        for event in events[:5]:  # 상위 5개만 출력
            print(event)
            print()
