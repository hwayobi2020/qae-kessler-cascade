"""
연쇄충돌 위험 클러스터 탐지

특정 시점에 공간적으로 밀집된 위성/잔해 클러스터를 찾아
연쇄충돌(Kessler Syndrome) 위험 지역 식별
"""

from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not installed. Using simple grid-based clustering.")

from tle_fetcher import TLEFetcher, SatelliteData
from orbit_propagator import OrbitPropagator, Position


@dataclass
class SpatialCluster:
    """공간 클러스터 정보"""
    cluster_id: int
    center: Tuple[float, float, float]  # ECI 좌표 (km)
    radius_km: float                     # 클러스터 반경
    members: List[SatelliteData]         # 클러스터 내 객체들
    member_positions: List[Position]     # 각 객체의 위치
    altitude_km: float                   # 평균 고도
    density: float                       # 밀도 (객체/km^3)
    risk_score: float                    # 연쇄충돌 위험 점수

    @property
    def size(self) -> int:
        return len(self.members)

    def __repr__(self):
        return (
            f"Cluster #{self.cluster_id}: {self.size} objects, "
            f"radius={self.radius_km:.1f}km, alt={self.altitude_km:.0f}km, "
            f"risk={self.risk_score:.2f}"
        )


class CascadeClusterDetector:
    """연쇄충돌 위험 클러스터 탐지기

    DBSCAN 알고리즘으로 공간적으로 가까운 위성/잔해 그룹 탐지
    - eps: 클러스터 반경 (km)
    - min_samples: 클러스터 최소 객체 수
    """

    # 잔해 확산 속도 (km/s) - 충돌 에너지에 따라 다름
    DEBRIS_SPREAD_VELOCITY = 0.5  # 보수적 추정

    # 위험 레벨 가중치
    DEBRIS_WEIGHT = 2.0      # 잔해는 위험도 높음
    ACTIVE_SAT_WEIGHT = 1.0  # 활성 위성
    LARGE_SAT_WEIGHT = 3.0   # 대형 객체 (ISS, 로켓바디)

    def __init__(
        self,
        eps_km: float = 50.0,      # 클러스터 반경 50km
        min_samples: int = 3,       # 최소 3개 이상
        time_window_hours: float = 1.0  # 잔해 확산 시간
    ):
        self.eps_km = eps_km
        self.min_samples = min_samples
        self.time_window_hours = time_window_hours
        self.propagator = OrbitPropagator()

    def _get_object_weight(self, sat: SatelliteData) -> float:
        """객체 유형별 위험 가중치"""
        name = sat.name.upper()

        if "DEB" in name or "DEBRIS" in name:
            return self.DEBRIS_WEIGHT
        elif "ISS" in name or "ZARYA" in name or "CSS" in name:
            return self.LARGE_SAT_WEIGHT
        elif "R/B" in name or "ROCKET" in name:
            return self.LARGE_SAT_WEIGHT
        else:
            return self.ACTIVE_SAT_WEIGHT

    def _calculate_positions(
        self,
        satellites: List[SatelliteData],
        time: datetime
    ) -> Tuple[np.ndarray, List[Position], List[SatelliteData]]:
        """모든 위성의 위치 계산"""
        positions = []
        valid_positions = []
        valid_sats = []

        for sat in satellites:
            # propagate_single은 satrec 객체를 받음
            pos = self.propagator.propagate_single(sat.satrec, time)
            if pos and not any(np.isnan([pos.x, pos.y, pos.z])):
                positions.append([pos.x, pos.y, pos.z])
                valid_positions.append(pos)
                valid_sats.append(sat)

        return np.array(positions), valid_positions, valid_sats

    def _simple_grid_clustering(
        self,
        positions: np.ndarray,
        satellites: List[SatelliteData],
        valid_positions: List[Position]
    ) -> List[SpatialCluster]:
        """sklearn 없을 때 간단한 그리드 기반 클러스터링"""
        if len(positions) == 0:
            return []

        # 그리드 셀 크기 = eps
        grid_size = self.eps_km

        # 각 위치를 그리드 셀에 할당
        grid_cells = defaultdict(list)
        for i, pos in enumerate(positions):
            cell = (
                int(pos[0] / grid_size),
                int(pos[1] / grid_size),
                int(pos[2] / grid_size)
            )
            grid_cells[cell].append(i)

        # min_samples 이상인 셀만 클러스터로
        clusters = []
        cluster_id = 0

        for cell, indices in grid_cells.items():
            if len(indices) >= self.min_samples:
                cluster_positions = positions[indices]
                center = np.mean(cluster_positions, axis=0)

                # 클러스터 반경 계산
                distances = np.linalg.norm(cluster_positions - center, axis=1)
                radius = np.max(distances) if len(distances) > 0 else 0

                # 고도 계산
                altitude = np.linalg.norm(center) - 6371  # 지구 반경 빼기

                # 멤버 정보
                members = [satellites[i] for i in indices]
                member_pos = [valid_positions[i] for i in indices]

                # 밀도 계산
                volume = (4/3) * np.pi * max(radius, 1)**3
                density = len(indices) / volume

                # 위험 점수
                risk = self._calculate_risk_score(members, density, radius)

                clusters.append(SpatialCluster(
                    cluster_id=cluster_id,
                    center=tuple(center),
                    radius_km=radius,
                    members=members,
                    member_positions=member_pos,
                    altitude_km=altitude,
                    density=density,
                    risk_score=risk
                ))
                cluster_id += 1

        return clusters

    def _calculate_risk_score(
        self,
        members: List[SatelliteData],
        density: float,
        radius_km: float
    ) -> float:
        """연쇄충돌 위험 점수 계산

        고려 요소:
        1. 클러스터 내 객체 수
        2. 객체 유형별 가중치 (잔해 > 활성위성)
        3. 공간 밀도 (반경 대비 객체 수)
        4. 잔해 비율
        """
        n_objects = len(members)

        # 가중 객체 수
        weighted_count = sum(self._get_object_weight(sat) for sat in members)

        # 잔해 비율
        debris_count = sum(1 for m in members
                         if "DEB" in m.name.upper() or "DEBRIS" in m.name.upper())
        debris_ratio = debris_count / n_objects if n_objects > 0 else 0

        # 밀도 점수: 반경 대비 객체 수 (작은 공간에 많은 객체 = 위험)
        # 100km당 객체 수로 정규화
        density_score = n_objects / max(radius_km / 100, 0.1)

        # 연쇄충돌 잠재력: 잔해가 있으면 새 잔해 생성 가능성
        cascade_potential = 1 + debris_ratio * 2  # 잔해 비율 가중

        # 위험 점수 계산
        # = 가중객체수 * 밀도점수 * 연쇄잠재력
        risk = weighted_count * density_score * cascade_potential

        # 정규화 (0-100 스케일)
        # 기준: 10개 객체가 100km 반경에 있으면 ~30점
        normalized_risk = risk / 3

        return min(normalized_risk, 100)

    def detect_clusters(
        self,
        satellites: List[SatelliteData],
        time: datetime
    ) -> List[SpatialCluster]:
        """공간 클러스터 탐지

        Args:
            satellites: 위성 데이터 리스트
            time: 분석 시점

        Returns:
            SpatialCluster 리스트 (위험도 순)
        """
        print(f"Calculating positions for {len(satellites)} objects at {time}...")
        positions, valid_positions, valid_sats = self._calculate_positions(satellites, time)

        if len(positions) < self.min_samples:
            print("Not enough valid positions for clustering")
            return []

        print(f"Valid positions: {len(positions)}")

        if HAS_SKLEARN:
            # DBSCAN 클러스터링
            print(f"Running DBSCAN (eps={self.eps_km}km, min_samples={self.min_samples})...")
            db = DBSCAN(eps=self.eps_km, min_samples=self.min_samples, n_jobs=-1)
            labels = db.fit_predict(positions)

            # 클러스터별 정리
            clusters = []
            unique_labels = set(labels)
            unique_labels.discard(-1)  # 노이즈 제외

            for cluster_id in unique_labels:
                mask = labels == cluster_id
                cluster_positions = positions[mask]

                # 클러스터 중심과 반경
                center = np.mean(cluster_positions, axis=0)
                distances = np.linalg.norm(cluster_positions - center, axis=1)
                radius = np.max(distances)

                # 고도
                altitude = np.linalg.norm(center) - 6371

                # 멤버 정보
                indices = np.where(mask)[0]
                members = [valid_sats[i] for i in indices]
                member_pos = [valid_positions[i] for i in indices]

                # 밀도
                volume = (4/3) * np.pi * max(radius, 1)**3
                density = len(members) / volume

                # 위험 점수
                risk = self._calculate_risk_score(members, density, radius)

                clusters.append(SpatialCluster(
                    cluster_id=int(cluster_id),
                    center=tuple(center),
                    radius_km=radius,
                    members=members,
                    member_positions=member_pos,
                    altitude_km=altitude,
                    density=density,
                    risk_score=risk
                ))
        else:
            # 간단한 그리드 클러스터링
            clusters = self._simple_grid_clustering(positions, valid_sats, valid_positions)

        # 위험도 순 정렬
        clusters.sort(key=lambda c: c.risk_score, reverse=True)

        print(f"Found {len(clusters)} clusters")
        return clusters

    def analyze_time_evolution(
        self,
        satellites: List[SatelliteData],
        start_time: datetime,
        duration_hours: float = 24.0,
        step_hours: float = 1.0
    ) -> Dict[datetime, List[SpatialCluster]]:
        """시간에 따른 클러스터 변화 분석

        Args:
            satellites: 위성 데이터
            start_time: 시작 시간
            duration_hours: 분석 기간
            step_hours: 시간 간격

        Returns:
            시간별 클러스터 딕셔너리
        """
        results = {}
        current = start_time
        end_time = start_time + timedelta(hours=duration_hours)

        step = 0
        total_steps = int(duration_hours / step_hours)

        while current <= end_time:
            step += 1
            print(f"\n[Step {step}/{total_steps}] Time: {current}")
            clusters = self.detect_clusters(satellites, current)
            results[current] = clusters

            # 고위험 클러스터 출력
            high_risk = [c for c in clusters if c.risk_score > 20]
            if high_risk:
                print(f"  High-risk clusters: {len(high_risk)}")
                for c in high_risk[:3]:
                    print(f"    {c}")

            current += timedelta(hours=step_hours)

        return results


def print_cluster_analysis(clusters: List[SpatialCluster], top_n: int = 10):
    """클러스터 분석 결과 출력"""
    print("\n" + "=" * 70)
    print("  CASCADE COLLISION RISK CLUSTER ANALYSIS")
    print("=" * 70)

    if not clusters:
        print("\nNo significant clusters detected.")
        return

    print(f"\nTotal clusters found: {len(clusters)}")

    # 위험도별 분류
    critical = [c for c in clusters if c.risk_score >= 50]
    high = [c for c in clusters if 30 <= c.risk_score < 50]
    medium = [c for c in clusters if 15 <= c.risk_score < 30]

    print(f"\n[CRITICAL] Risk >= 50:  {len(critical)} clusters")
    print(f"[HIGH]     Risk 30-50:  {len(high)} clusters")
    print(f"[MEDIUM]   Risk 15-30:  {len(medium)} clusters")

    print("\n" + "-" * 70)
    print(f"{'Rank':<5} {'Risk':<8} {'Size':<6} {'Radius':<10} {'Altitude':<12} {'Debris%':<10}")
    print("-" * 70)

    for i, cluster in enumerate(clusters[:top_n], 1):
        debris_count = sum(1 for m in cluster.members
                         if "DEB" in m.name.upper() or "DEBRIS" in m.name.upper())
        debris_pct = (debris_count / cluster.size * 100) if cluster.size > 0 else 0

        print(
            f"{i:<5} "
            f"{cluster.risk_score:<8.1f} "
            f"{cluster.size:<6} "
            f"{cluster.radius_km:<10.1f} "
            f"{cluster.altitude_km:<12.0f} "
            f"{debris_pct:<10.0f}%"
        )

    # 가장 위험한 클러스터 상세
    if clusters:
        print("\n" + "=" * 70)
        print("  MOST CRITICAL CLUSTER - DETAILS")
        print("=" * 70)

        c = clusters[0]
        print(f"\nCluster #{c.cluster_id}")
        print(f"  Risk Score: {c.risk_score:.1f}")
        print(f"  Objects: {c.size}")
        print(f"  Radius: {c.radius_km:.1f} km")
        print(f"  Altitude: {c.altitude_km:.0f} km")
        print(f"  Center (ECI): ({c.center[0]:.1f}, {c.center[1]:.1f}, {c.center[2]:.1f}) km")

        print(f"\n  Members:")
        for i, sat in enumerate(c.members[:10], 1):
            obj_type = "DEBRIS" if "DEB" in sat.name.upper() else "ACTIVE"
            print(f"    {i}. [{obj_type}] {sat.name} (NORAD: {sat.norad_id})")

        if c.size > 10:
            print(f"    ... and {c.size - 10} more objects")


def main():
    """클러스터 탐지 실행"""
    print("=" * 70)
    print("  CASCADE COLLISION RISK - CLUSTER DETECTION")
    print("  Identifying high-density regions for Kessler Syndrome risk")
    print("=" * 70)

    # 데이터 로드
    print("\n[1/3] Loading satellite data...")

    # Starlink + 잔해
    starlink_fetcher = TLEFetcher(group="starlink")
    starlink = starlink_fetcher.load_satellites()
    print(f"  Starlink: {len(starlink)} satellites")

    debris_fetcher = TLEFetcher(group="cosmos-2251-debris")
    debris = debris_fetcher.load_satellites()
    print(f"  Cosmos-2251 debris: {len(debris)} objects")

    # 더 많은 데이터 사용
    all_objects = starlink[:2000] + debris
    print(f"  Total: {len(all_objects)} objects")

    # 클러스터 탐지 - 다양한 스케일로 시도
    print("\n[2/3] Detecting spatial clusters...")

    # 1. 좁은 범위 (연쇄충돌 직접 위험)
    print("\n--- Small scale (50km radius, min 3 objects) ---")
    detector_small = CascadeClusterDetector(
        eps_km=50.0,
        min_samples=3
    )
    clusters_small = detector_small.detect_clusters(all_objects, datetime.utcnow())

    # 2. 중간 범위 (궤도 면 밀집)
    print("\n--- Medium scale (200km radius, min 5 objects) ---")
    detector_medium = CascadeClusterDetector(
        eps_km=200.0,
        min_samples=5
    )
    clusters_medium = detector_medium.detect_clusters(all_objects, datetime.utcnow())

    # 3. 넓은 범위 (궤도 쉘 밀집)
    print("\n--- Large scale (500km radius, min 10 objects) ---")
    detector_large = CascadeClusterDetector(
        eps_km=500.0,
        min_samples=10
    )
    clusters_large = detector_large.detect_clusters(all_objects, datetime.utcnow())

    # 결과 출력
    print("\n[3/3] Analysis Results")

    print("\n" + "=" * 70)
    print("  SMALL SCALE CLUSTERS (50km) - Immediate Cascade Risk")
    print("=" * 70)
    if clusters_small:
        print_cluster_analysis(clusters_small)
    else:
        print("No clusters at 50km scale - good news!")

    print("\n" + "=" * 70)
    print("  MEDIUM SCALE CLUSTERS (200km) - Orbital Plane Density")
    print("=" * 70)
    print_cluster_analysis(clusters_medium)

    print("\n" + "=" * 70)
    print("  LARGE SCALE CLUSTERS (500km) - Orbital Shell Density")
    print("=" * 70)
    print_cluster_analysis(clusters_large)

    print("\n" + "=" * 70)
    print("  Analysis Complete!")
    print("=" * 70)

    return clusters_small, clusters_medium, clusters_large


if __name__ == "__main__":
    main()
