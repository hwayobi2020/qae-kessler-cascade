"""
Visualizer - 충돌 분석 결과 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
from typing import List, Optional

from tle_fetcher import SatelliteData
from orbit_propagator import OrbitPropagator
from collision_detector import ConjunctionEvent


class Visualizer:
    """충돌 분석 결과 시각화 클래스"""

    def __init__(self):
        self.propagator = OrbitPropagator()
        # 한글 폰트 설정 시도
        try:
            plt.rcParams['font.family'] = 'Malgun Gothic'
        except:
            pass
        plt.rcParams['axes.unicode_minus'] = False

    def plot_conjunction_timeline(
        self,
        events: List[ConjunctionEvent],
        title: str = "Conjunction Events Timeline",
        save_path: Optional[str] = None
    ):
        """근접 이벤트 타임라인 플롯

        Args:
            events: ConjunctionEvent 리스트
            title: 그래프 제목
            save_path: 저장 경로 (None이면 화면 표시)
        """
        if not events:
            print("No events to plot")
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        # 데이터 준비
        times = [e.tca for e in events]
        distances = [e.min_distance_km for e in events]
        labels = [f"{e.sat1_name[:15]}\n{e.sat2_name[:15]}" for e in events]

        # 거리에 따른 색상 (가까울수록 빨간색)
        colors = plt.cm.RdYlGn(np.array(distances) / max(distances))

        # 산점도
        scatter = ax.scatter(
            times, distances,
            c=distances,
            cmap='RdYlGn',
            s=100,
            alpha=0.7,
            edgecolors='black'
        )

        # 컬러바
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Distance (km)')

        # 축 설정
        ax.set_xlabel('Time (UTC)')
        ax.set_ylabel('Minimum Distance (km)')
        ax.set_title(title)

        # x축 포맷
        fig.autofmt_xdate()

        # 그리드
        ax.grid(True, alpha=0.3)

        # 임계선
        ax.axhline(y=1.0, color='red', linestyle='--', label='1 km threshold')
        ax.axhline(y=5.0, color='orange', linestyle='--', label='5 km threshold')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved timeline plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_distance_histogram(
        self,
        events: List[ConjunctionEvent],
        title: str = "Conjunction Distance Distribution",
        save_path: Optional[str] = None
    ):
        """근접 거리 히스토그램

        Args:
            events: ConjunctionEvent 리스트
            title: 그래프 제목
            save_path: 저장 경로
        """
        if not events:
            print("No events to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        distances = [e.min_distance_km for e in events]

        # 히스토그램
        n, bins, patches = ax.hist(
            distances,
            bins=20,
            edgecolor='black',
            alpha=0.7
        )

        # 거리에 따른 색상
        for i, patch in enumerate(patches):
            bin_center = (bins[i] + bins[i+1]) / 2
            color = plt.cm.RdYlGn(bin_center / max(distances))
            patch.set_facecolor(color)

        ax.set_xlabel('Minimum Distance (km)')
        ax.set_ylabel('Number of Events')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # 통계 표시
        stats_text = (
            f"Total Events: {len(events)}\n"
            f"Min Distance: {min(distances):.3f} km\n"
            f"Max Distance: {max(distances):.3f} km\n"
            f"Mean Distance: {np.mean(distances):.3f} km"
        )
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved histogram to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_orbit_3d(
        self,
        satellites: List[SatelliteData],
        duration_hours: float = 1.5,
        step_minutes: float = 1.0,
        title: str = "Satellite Orbits",
        save_path: Optional[str] = None
    ):
        """3D 궤도 시각화

        Args:
            satellites: 시각화할 위성 리스트 (최대 10개 권장)
            duration_hours: 궤도 기간
            step_minutes: 시간 간격
            title: 그래프 제목
            save_path: 저장 경로
        """
        if not satellites:
            print("No satellites to plot")
            return

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 지구 그리기 (간단한 구)
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        earth_radius = 6371  # km
        x = earth_radius * np.outer(np.cos(u), np.sin(v))
        y = earth_radius * np.outer(np.sin(u), np.sin(v))
        z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='blue', alpha=0.3)

        # 시간 범위
        now = datetime.utcnow()
        times = []
        current = now
        end = now + timedelta(hours=duration_hours)
        step = timedelta(minutes=step_minutes)
        while current <= end:
            times.append(current)
            current += step

        # 색상 맵
        colors = plt.cm.rainbow(np.linspace(0, 1, len(satellites)))

        # 각 위성 궤도
        for sat, color in zip(satellites[:10], colors):  # 최대 10개
            positions = self.propagator.propagate_range(
                sat.satrec, now,
                now + timedelta(hours=duration_hours),
                step_minutes
            )

            if positions:
                xs = [p.x for p in positions]
                ys = [p.y for p in positions]
                zs = [p.z for p in positions]

                ax.plot(xs, ys, zs, color=color, label=sat.name[:20], linewidth=1.5)

                # 현재 위치 표시
                ax.scatter(xs[0], ys[0], zs[0], color=color, s=50, marker='o')

        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title(title)

        # 범례
        ax.legend(loc='upper left', fontsize=8)

        # 축 비율 동일하게
        max_range = 10000
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 3D orbit plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_conjunction_detail(
        self,
        event: ConjunctionEvent,
        sat1: SatelliteData,
        sat2: SatelliteData,
        window_minutes: float = 30.0,
        save_path: Optional[str] = None
    ):
        """특정 근접 이벤트 상세 시각화

        Args:
            event: ConjunctionEvent
            sat1, sat2: 위성 데이터
            window_minutes: TCA 전후 분석 범위 (분)
            save_path: 저장 경로
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 시간 범위
        start = event.tca - timedelta(minutes=window_minutes)
        end = event.tca + timedelta(minutes=window_minutes)

        times = []
        current = start
        step = timedelta(seconds=10)
        while current <= end:
            times.append(current)
            current += step

        # 거리 계산
        distances = []
        rel_velocities = []

        for t in times:
            pos1 = self.propagator.propagate_single(sat1.satrec, t)
            pos2 = self.propagator.propagate_single(sat2.satrec, t)

            if pos1 and pos2:
                dist = np.linalg.norm(pos1.position_vector - pos2.position_vector)
                rel_vel = np.linalg.norm(pos1.velocity_vector - pos2.velocity_vector)
                distances.append(dist)
                rel_velocities.append(rel_vel)
            else:
                distances.append(np.nan)
                rel_velocities.append(np.nan)

        time_labels = [(t - event.tca).total_seconds() / 60 for t in times]

        # 1. 거리 vs 시간
        ax1 = axes[0, 0]
        ax1.plot(time_labels, distances, 'b-', linewidth=2)
        ax1.axvline(x=0, color='red', linestyle='--', label='TCA')
        ax1.axhline(y=event.min_distance_km, color='green', linestyle=':', label=f'Min: {event.min_distance_km:.3f} km')
        ax1.set_xlabel('Time from TCA (minutes)')
        ax1.set_ylabel('Distance (km)')
        ax1.set_title('Distance vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 상대 속도 vs 시간
        ax2 = axes[0, 1]
        ax2.plot(time_labels, rel_velocities, 'r-', linewidth=2)
        ax2.axvline(x=0, color='red', linestyle='--', label='TCA')
        ax2.set_xlabel('Time from TCA (minutes)')
        ax2.set_ylabel('Relative Velocity (km/s)')
        ax2.set_title('Relative Velocity vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 3D 궤적 (TCA 주변)
        ax3 = axes[1, 0]
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')

        pos1_list = []
        pos2_list = []
        for t in times:
            pos1 = self.propagator.propagate_single(sat1.satrec, t)
            pos2 = self.propagator.propagate_single(sat2.satrec, t)
            if pos1 and pos2:
                pos1_list.append(pos1.position_vector)
                pos2_list.append(pos2.position_vector)

        if pos1_list and pos2_list:
            pos1_arr = np.array(pos1_list)
            pos2_arr = np.array(pos2_list)

            ax3.plot(pos1_arr[:, 0], pos1_arr[:, 1], pos1_arr[:, 2], 'b-', label=sat1.name[:15])
            ax3.plot(pos2_arr[:, 0], pos2_arr[:, 1], pos2_arr[:, 2], 'r-', label=sat2.name[:15])

            # TCA 위치 표시
            tca_idx = len(times) // 2
            ax3.scatter(*pos1_arr[tca_idx], color='blue', s=100, marker='*')
            ax3.scatter(*pos2_arr[tca_idx], color='red', s=100, marker='*')

        ax3.set_xlabel('X (km)')
        ax3.set_ylabel('Y (km)')
        ax3.set_zlabel('Z (km)')
        ax3.set_title('3D Trajectories near TCA')
        ax3.legend()

        # 4. 이벤트 정보 텍스트
        ax4 = axes[1, 1]
        ax4.axis('off')

        info_text = (
            f"Conjunction Event Details\n"
            f"{'='*40}\n\n"
            f"Satellite 1: {event.sat1_name}\n"
            f"  NORAD ID: {event.sat1_norad_id}\n\n"
            f"Satellite 2: {event.sat2_name}\n"
            f"  NORAD ID: {event.sat2_norad_id}\n\n"
            f"Time of Closest Approach (TCA):\n"
            f"  {event.tca.strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n"
            f"Minimum Distance: {event.min_distance_km:.4f} km\n"
            f"                  {event.min_distance_km * 1000:.1f} m\n\n"
            f"Relative Velocity: {event.relative_velocity_km_s:.4f} km/s\n"
            f"                   {event.relative_velocity_km_s * 1000:.1f} m/s"
        )

        ax4.text(
            0.1, 0.9, info_text,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        )

        plt.suptitle(
            f"Conjunction: {event.sat1_name} - {event.sat2_name}",
            fontsize=14, fontweight='bold'
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved conjunction detail to {save_path}")
        else:
            plt.show()

        plt.close()


if __name__ == "__main__":
    from tle_fetcher import TLEFetcher
    from collision_detector import CollisionDetector

    # 테스트
    fetcher = TLEFetcher(group="stations")
    satellites = fetcher.load_satellites()

    if satellites:
        visualizer = Visualizer()

        # 3D 궤도 시각화 테스트
        visualizer.plot_orbit_3d(
            satellites[:5],
            duration_hours=1.5,
            title="Space Stations Orbits"
        )
