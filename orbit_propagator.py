"""
Orbit Propagator - SGP4를 이용한 궤도 전파
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from dataclasses import dataclass

from sgp4.api import jday


@dataclass
class Position:
    """위성 위치/속도 데이터"""
    time: datetime
    # ECI 좌표 (km)
    x: float
    y: float
    z: float
    # 속도 (km/s)
    vx: float
    vy: float
    vz: float

    @property
    def position_vector(self) -> np.ndarray:
        """위치 벡터 반환"""
        return np.array([self.x, self.y, self.z])

    @property
    def velocity_vector(self) -> np.ndarray:
        """속도 벡터 반환"""
        return np.array([self.vx, self.vy, self.vz])


class OrbitPropagator:
    """SGP4를 이용한 궤도 전파 클래스"""

    def __init__(self):
        pass

    @staticmethod
    def datetime_to_jday(dt: datetime) -> Tuple[float, float]:
        """datetime을 Julian Date로 변환"""
        jd, fr = jday(
            dt.year, dt.month, dt.day,
            dt.hour, dt.minute, dt.second + dt.microsecond / 1e6
        )
        return jd, fr

    def propagate_single(self, satrec, dt: datetime) -> Optional[Position]:
        """단일 시점에서 위성 위치 계산

        Args:
            satrec: SGP4 Satrec 객체
            dt: 계산할 시간

        Returns:
            Position 객체 또는 None (에러 시)
        """
        jd, fr = self.datetime_to_jday(dt)

        # SGP4 전파
        error, position, velocity = satrec.sgp4(jd, fr)

        if error != 0:
            return None

        return Position(
            time=dt,
            x=position[0],
            y=position[1],
            z=position[2],
            vx=velocity[0],
            vy=velocity[1],
            vz=velocity[2]
        )

    def propagate_range(
        self,
        satrec,
        start_time: datetime,
        end_time: datetime,
        step_minutes: float = 1.0
    ) -> List[Position]:
        """시간 범위에서 위성 궤도 전파

        Args:
            satrec: SGP4 Satrec 객체
            start_time: 시작 시간
            end_time: 종료 시간
            step_minutes: 시간 간격 (분)

        Returns:
            Position 객체 리스트
        """
        positions = []
        current_time = start_time
        step = timedelta(minutes=step_minutes)

        while current_time <= end_time:
            pos = self.propagate_single(satrec, current_time)
            if pos:
                positions.append(pos)
            current_time += step

        return positions

    def propagate_satellites_batch(
        self,
        satellites: List,  # List[SatelliteData]
        times: List[datetime]
    ) -> dict:
        """여러 위성에 대해 여러 시점에서 일괄 전파

        Args:
            satellites: SatelliteData 객체 리스트
            times: 시간 리스트

        Returns:
            {norad_id: {time_index: Position}} 형태의 딕셔너리
        """
        results = {}

        # Julian Date 미리 계산
        jd_fr_list = [self.datetime_to_jday(t) for t in times]

        for sat in satellites:
            sat_positions = {}
            for i, (jd, fr) in enumerate(jd_fr_list):
                error, position, velocity = sat.satrec.sgp4(jd, fr)

                if error == 0:
                    sat_positions[i] = Position(
                        time=times[i],
                        x=position[0],
                        y=position[1],
                        z=position[2],
                        vx=velocity[0],
                        vy=velocity[1],
                        vz=velocity[2]
                    )

            results[sat.norad_id] = sat_positions

        return results

    def propagate_to_array(
        self,
        satellites: List,
        times: List[datetime]
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """위성 위치를 NumPy 배열로 반환 (빠른 계산용)

        Args:
            satellites: SatelliteData 객체 리스트
            times: 시간 리스트

        Returns:
            positions: (n_times, n_sats, 3) 형태의 위치 배열
            velocities: (n_times, n_sats, 3) 형태의 속도 배열
            valid_sat_ids: 유효한 위성 NORAD ID 리스트
        """
        n_times = len(times)
        n_sats = len(satellites)

        positions = np.full((n_times, n_sats, 3), np.nan)
        velocities = np.full((n_times, n_sats, 3), np.nan)
        valid_sat_ids = []

        # Julian Date 미리 계산
        jd_fr_list = [self.datetime_to_jday(t) for t in times]

        for sat_idx, sat in enumerate(satellites):
            valid_count = 0
            for time_idx, (jd, fr) in enumerate(jd_fr_list):
                error, pos, vel = sat.satrec.sgp4(jd, fr)

                if error == 0:
                    positions[time_idx, sat_idx] = pos
                    velocities[time_idx, sat_idx] = vel
                    valid_count += 1

            if valid_count > 0:
                valid_sat_ids.append(sat.norad_id)

        return positions, velocities, valid_sat_ids


def generate_time_range(
    start_time: datetime,
    duration_hours: float = 24.0,
    step_minutes: float = 1.0
) -> List[datetime]:
    """시간 범위 생성

    Args:
        start_time: 시작 시간
        duration_hours: 기간 (시간)
        step_minutes: 간격 (분)

    Returns:
        datetime 리스트
    """
    times = []
    current = start_time
    end_time = start_time + timedelta(hours=duration_hours)
    step = timedelta(minutes=step_minutes)

    while current <= end_time:
        times.append(current)
        current += step

    return times


if __name__ == "__main__":
    from tle_fetcher import TLEFetcher

    # 테스트
    fetcher = TLEFetcher(group="stations")  # 우주 정거장만 테스트
    satellites = fetcher.load_satellites()

    if satellites:
        propagator = OrbitPropagator()
        now = datetime.utcnow()

        # ISS 위치 계산
        iss = fetcher.get_satellite_by_name("ISS")
        if iss:
            iss_sat = iss[0]
            pos = propagator.propagate_single(iss_sat.satrec, now)
            if pos:
                print(f"\n{iss_sat.name} position at {now}:")
                print(f"  X: {pos.x:.2f} km")
                print(f"  Y: {pos.y:.2f} km")
                print(f"  Z: {pos.z:.2f} km")
                print(f"  Distance from Earth center: {np.linalg.norm(pos.position_vector):.2f} km")
