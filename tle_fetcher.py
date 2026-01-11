"""
TLE Fetcher - CelesTrak에서 TLE 데이터 수집
"""

import requests
from sgp4.api import Satrec, WGS72
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SatelliteData:
    """위성 데이터 클래스"""
    name: str
    norad_id: int
    satrec: Satrec
    line1: str
    line2: str


class TLEFetcher:
    """CelesTrak에서 TLE 데이터를 가져오는 클래스"""

    CELESTRAK_BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"

    # 사용 가능한 카탈로그 그룹
    CATALOG_GROUPS = {
        "active": "active",           # 활성 위성
        "starlink": "starlink",       # Starlink
        "oneweb": "oneweb",           # OneWeb
        "stations": "stations",       # 우주 정거장
        "weather": "weather",         # 기상 위성
        "geo": "geo",                 # 정지 궤도 위성
        # 우주 잔해 (Debris) 카탈로그
        "cosmos-2251-debris": "cosmos-2251-debris",  # COSMOS 2251 충돌 잔해
        "iridium-33-debris": "iridium-33-debris",    # Iridium 33 충돌 잔해
        "1999-025": "1999-025",                       # Fengyun-1C 잔해 (ASAT 테스트)
    }

    def __init__(self, group: str = "active"):
        """
        Args:
            group: CelesTrak 카탈로그 그룹 (기본: active)
        """
        self.group = group
        self.satellites: List[SatelliteData] = []

    def fetch_tle(self) -> str:
        """CelesTrak에서 TLE 데이터 다운로드"""
        url = f"{self.CELESTRAK_BASE_URL}?GROUP={self.group}&FORMAT=tle"

        print(f"Fetching TLE data from CelesTrak ({self.group})...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        return response.text

    def parse_tle(self, tle_data: str) -> List[SatelliteData]:
        """TLE 텍스트를 파싱하여 위성 객체 리스트 생성

        Args:
            tle_data: TLE 포맷 텍스트 (3줄씩: 이름, Line1, Line2)

        Returns:
            SatelliteData 객체 리스트
        """
        lines = tle_data.strip().split('\n')
        satellites = []

        i = 0
        while i + 2 < len(lines):
            name = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()

            # TLE 라인 유효성 검사
            if not line1.startswith('1 ') or not line2.startswith('2 '):
                i += 1
                continue

            try:
                # SGP4 Satrec 객체 생성
                satrec = Satrec.twoline2rv(line1, line2, WGS72)

                # NORAD ID 추출
                norad_id = int(line1[2:7])

                sat_data = SatelliteData(
                    name=name,
                    norad_id=norad_id,
                    satrec=satrec,
                    line1=line1,
                    line2=line2
                )
                satellites.append(sat_data)

            except Exception as e:
                print(f"Warning: Failed to parse TLE for {name}: {e}")

            i += 3

        return satellites

    def load_satellites(self) -> List[SatelliteData]:
        """TLE 데이터를 가져와서 위성 리스트 반환"""
        tle_data = self.fetch_tle()
        self.satellites = self.parse_tle(tle_data)
        print(f"Loaded {len(self.satellites)} satellites")
        return self.satellites

    def load_from_file(self, filepath: str) -> List[SatelliteData]:
        """로컬 TLE 파일에서 로드"""
        with open(filepath, 'r') as f:
            tle_data = f.read()
        self.satellites = self.parse_tle(tle_data)
        print(f"Loaded {len(self.satellites)} satellites from {filepath}")
        return self.satellites

    def save_to_file(self, filepath: str):
        """현재 TLE 데이터를 파일로 저장"""
        tle_data = self.fetch_tle()
        with open(filepath, 'w') as f:
            f.write(tle_data)
        print(f"Saved TLE data to {filepath}")

    def get_satellite_by_norad_id(self, norad_id: int) -> Optional[SatelliteData]:
        """NORAD ID로 위성 검색"""
        for sat in self.satellites:
            if sat.norad_id == norad_id:
                return sat
        return None

    def get_satellite_by_name(self, name: str) -> List[SatelliteData]:
        """이름으로 위성 검색 (부분 일치)"""
        name_lower = name.lower()
        return [sat for sat in self.satellites if name_lower in sat.name.lower()]


if __name__ == "__main__":
    # 테스트
    fetcher = TLEFetcher(group="active")
    satellites = fetcher.load_satellites()

    print(f"\nFirst 5 satellites:")
    for sat in satellites[:5]:
        print(f"  {sat.norad_id}: {sat.name}")
