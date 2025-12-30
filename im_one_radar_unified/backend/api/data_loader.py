"""
Data loading utilities for the Segment Radar API.
Loads CSV/Parquet data from outputs directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from django.conf import settings


class DataLoader:
    """Singleton data loader for segment radar outputs."""
    
    _instance: Optional['DataLoader'] = None
    _cached_data: Dict[str, pd.DataFrame] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def outputs_dir(self) -> Path:
        return settings.OUTPUTS_DIR
    
    def get_latest_mvp_dir(self) -> Optional[Path]:
        """Find the latest mvp_* directory if it exists."""
        mvp_dirs = sorted(self.outputs_dir.glob("mvp_*"), reverse=True)
        if mvp_dirs:
            return mvp_dirs[0]
        return None
    
    def find_file(self, *relative_paths: str) -> Optional[Path]:
        """
        Find a file in outputs directory.
        First checks directly under outputs_dir, then in mvp_* subdirectories.
        """
        for rel_path in relative_paths:
            direct_path = self.outputs_dir / rel_path
            if direct_path.exists():
                return direct_path
        
        mvp_dir = self.get_latest_mvp_dir()
        if mvp_dir:
            for rel_path in relative_paths:
                mvp_path = mvp_dir / rel_path
                if mvp_path.exists():
                    return mvp_path
        
        return None
    
    def transform_wide_forecasts_to_long(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform wide format forecasts to long format expected by API.
        """
        kpi_list = ['예금총잔액', '대출총잔액', '카드총사용', '디지털거래금액', '순유입', 'FX총액']
        horizon_suffixes = {'_x': 1, '_y': 2, '': 3}
        
        records = []
        for _, row in df.iterrows():
            segment_id = row.get('segment_id', '')
            forecast_month = row.get('pred_month', '')
            
            for kpi in kpi_list:
                for suffix, horizon in horizon_suffixes.items():
                    col_name = f"{kpi}{suffix}" if suffix else kpi
                    if col_name in row and pd.notna(row[col_name]):
                        records.append({
                            'segment_id': segment_id,
                            'kpi': kpi,
                            'horizon': horizon,
                            'forecast_month': forecast_month,
                            'predicted': float(row[col_name]),
                            'actual': None,
                            'lower': None,
                            'upper': None,
                        })
        
        return pd.DataFrame(records)
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all output files from outputs/ directory."""
        data = {}
        
        # Load forecast metrics
        path = self.find_file("forecast_metrics.csv")
        if path:
            data["forecast_metrics"] = pd.read_csv(path)
        
        # Load watchlist alerts
        path = self.find_file("watchlist_alerts.csv")
        if path:
            data["watchlist_alerts"] = pd.read_csv(path)
        
        # Load watchlist drivers
        path = self.find_file("watchlist_drivers.csv")
        if path:
            data["watchlist_drivers"] = pd.read_csv(path)
        
        # Load actions
        path = self.find_file("actions_top.csv")
        if path:
            data["actions"] = pd.read_csv(path)
        
        # Load segment panel (support both parquet and csv)
        panel_path = self.find_file("segment_panel.parquet")
        if panel_path:
            try:
                data["panel"] = pd.read_parquet(panel_path)
            except Exception:
                pass
        
        if "panel" not in data:
            panel_path = self.find_file("segment_panel.csv")
            if panel_path:
                data["panel"] = pd.read_csv(panel_path)
        
        # Load segment forecasts
        forecasts_path = self.find_file(
            "forecasts/segment_forecasts_next.csv",
            "segment_forecasts_next.csv"
        )
        if forecasts_path:
            raw_forecasts = pd.read_csv(forecasts_path)
            if 'pred_month' in raw_forecasts.columns:
                data["segment_forecasts"] = self.transform_wide_forecasts_to_long(raw_forecasts)
            else:
                data["segment_forecasts"] = raw_forecasts
        
        return data
    
    def get_data(self) -> Dict[str, pd.DataFrame]:
        """Get cached data or load from disk."""
        if not self._cached_data:
            self._cached_data = self.load_all()
        return self._cached_data
    
    def refresh(self):
        """Refresh cached data."""
        self._cached_data = self.load_all()
    
    def get_panel(self) -> Optional[pd.DataFrame]:
        """Get panel DataFrame."""
        return self.get_data().get("panel")
    
    def get_forecasts(self) -> Optional[pd.DataFrame]:
        """Get forecasts DataFrame."""
        return self.get_data().get("segment_forecasts")
    
    def get_watchlist(self) -> Optional[pd.DataFrame]:
        """Get watchlist alerts DataFrame."""
        return self.get_data().get("watchlist_alerts")
    
    def get_drivers(self) -> Optional[pd.DataFrame]:
        """Get watchlist drivers DataFrame."""
        return self.get_data().get("watchlist_drivers")
    
    def get_actions(self) -> Optional[pd.DataFrame]:
        """Get actions DataFrame."""
        return self.get_data().get("actions")


# Singleton instance
data_loader = DataLoader()
