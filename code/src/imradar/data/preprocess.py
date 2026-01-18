from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd

from imradar.config import RadarConfig
from imradar.utils import parse_bucket_kor_to_number


@dataclass
class PanelBuildResult:
    panel: pd.DataFrame
    segment_meta: pd.DataFrame


def _safe_read_csv(path: Union[str, Path], encoding: Optional[str] = None) -> pd.DataFrame:
    """여러 인코딩 시도하여 CSV 읽기"""
    encodings_to_try = [encoding, "cp949", "euc-kr", "utf-8-sig", "utf-8", None]
    encodings_to_try = [e for e in encodings_to_try if e is not None] + [None]
    
    last_err = None
    for enc in encodings_to_try:
        try:
            if enc is None:
                return pd.read_csv(path)
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to read CSV {path}. Last error: {last_err}")


def load_p0_data(
    path: Union[str, Path],
    cfg: Optional[RadarConfig] = None,
) -> pd.DataFrame:
    """
    P0 내부 세그먼트/고객 월별 금융지표 데이터 로드
    
    iMbank_data.csv.csv 파일 로드
    핵심 컬럼: 기준년월, 업종_중분류, 사업장_시도/시군구, 법인_고객등급, 전담고객여부
              + 예금/여신/카드/채널/입출금/FX 금액
    
    Args:
        path: iMbank_data.csv.csv 파일 경로
        cfg: RadarConfig
    
    Returns:
        로드된 원본 DataFrame
    """
    if cfg is None:
        cfg = RadarConfig()
    
    df = _safe_read_csv(path)
    
    print(f"  ✓ P0 데이터 로드: {len(df):,}행, {len(df.columns)}개 컬럼")
    print(f"    기준년월 범위: {df[cfg.month_col].min()} ~ {df[cfg.month_col].max()}")
    
    return df


def _to_month_start(yyyymm: int) -> pd.Timestamp:
    y = int(yyyymm) // 100
    m = int(yyyymm) % 100
    return pd.Timestamp(year=y, month=m, day=1)


def add_time_columns(df: pd.DataFrame, month_col: str = "기준년월") -> pd.DataFrame:
    out = df.copy()
    out["month"] = out[month_col].apply(_to_month_start)
    out["year"] = out["month"].dt.year
    out["month_num"] = out["month"].dt.month
    # Month index from start
    min_month = out["month"].min()
    out["t"] = (out["month"].dt.to_period("M") - min_month.to_period("M")).apply(lambda p: p.n)
    return out


def convert_bucket_counts(df: pd.DataFrame, bucket_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in bucket_cols:
        if c in out.columns:
            out[c] = out[c].apply(parse_bucket_kor_to_number).astype("float32")
    return out


def aggregate_to_segment_month(
    raw_df: pd.DataFrame,
    cfg: RadarConfig,
    additional_group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate raw records to (segment_keys, month) level.

    - Amount columns: sum
    - Bucket count columns: sum (approximate) + mean (intensity); both are kept
    - Adds customer_count (number of raw rows per segment-month)
    """
    df = raw_df.copy()

    group_cols = [cfg.month_col] + cfg.segment_keys
    if additional_group_cols:
        group_cols += additional_group_cols

    # Ensure segment keys are non-null for stable segment_id creation
    for c in cfg.segment_keys:
        if c in df.columns:
            df[c] = df[c].fillna("미상")

    # Convert bucket columns to numeric so we can aggregate
    df = convert_bucket_counts(df, cfg.bucket_count_cols)

    # Identify numeric columns (amounts + converted buckets)
    amount_cols = [c for c in cfg.amount_cols if c in df.columns]
    bucket_cols = [c for c in cfg.bucket_count_cols if c in df.columns]

    agg_dict: Dict[str, List[str]] = {}
    for c in amount_cols:
        agg_dict[c] = ["sum"]
    for c in bucket_cols:
        agg_dict[c] = ["sum", "mean"]

    # Customer count proxy
    df["_row"] = 1
    agg_dict["_row"] = ["sum"]

    g = df.groupby(group_cols, dropna=False).agg(agg_dict)
    # Flatten MultiIndex columns
    g.columns = ["__".join([c for c in col if c]) for col in g.columns.to_flat_index()]
    g = g.reset_index()
    g = g.rename(columns={"_row__sum": "customer_count"})

    return g


def build_full_panel(segment_month: pd.DataFrame, cfg: RadarConfig) -> PanelBuildResult:
    """
    Build a complete segment × month panel.

    Strategy:
    - Create full month range from global min to max.
    - For each segment, fill missing months with 0 after the segment's first appearance.
    - Months before first appearance are marked 'pre_birth'=1 and kept as NaN for KPI amounts (optional handling).
    """
    df = segment_month.copy()
    df = add_time_columns(df, month_col=cfg.month_col)

    # Segment id
    df["segment_id"] = df[cfg.segment_keys].astype(str).agg("|".join, axis=1)

    # Segment meta
    meta_cols = cfg.segment_keys + ["segment_id"]
    segment_meta = df[meta_cols].drop_duplicates().reset_index(drop=True)

    # Full month index
    all_months = pd.date_range(df["month"].min(), df["month"].max(), freq="MS")
    month_df = pd.DataFrame({"month": all_months})
    month_df["year"] = month_df["month"].dt.year
    month_df["month_num"] = month_df["month"].dt.month
    month_df["t"] = (month_df["month"].dt.to_period("M") - df["month"].min().to_period("M")).apply(lambda p: p.n)

    # Cross join segments × months
    segments = segment_meta[["segment_id"]]
    segments["key"] = 1
    month_df["key"] = 1
    panel = segments.merge(month_df, on="key", how="outer").drop(columns=["key"])

    # Join static keys back
    panel = panel.merge(segment_meta, on="segment_id", how="left")

    # Join observed data
    obs_cols = [c for c in df.columns if c not in ["year", "month_num", "t"]]
    panel = panel.merge(df[obs_cols], on=["segment_id", "month"] + cfg.segment_keys, how="left")

    # Determine first appearance per segment (customer_count not null)
    first_month = (
        panel.loc[panel["customer_count"].notna(), ["segment_id", "month"]]
        .groupby("segment_id")["month"]
        .min()
        .rename("first_month")
        .reset_index()
    )
    panel = panel.merge(first_month, on="segment_id", how="left")
    panel["pre_birth"] = (panel["month"] < panel["first_month"]).astype(int)

    # Fill missing numeric aggregates after birth with 0
    numeric_cols = [c for c in panel.columns if c.endswith("__sum") or c.endswith("__mean")] + ["customer_count"]
    for c in numeric_cols:
        if c in panel.columns:
            panel.loc[panel["pre_birth"] == 0, c] = panel.loc[panel["pre_birth"] == 0, c].fillna(0.0)

    # Segment age (months since first_month)
    panel["segment_age"] = (panel["month"].dt.to_period("M") - panel["first_month"].dt.to_period("M")).apply(
        lambda p: p.n if pd.notnull(p) else np.nan
    )
    panel.loc[panel["pre_birth"] == 1, "segment_age"] = np.nan

    return PanelBuildResult(panel=panel, segment_meta=segment_meta)


# ============== 확장된 P0 처리 함수들 ==============

def process_p0_full_pipeline(
    p0_path: Union[str, Path],
    external_data: Optional[pd.DataFrame] = None,
    cfg: Optional[RadarConfig] = None,
    verbose: bool = True,
) -> PanelBuildResult:
    """
    P0 데이터 전체 처리 파이프라인
    
    1. P0 데이터 로드
    2. 세그먼트-월 집계
    3. 완전 패널 구축
    4. 외부 데이터 병합 (선택)
    5. KPI 계산
    
    Args:
        p0_path: P0 데이터 파일 경로
        external_data: 외부 데이터 DataFrame (선택)
        cfg: RadarConfig
        verbose: 진행 상황 출력
    
    Returns:
        PanelBuildResult
    """
    if cfg is None:
        cfg = RadarConfig()
    
    def _log(msg):
        if verbose:
            print(msg)
    
    # 1. P0 데이터 로드
    _log("[1/5] P0 데이터 로드 중...")
    raw_df = load_p0_data(p0_path, cfg)
    
    # 2. 세그먼트-월 집계
    _log("[2/5] 세그먼트-월 집계 중...")
    segment_month = aggregate_to_segment_month(raw_df, cfg)
    _log(f"      → {len(segment_month):,}개 세그먼트-월 레코드")
    
    # 3. 완전 패널 구축
    _log("[3/5] 완전 패널 구축 중...")
    result = build_full_panel(segment_month, cfg)
    _log(f"      → {len(result.panel):,}개 패널 레코드, {len(result.segment_meta):,}개 세그먼트")
    
    # 4. 외부 데이터 병합
    if external_data is not None:
        _log("[4/5] 외부 데이터 병합 중...")
        result.panel["month"] = pd.to_datetime(result.panel["month"])
        external_data["month"] = pd.to_datetime(external_data["month"])
        
        result.panel = result.panel.merge(external_data, on="month", how="left")
        
        # Forward fill for missing external data
        external_cols = [c for c in external_data.columns if c != "month"]
        for col in external_cols:
            if col in result.panel.columns:
                result.panel[col] = result.panel.groupby("segment_id")[col].ffill()
        
        _log(f"      → {len(external_cols)}개 외부 변수 추가됨")
    else:
        _log("[4/5] 외부 데이터 없음 - 스킵")
    
    # 5. KPI 계산
    _log("[5/5] KPI 계산 중...")
    result.panel = compute_kpis(result.panel, cfg)
    
    _log(f"\n  ★ 패널 구축 완료!")
    _log(f"     - 총 레코드: {len(result.panel):,}")
    _log(f"     - 세그먼트 수: {len(result.segment_meta):,}")
    _log(f"     - 컬럼 수: {len(result.panel.columns)}")
    
    return result


def compute_kpis(df: pd.DataFrame, cfg: RadarConfig) -> pd.DataFrame:
    """
    KPI 계산 (예금총잔액, 대출총잔액 등)
    
    Args:
        df: 패널 데이터
        cfg: RadarConfig
    
    Returns:
        KPI가 추가된 DataFrame
    """
    out = df.copy()
    
    for kpi_name, components in cfg.kpi_defs.items():
        kpi_value = pd.Series(0.0, index=out.index)
        
        for component in components:
            # 부호 처리 (예: "-요구불출금금액")
            if component.startswith("-"):
                col = component[1:]
                sign = -1
            else:
                col = component
                sign = 1
            
            # __sum 접미사 찾기
            col_sum = f"{col}__sum"
            if col_sum in out.columns:
                kpi_value += sign * out[col_sum].fillna(0)
            elif col in out.columns:
                kpi_value += sign * out[col].fillna(0)
        
        out[kpi_name] = kpi_value
    
    # 파생 KPI 계산
    eps = 1e-9
    
    # 한도소진율
    if "여신한도금액__sum" in out.columns:
        loan_cols = ["여신_운전자금대출잔액__sum", "여신_시설자금대출잔액__sum"]
        loan_used = sum(out[c].fillna(0) for c in loan_cols if c in out.columns)
        out["한도소진율"] = loan_used / (out["여신한도금액__sum"] + eps)
        out["한도소진율"] = out["한도소진율"].clip(0, 1)
    
    # 디지털비중
    if "디지털거래금액" in out.columns:
        total_channel = out.get("디지털거래금액", 0) + out.get("창구거래금액__sum", 0).fillna(0)
        out["디지털비중"] = out["디지털거래금액"] / (total_channel + eps)
        out["디지털비중"] = out["디지털비중"].clip(0, 1)
    
    # 자동이체비중
    if "자동이체금액__sum" in out.columns and "요구불출금금액__sum" in out.columns:
        out["자동이체비중"] = out["자동이체금액__sum"] / (out["요구불출금금액__sum"] + eps)
        out["자동이체비중"] = out["자동이체비중"].clip(0, 1)
    
    return out


def add_segment_grade_features(
    df: pd.DataFrame,
    cfg: RadarConfig,
) -> pd.DataFrame:
    """
    세그먼트 등급 관련 피처 추가
    
    고객등급, 전담여부 등을 수치화
    
    Args:
        df: 패널 데이터
        cfg: RadarConfig
    
    Returns:
        등급 피처가 추가된 DataFrame
    """
    out = df.copy()
    
    # 고객등급 가중치
    if "법인_고객등급" in out.columns:
        out["grade_weight"] = out["법인_고객등급"].map(cfg.grade_weights).fillna(1.0)
    
    # 전담고객 가중치
    if "전담고객여부" in out.columns:
        out["dedicated_weight"] = out["전담고객여부"].map(cfg.dedicated_weights).fillna(1.0)
    
    # 종합 가중치
    if "grade_weight" in out.columns and "dedicated_weight" in out.columns:
        out["segment_weight"] = out["grade_weight"] * out["dedicated_weight"]
    
    return out


def detect_segment_anomalies(
    df: pd.DataFrame,
    kpi_cols: List[str],
    group_col: str = "segment_id",
    time_col: str = "month",
    zscore_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    세그먼트별 이상치 탐지
    
    각 세그먼트-KPI 조합에서 z-score 기반 이상치 플래그 생성
    
    Args:
        df: 패널 데이터
        kpi_cols: KPI 컬럼 리스트
        group_col: 세그먼트 컬럼
        time_col: 시간 컬럼
        zscore_threshold: 이상치 임계값
    
    Returns:
        이상치 플래그가 추가된 DataFrame
    """
    out = df.copy()
    eps = 1e-9
    
    for kpi in kpi_cols:
        if kpi not in out.columns:
            continue
        
        # 세그먼트별 통계
        g = out.groupby(group_col)[kpi]
        mean_val = g.transform("mean")
        std_val = g.transform("std")
        
        # Z-score
        out[f"{kpi}__zscore"] = (out[kpi] - mean_val) / (std_val + eps)
        
        # 이상치 플래그
        out[f"{kpi}__is_anomaly"] = (out[f"{kpi}__zscore"].abs() > zscore_threshold).astype(int)
    
    return out


def compute_segment_summary(
    df: pd.DataFrame,
    cfg: RadarConfig,
) -> pd.DataFrame:
    """
    세그먼트 요약 통계 계산
    
    Args:
        df: 패널 데이터
        cfg: RadarConfig
    
    Returns:
        세그먼트별 요약 DataFrame
    """
    kpi_cols = list(cfg.kpi_defs.keys())
    
    # 마지막 관측 월
    last_obs = df[df["customer_count"] > 0].groupby("segment_id").agg({
        "month": "max",
        "customer_count": "last",
    }).reset_index()
    last_obs.columns = ["segment_id", "last_active_month", "last_customer_count"]
    
    # 세그먼트별 KPI 평균
    segment_stats = df.groupby("segment_id").agg({
        **{kpi: ["mean", "std", "min", "max"] for kpi in kpi_cols if kpi in df.columns},
        "segment_age": "max",
    }).reset_index()
    
    # 컬럼 이름 정리
    new_cols = ["segment_id"]
    for col in segment_stats.columns[1:]:
        if isinstance(col, tuple):
            new_cols.append(f"{col[0]}__{col[1]}")
        else:
            new_cols.append(col)
    segment_stats.columns = new_cols
    
    # 병합
    summary = segment_stats.merge(last_obs, on="segment_id", how="left")
    
    # 세그먼트 메타 정보 추가
    meta_cols = cfg.segment_keys + ["segment_id"]
    if all(c in df.columns for c in meta_cols):
        segment_meta = df[meta_cols].drop_duplicates()
        summary = summary.merge(segment_meta, on="segment_id", how="left")
    
    return summary


def validate_panel_data(df: pd.DataFrame, cfg: RadarConfig) -> Dict[str, any]:
    """
    패널 데이터 품질 검증
    
    Args:
        df: 패널 데이터
        cfg: RadarConfig
    
    Returns:
        검증 결과 딕셔너리
    """
    result = {
        "is_valid": True,
        "warnings": [],
        "errors": [],
        "stats": {}
    }
    
    # 1. 기본 통계
    result["stats"]["row_count"] = len(df)
    result["stats"]["segment_count"] = df["segment_id"].nunique()
    result["stats"]["month_count"] = df["month"].nunique()
    
    if "month" in df.columns:
        result["stats"]["date_range"] = f"{df['month'].min()} ~ {df['month'].max()}"
    
    # 2. 필수 컬럼 확인
    required_cols = ["segment_id", "month"] + cfg.segment_keys
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        result["errors"].append(f"필수 컬럼 누락: {missing_required}")
        result["is_valid"] = False
    
    # 3. 결측치 비율
    missing_pct = df.isnull().sum() / len(df) * 100
    high_missing = missing_pct[missing_pct > 30]
    if len(high_missing) > 0:
        result["warnings"].append(f"결측치 30% 초과 컬럼: {dict(high_missing.head(10))}")
    
    # 4. 중복 확인
    dup_count = df.duplicated(subset=["segment_id", "month"]).sum()
    if dup_count > 0:
        result["errors"].append(f"중복 레코드: {dup_count}개")
        result["is_valid"] = False
    
    # 5. 월별 연속성 확인
    months = pd.to_datetime(df["month"]).sort_values().unique()
    expected_months = pd.date_range(months.min(), months.max(), freq="MS")
    missing_months = set(expected_months) - set(months)
    if missing_months:
        result["warnings"].append(f"누락된 월: {len(missing_months)}개")
    
    result["stats"]["missing_total"] = df.isnull().sum().sum()
    result["stats"]["duplicate_count"] = dup_count
    
    return result


# 테스트 코드
if __name__ == "__main__":
    print("P0 전처리 모듈 테스트")
    print("=" * 50)
    
    from pathlib import Path
    
    # 테스트 데이터 경로
    data_dir = Path(r"C:\Users\campus4D052\Desktop\최종 프로젝트\외부 데이터")
    p0_path = data_dir / "iMbank_data.csv.csv"
    
    if p0_path.exists():
        cfg = RadarConfig()
        
        # P0 데이터 로드
        raw_df = load_p0_data(p0_path, cfg)
        print(f"\n로드된 컬럼: {raw_df.columns.tolist()[:10]}...")
        
        # 세그먼트-월 집계
        segment_month = aggregate_to_segment_month(raw_df, cfg)
        print(f"\n집계 결과: {len(segment_month):,}개 레코드")
        print(f"세그먼트 수: {segment_month[cfg.segment_keys[0]].nunique()}")
        
        # 패널 구축
        result = build_full_panel(segment_month, cfg)
        print(f"\n패널 구축 결과:")
        print(f"  - 패널 레코드: {len(result.panel):,}")
        print(f"  - 세그먼트 수: {len(result.segment_meta):,}")
        
        # 검증
        validation = validate_panel_data(result.panel, cfg)
        print(f"\n검증 결과: {validation['is_valid']}")
        if validation['warnings']:
            print(f"  경고: {validation['warnings']}")
    else:
        print(f"P0 데이터 파일을 찾을 수 없습니다: {p0_path}")
