"""
외부 거시경제 데이터 로더 및 전처리 모듈

외부 데이터 종류:
- ESI (Economic Sentiment Index): 경기심리지수
- Fed 금리: 미국 연방준비제도 금리
- 환율 (USD/KRW): 달러-원 환율 (일별/월별)
- 유가: WTI/Brent 원유 가격
- 무역: 수출입 실적
- KASI: 한국 캘린더 (공휴일, 영업일 등)
- 국내 금리: 콜금리, KORIBOR, CD, CP, 국고채 등
"""
from __future__ import annotations

from typing import Optional, List, Dict, Union
from pathlib import Path
import pandas as pd
import numpy as np


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


def load_esi(path: Union[str, Path]) -> pd.DataFrame:
    """
    ESI (Economic Sentiment Index) 데이터 로드
    
    입력 컬럼: year, month, ym, ESI
    출력 컬럼: month (datetime), ESI
    """
    df = _safe_read_csv(path)
    
    # ym 컬럼을 datetime으로 변환
    if "ym" in df.columns:
        df["month"] = pd.to_datetime(df["ym"].astype(str), format="%Y%m")
    elif "year" in df.columns and "month" in df.columns:
        df["month"] = pd.to_datetime(df["year"].astype(str) + df["month"].astype(str).str.zfill(2), format="%Y%m")
    
    # 필요한 컬럼만 선택
    result = df[["month", "ESI"]].copy()
    result = result.sort_values("month").reset_index(drop=True)
    
    return result


def load_fed_rate(path: Union[str, Path], use_derived: bool = True) -> pd.DataFrame:
    """
    Fed 금리 데이터 로드
    
    Args:
        path: CSV 파일 경로
        use_derived: True면 파생변수 파일, False면 원본 파일 사용
    
    출력 컬럼:
        - month (datetime)
        - fed_rate: 연준 금리
        - 파생변수: fed_rate_mom_diff, fed_rate_yoy_pct, fed_rate_ma3, fed_rate_vol_3m, policy_stance 등
    """
    df = _safe_read_csv(path)
    
    # date 컬럼을 datetime으로 변환 후 월 시작일로 변경
    if "date" in df.columns:
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    
    # 필요한 컬럼 선택
    cols_to_keep = ["month", "fed_rate", "fed_rate_eom"]
    
    if use_derived:
        derived_cols = [
            "fed_rate_mom_diff", "fed_rate_mom_pct", "fed_rate_yoy_diff", "fed_rate_yoy_pct",
            "fed_rate_ma3", "fed_rate_ma6", "fed_rate_vol_3m", "policy_stance",
            "fed_rate_cumulative_change", "rate_level"
        ]
        cols_to_keep.extend([c for c in derived_cols if c in df.columns])
    
    result = df[[c for c in cols_to_keep if c in df.columns]].copy()
    
    # policy_stance를 숫자로 변환
    if "policy_stance" in result.columns:
        stance_map = {"cut": -1, "hold": 0, "hike": 1}
        result["policy_stance_num"] = result["policy_stance"].map(stance_map).fillna(0)
    
    # rate_level을 숫자로 변환
    if "rate_level" in result.columns:
        level_map = {"low": 0, "medium": 1, "high": 2}
        result["rate_level_num"] = result["rate_level"].map(level_map).fillna(0)
    
    return result.sort_values("month").reset_index(drop=True)


def load_usd_krw_daily(path: Union[str, Path]) -> pd.DataFrame:
    """
    USD/KRW 일별 환율 데이터 로드 및 월별 집계
    
    출력 컬럼:
        - month (datetime)
        - usd_krw_mean: 월 평균 환율
        - usd_krw_std: 월 변동성
        - usd_krw_min, usd_krw_max: 월 최저/최고
        - usd_krw_eom: 월말 환율
        - usd_krw_mom_pct: 월간 변화율
    """
    df = _safe_read_csv(path)
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    
    # 월별 집계
    monthly = df.groupby("month").agg(
        usd_krw_mean=("rate", "mean"),
        usd_krw_std=("rate", "std"),
        usd_krw_min=("rate", "min"),
        usd_krw_max=("rate", "max"),
    ).reset_index()
    
    # 월말 환율 추출
    eom = df.sort_values("date").groupby("month")["rate"].last().reset_index()
    eom.columns = ["month", "usd_krw_eom"]
    
    monthly = monthly.merge(eom, on="month", how="left")
    
    # 월간 변화율 계산
    monthly = monthly.sort_values("month")
    monthly["usd_krw_mom_pct"] = monthly["usd_krw_eom"].pct_change() * 100
    
    # 변동성 지표
    monthly["usd_krw_range"] = monthly["usd_krw_max"] - monthly["usd_krw_min"]
    monthly["usd_krw_range_pct"] = monthly["usd_krw_range"] / monthly["usd_krw_mean"] * 100
    
    return monthly.reset_index(drop=True)


def load_oil_price(path: Union[str, Path], use_derived: bool = True) -> pd.DataFrame:
    """
    원유(WTI/Brent) 가격 데이터 로드
    
    출력 컬럼:
        - month (datetime)
        - wti_price, brent_price
        - 파생변수: wti_mom_pct, wti_yoy_pct, wti_ma3, wti_vol_3m 등
    """
    df = _safe_read_csv(path)
    
    if "date" in df.columns:
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    
    cols_to_keep = ["month", "wti_price", "brent_price"]
    
    if use_derived:
        derived_cols = [
            "wti_mom_pct", "wti_yoy_pct", "wti_ma3", "wti_ma6", "wti_vol_3m",
            "brent_mom_pct", "brent_yoy_pct", "brent_ma3", "brent_vol_3m",
            "wti_brent_spread"
        ]
        cols_to_keep.extend([c for c in derived_cols if c in df.columns])
    
    result = df[[c for c in cols_to_keep if c in df.columns]].copy()
    return result.sort_values("month").reset_index(drop=True)


def load_trade_data(path: Union[str, Path], use_derived: bool = True) -> pd.DataFrame:
    """
    무역 (수출입) 데이터 로드
    
    출력 컬럼:
        - month (datetime)
        - export_amt, import_amt, trade_balance
        - 파생변수: export_mom_pct, export_yoy_pct, export_ma3 등
    """
    df = _safe_read_csv(path)
    
    # year 컬럼 처리 (2021.06 형식)
    if "year" in df.columns:
        # 문자열일 수 있음
        df["year_str"] = df["year"].astype(str)
        # "2021.06" -> "202106"
        df["ym"] = df["year_str"].str.replace(".", "").str[:6]
        df["month"] = pd.to_datetime(df["ym"], format="%Y%m", errors="coerce")
    
    # 컬럼명 매핑 (원본 파일)
    col_map = {
        "expDlr": "export_amt",
        "impDlr": "import_amt", 
        "balPayments": "trade_balance",
        "expCnt": "export_cnt",
        "impCnt": "import_cnt"
    }
    
    for old, new in col_map.items():
        if old in df.columns:
            df[new] = df[old]
    
    # 필요한 컬럼 선택
    cols_to_keep = ["month", "export_amt", "import_amt", "trade_balance", "export_cnt", "import_cnt"]
    
    if use_derived:
        derived_cols = [
            "export_mom_pct", "import_mom_pct", "balance_mom_pct",
            "export_yoy_pct", "import_yoy_pct", "balance_yoy_pct",
            "export_ma3", "import_ma3", "export_ma6", "import_ma6"
        ]
        cols_to_keep.extend([c for c in derived_cols if c in df.columns])
    
    result = df[[c for c in cols_to_keep if c in df.columns]].copy()
    return result.dropna(subset=["month"]).sort_values("month").reset_index(drop=True)


def load_cpi(path: Union[str, Path]) -> pd.DataFrame:
    """
    소비자물가지수(CPI) 데이터 로드 (멀티헤더 형식)
    
    출력 컬럼:
        - month (datetime)
        - cpi_total: 총지수
        - cpi_mom: 전월대비 변화율
        - cpi_yoy: 전년동월대비 변화율
    """
    df = _safe_read_csv(path)
    
    # 멀티헤더 처리: 첫 번째 행이 헤더일 수 있음
    if df.columns[0] in ['시점', 'date', 'ym', '년월']:
        pass  # 이미 처리됨
    else:
        # 첫 행을 헤더로 사용하거나 컬럼 재구성
        if 'Unnamed' in str(df.columns[0]) or df.iloc[0].dtype == 'object':
            new_header = df.iloc[0]
            df = df[1:]
            df.columns = new_header
            df = df.reset_index(drop=True)
    
    # 날짜 컬럼 찾기 및 변환
    date_col = None
    for col in ['시점', 'date', 'ym', '년월', 'month']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        # 시점 형식: "2022.01" 또는 "202201"
        df["month"] = pd.to_datetime(
            df[date_col].astype(str).str.replace(".", "").str[:6], 
            format="%Y%m", errors="coerce"
        )
    
    # 총지수 컬럼 찾기 (보통 '총지수' 또는 'CPI' 포함)
    cpi_col = None
    for col in df.columns:
        if '총지수' in str(col) or col == 'CPI':
            cpi_col = col
            break
    
    result = pd.DataFrame({"month": df["month"]})
    
    if cpi_col:
        result["cpi_total"] = pd.to_numeric(df[cpi_col], errors="coerce")
        result = result.sort_values("month")
        result["cpi_mom"] = result["cpi_total"].pct_change() * 100
        result["cpi_yoy"] = result["cpi_total"].pct_change(12) * 100
        result["cpi_ma3"] = result["cpi_total"].rolling(3, min_periods=1).mean()
    
    return result.dropna(subset=["month"]).reset_index(drop=True)


def load_ppi(path: Union[str, Path]) -> pd.DataFrame:
    """
    생산자물가지수(PPI) 데이터 로드
    
    출력 컬럼:
        - month (datetime)
        - ppi_total: 총지수
        - ppi_mom, ppi_yoy: 변화율
    """
    df = _safe_read_csv(path)
    
    # 날짜 컬럼 찾기
    date_col = None
    for col in ['시점', 'date', 'ym', '년월', 'month']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        df["month"] = pd.to_datetime(
            df[date_col].astype(str).str.replace(".", "").str[:6], 
            format="%Y%m", errors="coerce"
        )
    
    # 총지수 필터링 (품목명 컬럼에 '총지수'가 있는 경우)
    item_col = None
    for col in ['품목명', '품목', '항목', 'item']:
        if col in df.columns:
            item_col = col
            break
    
    if item_col:
        df = df[df[item_col].astype(str).str.contains('총지수|합계|전체', na=False)]
    
    # 지수 컬럼 찾기
    ppi_col = None
    for col in df.columns:
        if '지수' in str(col) and col not in [item_col, date_col, 'month']:
            ppi_col = col
            break
    
    if ppi_col is None:
        # 숫자 컬럼 중 첫번째 사용
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            ppi_col = numeric_cols[0]
    
    result = pd.DataFrame({"month": df["month"]})
    
    if ppi_col:
        result["ppi_total"] = pd.to_numeric(df[ppi_col], errors="coerce")
        result = result.sort_values("month").drop_duplicates(subset=["month"])
        result["ppi_mom"] = result["ppi_total"].pct_change() * 100
        result["ppi_yoy"] = result["ppi_total"].pct_change(12) * 100
    
    return result.dropna(subset=["month"]).reset_index(drop=True)


def load_employment_data(
    unemployment_path: Optional[Union[str, Path]] = None,
    employment_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    고용률/실업률 데이터 로드 (멀티헤더 형식)
    
    출력 컬럼:
        - month (datetime)
        - unemployment_rate: 실업률 (계)
        - employment_rate: 고용률 (계)
        - 파생변수: _mom, _yoy
    """
    result = None
    
    # 실업률 로드
    if unemployment_path and Path(unemployment_path).exists():
        df = _safe_read_csv(unemployment_path)
        
        # 멀티헤더 처리 (보통 "계", "남", "여" 등이 있음)
        # 날짜 컬럼 찾기
        date_col = None
        for col in ['시점', 'date', 'ym', '년월', 'month']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df["month"] = pd.to_datetime(
                df[date_col].astype(str).str.replace(".", "").str[:6], 
                format="%Y%m", errors="coerce"
            )
        
        # "계" 컬럼 찾기
        rate_col = None
        for col in df.columns:
            if '계' in str(col) or col == '실업률':
                rate_col = col
                break
        
        if rate_col:
            result = pd.DataFrame({"month": df["month"]})
            result["unemployment_rate"] = pd.to_numeric(df[rate_col], errors="coerce")
            result = result.dropna(subset=["month"])
    
    # 고용률 로드
    if employment_path and Path(employment_path).exists():
        df = _safe_read_csv(employment_path)
        
        date_col = None
        for col in ['시점', 'date', 'ym', '년월', 'month']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df["month"] = pd.to_datetime(
                df[date_col].astype(str).str.replace(".", "").str[:6], 
                format="%Y%m", errors="coerce"
            )
        
        rate_col = None
        for col in df.columns:
            if '계' in str(col) or col == '고용률':
                rate_col = col
                break
        
        if rate_col:
            emp_df = pd.DataFrame({"month": df["month"]})
            emp_df["employment_rate"] = pd.to_numeric(df[rate_col], errors="coerce")
            emp_df = emp_df.dropna(subset=["month"])
            
            if result is None:
                result = emp_df
            else:
                result = result.merge(emp_df, on="month", how="outer")
    
    if result is not None:
        result = result.sort_values("month").reset_index(drop=True)
        
        # 파생변수 생성
        if "unemployment_rate" in result.columns:
            result["unemployment_mom"] = result["unemployment_rate"].diff()
            result["unemployment_yoy"] = result["unemployment_rate"].diff(12)
        
        if "employment_rate" in result.columns:
            result["employment_mom"] = result["employment_rate"].diff()
            result["employment_yoy"] = result["employment_rate"].diff(12)
    else:
        result = pd.DataFrame(columns=["month"])
    
    return result


def load_retail_sales(path: Union[str, Path]) -> pd.DataFrame:
    """
    소매판매지수 데이터 로드 (멀티헤더 형식)
    
    출력 컬럼:
        - month (datetime)
        - retail_total: 소매판매지수 총지수
        - retail_mom, retail_yoy: 변화율
    """
    df = _safe_read_csv(path)
    
    # 날짜 컬럼 찾기
    date_col = None
    for col in ['시점', 'date', 'ym', '년월', 'month']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        df["month"] = pd.to_datetime(
            df[date_col].astype(str).str.replace(".", "").str[:6], 
            format="%Y%m", errors="coerce"
        )
    
    # 총지수 컬럼 찾기
    retail_col = None
    for col in df.columns:
        col_str = str(col).lower()
        if '총지수' in col_str or '소매' in col_str or 'retail' in col_str:
            retail_col = col
            break
    
    if retail_col is None:
        # 숫자 컬럼 중 첫번째 사용
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            retail_col = numeric_cols[0]
    
    result = pd.DataFrame({"month": df["month"]})
    
    if retail_col:
        result["retail_total"] = pd.to_numeric(df[retail_col], errors="coerce")
        result = result.sort_values("month")
        result["retail_mom"] = result["retail_total"].pct_change() * 100
        result["retail_yoy"] = result["retail_total"].pct_change(12) * 100
        result["retail_ma3"] = result["retail_total"].rolling(3, min_periods=1).mean()
    
    return result.dropna(subset=["month"]).reset_index(drop=True)


def load_industry_index(path: Union[str, Path]) -> pd.DataFrame:
    """
    산업/규모별 기업지수 데이터 로드 (연 단위, 멀티헤더 3단)
    
    연 단위 데이터를 해당 연도 12개월로 확장하여 반환
    
    출력 컬럼:
        - month (datetime)
        - industry_index_total: 산업지수 총지수
        - industry_index_manufacturing: 제조업
        - industry_index_services: 서비스업
    """
    df = _safe_read_csv(path)
    
    # 연도 컬럼 찾기
    year_col = None
    for col in ['시점', '년도', 'year', '연도']:
        if col in df.columns:
            year_col = col
            break
    
    if year_col is None:
        # 첫 컬럼이 연도일 수 있음
        year_col = df.columns[0]
    
    # 연도를 숫자로 변환
    df["year"] = pd.to_numeric(df[year_col].astype(str).str[:4], errors="coerce")
    df = df.dropna(subset=["year"])
    
    # 지수 컬럼 찾기 (멀티헤더일 경우 flatten 필요)
    # 총지수, 제조업, 서비스업 등 분류
    result_dict = {}
    
    for col in df.columns:
        col_str = str(col).lower()
        if '총지수' in col_str or '합계' in col_str or '전체' in col_str:
            result_dict["industry_index_total"] = col
        elif '제조' in col_str:
            result_dict["industry_index_manufacturing"] = col
        elif '서비스' in col_str:
            result_dict["industry_index_services"] = col
    
    if not result_dict:
        # 숫자 컬럼 중 첫번째 사용
        numeric_cols = [c for c in df.columns if c not in [year_col, "year"] and 
                       pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) > 0:
            result_dict["industry_index_total"] = numeric_cols[0]
    
    # 연도 -> 12개월 확장
    expanded_rows = []
    for _, row in df.iterrows():
        year = int(row["year"])
        for month in range(1, 13):
            month_date = pd.Timestamp(year=year, month=month, day=1)
            row_dict = {"month": month_date}
            for new_col, old_col in result_dict.items():
                row_dict[new_col] = pd.to_numeric(row[old_col], errors="coerce")
            expanded_rows.append(row_dict)
    
    result = pd.DataFrame(expanded_rows)
    return result.sort_values("month").reset_index(drop=True)


def load_trade_data_clean(path: Union[str, Path]) -> pd.DataFrame:
    """
    무역 데이터 로드 (총계 행 제거 포함)
    
    출력 컬럼:
        - month (datetime)
        - export_amt, import_amt, trade_balance
        - export_mom_pct, import_mom_pct, export_yoy_pct, import_yoy_pct
    """
    df = _safe_read_csv(path)
    
    # 총계 행 제거 (보통 마지막 행이거나 '합계', '총계' 텍스트 포함)
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        df = df[~df[col].astype(str).str.contains('합계|총계|Total|SUM', case=False, na=False)]
    
    # year/date 컬럼 처리
    if "year" in df.columns:
        # "2021.06" 형식 또는 "202106" 형식
        df["ym"] = df["year"].astype(str).str.replace(".", "").str[:6]
        df["month"] = pd.to_datetime(df["ym"], format="%Y%m", errors="coerce")
    elif "date" in df.columns:
        df["month"] = pd.to_datetime(df["date"])
    elif "ym" in df.columns:
        df["month"] = pd.to_datetime(df["ym"].astype(str), format="%Y%m", errors="coerce")
    
    # 컬럼명 매핑
    col_map = {
        "expDlr": "export_amt",
        "impDlr": "import_amt",
        "balPayments": "trade_balance",
        "수출": "export_amt",
        "수입": "import_amt",
        "무역수지": "trade_balance",
    }
    
    for old, new in col_map.items():
        if old in df.columns:
            df[new] = pd.to_numeric(df[old], errors="coerce")
    
    # 무역수지 계산 (없는 경우)
    if "trade_balance" not in df.columns and "export_amt" in df.columns and "import_amt" in df.columns:
        df["trade_balance"] = df["export_amt"] - df["import_amt"]
    
    cols = ["month", "export_amt", "import_amt", "trade_balance"]
    result = df[[c for c in cols if c in df.columns]].copy()
    result = result.dropna(subset=["month"]).sort_values("month").reset_index(drop=True)
    
    # 파생변수 생성
    if "export_amt" in result.columns:
        result["export_mom_pct"] = result["export_amt"].pct_change() * 100
        result["export_yoy_pct"] = result["export_amt"].pct_change(12) * 100
        result["export_ma3"] = result["export_amt"].rolling(3, min_periods=1).mean()
    
    if "import_amt" in result.columns:
        result["import_mom_pct"] = result["import_amt"].pct_change() * 100
        result["import_yoy_pct"] = result["import_amt"].pct_change(12) * 100
        result["import_ma3"] = result["import_amt"].rolling(3, min_periods=1).mean()
    
    return result


def load_kasi_calendar(path: Union[str, Path]) -> pd.DataFrame:
    """
    KASI 캘린더 데이터 로드 (공휴일, 영업일 등)
    
    출력 컬럼:
        - month (datetime)
        - holiday_cnt, weekend_cnt, business_cnt, total_days
        - seol_cnt, chuseok_cnt (설/추석)
        - holiday_ratio: 공휴일 비율
    """
    df = _safe_read_csv(path)
    
    if "ym" in df.columns:
        df["month"] = pd.to_datetime(df["ym"].astype(str), format="%Y%m")
    
    # 파생 변수 추가
    if "holiday_cnt" in df.columns and "total_days" in df.columns:
        df["holiday_ratio"] = df["holiday_cnt"] / df["total_days"]
    
    if "business_cnt" in df.columns and "total_days" in df.columns:
        df["business_ratio"] = df["business_cnt"] / df["total_days"]
    
    # 연휴 플래그
    if "seol_cnt" in df.columns:
        df["has_seol"] = (df["seol_cnt"] > 0).astype(int)
    if "chuseok_cnt" in df.columns:
        df["has_chuseok"] = (df["chuseok_cnt"] > 0).astype(int)
    
    cols = ["month", "holiday_cnt", "weekend_cnt", "business_cnt", "total_days",
            "seol_cnt", "chuseok_cnt", "holiday_ratio", "business_ratio",
            "has_seol", "has_chuseok"]
    
    result = df[[c for c in cols if c in df.columns]].copy()
    return result.sort_values("month").reset_index(drop=True)


def load_kr_interest_rate(path: Union[str, Path]) -> pd.DataFrame:
    """
    국내 금리 데이터 로드 (일별 -> 월별 집계)
    
    컬럼: 콜금리, KORIBOR, CD, CP, 국고채, 회사채 등
    
    출력 컬럼:
        - month (datetime)
        - call_rate: 콜금리
        - koribor_3m/6m/12m: KORIBOR
        - cd_rate, cp_rate: CD/CP 금리
        - govt_bond: 국고채
        - corp_bond_aa, corp_bond_bbb: 회사채
        - credit_spread_aa, credit_spread_bbb: 신용 스프레드
        - call_rate_mom_diff: 콜금리 월간 변동
    """
    df = _safe_read_csv(path)
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    
    # 숫자 컬럼만 선택
    rate_cols = [c for c in df.columns if c not in ["date", "month"]]
    
    # 월별 집계 (평균)
    agg_dict = {c: "mean" for c in rate_cols if df[c].dtype in ["float64", "int64", "float32", "int32"]}
    
    if agg_dict:
        monthly = df.groupby("month").agg(agg_dict).reset_index()
        
        # 컬럼명 정리 (한글 -> 영어로 변환)
        col_rename = {
            "콜금리_전체": "call_rate",
            "콜금리_중개회사거래": "call_rate_broker",
            "콜금리_은행증권금융차입": "call_rate_bank_fin",
            "KOFR": "kofr",
            "KORIBOR_3개월": "koribor_3m",
            "KORIBOR_6개월": "koribor_6m",
            "KORIBOR_12개월": "koribor_12m",
            "CD": "cd_rate",
            "CP": "cp_rate",
            "국고채": "govt_bond",
            "회사채_AA-": "corp_bond_aa",
            "회사채_BBB-": "corp_bond_bbb"
        }
        monthly = monthly.rename(columns={k: v for k, v in col_rename.items() if k in monthly.columns})
        
        # 파생 변수 생성
        monthly = monthly.sort_values("month")
        
        # 신용 스프레드 계산 (회사채 - 국고채)
        if "corp_bond_aa" in monthly.columns and "govt_bond" in monthly.columns:
            monthly["credit_spread_aa"] = monthly["corp_bond_aa"] - monthly["govt_bond"]
        if "corp_bond_bbb" in monthly.columns and "govt_bond" in monthly.columns:
            monthly["credit_spread_bbb"] = monthly["corp_bond_bbb"] - monthly["govt_bond"]
        
        # BBB-AA 스프레드 (신용 위험 프리미엄)
        if "corp_bond_bbb" in monthly.columns and "corp_bond_aa" in monthly.columns:
            monthly["credit_spread_bbb_aa"] = monthly["corp_bond_bbb"] - monthly["corp_bond_aa"]
        
        # 콜금리 월간 변동
        if "call_rate" in monthly.columns:
            monthly["call_rate_mom_diff"] = monthly["call_rate"].diff()
            monthly["call_rate_ma3"] = monthly["call_rate"].rolling(3, min_periods=1).mean()
        
        # 장단기 금리차 (term spread)
        if "koribor_12m" in monthly.columns and "koribor_3m" in monthly.columns:
            monthly["term_spread"] = monthly["koribor_12m"] - monthly["koribor_3m"]
        
        return monthly.sort_values("month").reset_index(drop=True)
    
    return pd.DataFrame(columns=["month"])


# ============================================================================
# 부동산 데이터 로더
# ============================================================================

def load_housing_data(
    data_dir: Union[str, Path],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    부동산 데이터 로드 (KOSIS 형식)
    
    파일:
    - 거래량_아파트.csv: 아파트 거래량
    - 전세지수_아파트.csv: 전세가격지수  
    - 주택가격지수_아파트.csv: 주택가격지수
    
    Returns:
        월별 전국 집계 부동산 지표
    """
    data_dir = Path(data_dir)
    result = None
    
    def _log(msg):
        if verbose:
            print(msg)
    
    def _parse_kosis_month(ym: int) -> pd.Timestamp:
        """YYYYMM 형식을 datetime으로 변환"""
        y = ym // 100
        m = ym % 100
        return pd.Timestamp(year=y, month=m, day=1)
    
    # 1. 주택가격지수
    price_path = data_dir / "주택가격지수_아파트.csv"
    if price_path.exists():
        try:
            df = _safe_read_csv(price_path)
            # 전국 데이터만 추출
            national = df[df["CLS_NM"] == "전국"].copy()
            
            if not national.empty:
                national["month"] = national["WRTTIME_IDTFR_ID"].apply(_parse_kosis_month)
                national = national.groupby("month")["DTA_VAL"].mean().reset_index()
                national.columns = ["month", "housing_price_idx"]
                
                # 전월 대비 변화율
                national = national.sort_values("month")
                national["housing_price_mom"] = national["housing_price_idx"].pct_change() * 100
                national["housing_price_yoy"] = national["housing_price_idx"].pct_change(12) * 100
                
                result = national
                _log(f"  ✓ 주택가격지수 로드 완료: {len(national)}행")
        except Exception as e:
            _log(f"  ✗ 주택가격지수 로드 실패: {e}")
    
    # 2. 전세지수
    jeonse_path = data_dir / "전세지수_아파트.csv"
    if jeonse_path.exists():
        try:
            df = _safe_read_csv(jeonse_path)
            national = df[df["CLS_NM"] == "전국"].copy()
            
            if not national.empty:
                national["month"] = national["WRTTIME_IDTFR_ID"].apply(_parse_kosis_month)
                national = national.groupby("month")["DTA_VAL"].mean().reset_index()
                national.columns = ["month", "jeonse_price_idx"]
                
                national = national.sort_values("month")
                national["jeonse_price_mom"] = national["jeonse_price_idx"].pct_change() * 100
                national["jeonse_price_yoy"] = national["jeonse_price_idx"].pct_change(12) * 100
                
                if result is not None:
                    result = result.merge(national, on="month", how="outer")
                else:
                    result = national
                _log(f"  ✓ 전세지수 로드 완료: {len(national)}행")
        except Exception as e:
            _log(f"  ✗ 전세지수 로드 실패: {e}")
    
    # 3. 아파트 거래량
    volume_path = data_dir / "거래량_아파트.csv"
    if volume_path.exists():
        try:
            df = _safe_read_csv(volume_path)
            
            # 전국 합계 계산 (지역별 데이터 합산)
            df["month"] = df["WRTTIME_IDTFR_ID"].apply(_parse_kosis_month)
            monthly = df.groupby("month")["DTA_VAL"].sum().reset_index()
            monthly.columns = ["month", "apt_transaction_vol"]
            
            monthly = monthly.sort_values("month")
            monthly["apt_vol_mom"] = monthly["apt_transaction_vol"].pct_change() * 100
            monthly["apt_vol_yoy"] = monthly["apt_transaction_vol"].pct_change(12) * 100
            # 이동평균
            monthly["apt_vol_ma3"] = monthly["apt_transaction_vol"].rolling(3, min_periods=1).mean()
            
            if result is not None:
                result = result.merge(monthly, on="month", how="outer")
            else:
                result = monthly
            _log(f"  ✓ 아파트 거래량 로드 완료: {len(monthly)}행")
        except Exception as e:
            _log(f"  ✗ 아파트 거래량 로드 실패: {e}")
    
    if result is not None:
        result = result.sort_values("month").reset_index(drop=True)
        
        # 매매가격 대비 전세가격 비율 (전세가율)
        if "housing_price_idx" in result.columns and "jeonse_price_idx" in result.columns:
            result["jeonse_to_sale_ratio"] = result["jeonse_price_idx"] / result["housing_price_idx"] * 100
        
        return result
    
    return pd.DataFrame(columns=["month"])


def load_all_external_data(
    data_dir: Union[str, Path],
    file_config: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    모든 외부 데이터를 로드하고 월별로 병합
    
    Args:
        data_dir: 외부 데이터가 있는 디렉토리 경로
        file_config: 파일명 설정 딕셔너리 (없으면 기본값 사용)
        verbose: 진행 상황 출력 여부
    
    Returns:
        월별로 병합된 외부 데이터 DataFrame
    """
    data_dir = Path(data_dir)
    
    # 기본 파일 설정 (확장)
    default_config = {
        "esi": "ESI.csv",
        "fed_rate": "fed_금리_파생변수_202106_202506.csv",
        "usd_krw": "usd_krw_raw_daily.csv",
        "oil": "원자재(유가)_파생변수.csv",
        "trade": "무역_P1_원본.csv",
        "kasi": "KASI.csv",
        "kr_rate": "P1_금리.csv",
        # 새로 추가된 P1 데이터
        "cpi": "소비자물가(CPI).csv",
        "ppi": "생산자물가(PPI).csv",
        "unemployment": "실업률.csv",
        "employment": "고용률.csv",
        "retail": "소매판매지수.csv",
        "industry": "산업,규모별 기업지수.csv",
    }
    
    if file_config:
        default_config.update(file_config)
    
    merged = None
    
    def _log(msg):
        if verbose:
            print(msg)
    
    # ESI 로드
    esi_path = data_dir / default_config["esi"]
    if esi_path.exists():
        try:
            esi = load_esi(esi_path)
            merged = esi if merged is None else merged.merge(esi, on="month", how="outer")
            _log(f"  ✓ ESI 로드 완료: {len(esi)}행")
        except Exception as e:
            _log(f"  ✗ ESI 로드 실패: {e}")
    
    # Fed 금리 로드
    fed_path = data_dir / default_config["fed_rate"]
    if fed_path.exists():
        try:
            fed = load_fed_rate(fed_path, use_derived=True)
            merged = fed if merged is None else merged.merge(fed, on="month", how="outer")
            _log(f"  ✓ Fed 금리 로드 완료: {len(fed)}행")
        except Exception as e:
            _log(f"  ✗ Fed 금리 로드 실패: {e}")
    
    # USD/KRW 환율 로드
    fx_path = data_dir / default_config["usd_krw"]
    if fx_path.exists():
        try:
            fx = load_usd_krw_daily(fx_path)
            merged = fx if merged is None else merged.merge(fx, on="month", how="outer")
            _log(f"  ✓ USD/KRW 환율 로드 완료: {len(fx)}행")
        except Exception as e:
            _log(f"  ✗ USD/KRW 환율 로드 실패: {e}")
    
    # 유가 로드
    oil_path = data_dir / default_config["oil"]
    if oil_path.exists():
        try:
            oil = load_oil_price(oil_path, use_derived=True)
            merged = oil if merged is None else merged.merge(oil, on="month", how="outer")
            _log(f"  ✓ 유가 로드 완료: {len(oil)}행")
        except Exception as e:
            _log(f"  ✗ 유가 로드 실패: {e}")
    
    # 무역 데이터 로드 (총계 제거 포함)
    trade_path = data_dir / default_config["trade"]
    if trade_path.exists():
        try:
            trade = load_trade_data_clean(trade_path)
            merged = trade if merged is None else merged.merge(trade, on="month", how="outer")
            _log(f"  ✓ 무역 데이터 로드 완료: {len(trade)}행")
        except Exception as e:
            _log(f"  ✗ 무역 데이터 로드 실패: {e}")
    
    # KASI 캘린더 로드
    kasi_path = data_dir / default_config["kasi"]
    if kasi_path.exists():
        try:
            kasi = load_kasi_calendar(kasi_path)
            merged = kasi if merged is None else merged.merge(kasi, on="month", how="outer")
            _log(f"  ✓ KASI 캘린더 로드 완료: {len(kasi)}행")
        except Exception as e:
            _log(f"  ✗ KASI 캘린더 로드 실패: {e}")
    
    # 국내 금리 로드
    kr_rate_path = data_dir / default_config["kr_rate"]
    if kr_rate_path.exists():
        try:
            kr_rate = load_kr_interest_rate(kr_rate_path)
            merged = kr_rate if merged is None else merged.merge(kr_rate, on="month", how="outer")
            _log(f"  ✓ 국내 금리 로드 완료: {len(kr_rate)}행")
        except Exception as e:
            _log(f"  ✗ 국내 금리 로드 실패: {e}")
    
    # ========== 새로 추가된 P1 데이터 ==========
    
    # CPI (소비자물가지수) 로드
    cpi_path = data_dir / default_config["cpi"]
    if cpi_path.exists():
        try:
            cpi = load_cpi(cpi_path)
            merged = cpi if merged is None else merged.merge(cpi, on="month", how="outer")
            _log(f"  ✓ CPI 로드 완료: {len(cpi)}행")
        except Exception as e:
            _log(f"  ✗ CPI 로드 실패: {e}")
    
    # PPI (생산자물가지수) 로드
    ppi_path = data_dir / default_config["ppi"]
    if ppi_path.exists():
        try:
            ppi = load_ppi(ppi_path)
            merged = ppi if merged is None else merged.merge(ppi, on="month", how="outer")
            _log(f"  ✓ PPI 로드 완료: {len(ppi)}행")
        except Exception as e:
            _log(f"  ✗ PPI 로드 실패: {e}")
    
    # 고용/실업률 로드
    unemp_path = data_dir / default_config["unemployment"]
    emp_path = data_dir / default_config["employment"]
    if unemp_path.exists() or emp_path.exists():
        try:
            employment = load_employment_data(
                unemployment_path=unemp_path if unemp_path.exists() else None,
                employment_path=emp_path if emp_path.exists() else None,
            )
            if not employment.empty:
                merged = employment if merged is None else merged.merge(employment, on="month", how="outer")
                _log(f"  ✓ 고용 데이터 로드 완료: {len(employment)}행")
        except Exception as e:
            _log(f"  ✗ 고용 데이터 로드 실패: {e}")
    
    # 소매판매지수 로드
    retail_path = data_dir / default_config["retail"]
    if retail_path.exists():
        try:
            retail = load_retail_sales(retail_path)
            merged = retail if merged is None else merged.merge(retail, on="month", how="outer")
            _log(f"  ✓ 소매판매지수 로드 완료: {len(retail)}행")
        except Exception as e:
            _log(f"  ✗ 소매판매지수 로드 실패: {e}")
    
    # 산업/규모별 기업지수 로드 (연->월 확장)
    industry_path = data_dir / default_config["industry"]
    if industry_path.exists():
        try:
            industry = load_industry_index(industry_path)
            merged = industry if merged is None else merged.merge(industry, on="month", how="outer")
            _log(f"  ✓ 산업지수 로드 완료: {len(industry)}행")
        except Exception as e:
            _log(f"  ✗ 산업지수 로드 실패: {e}")
    
    # ========== 부동산 데이터 (신규 추가) ==========
    try:
        housing = load_housing_data(data_dir, verbose=False)
        if housing is not None and not housing.empty:
            merged = housing if merged is None else merged.merge(housing, on="month", how="outer")
            _log(f"  ✓ 부동산 데이터 로드 완료: {len(housing)}행")
    except Exception as e:
        _log(f"  ✗ 부동산 데이터 로드 실패: {e}")
    
    if merged is not None:
        merged = merged.sort_values("month").reset_index(drop=True)
        _log(f"\n  ★ 외부 데이터 병합 완료: {len(merged)}행, {len(merged.columns)}개 컬럼")
    
    return merged


def merge_external_to_panel(
    panel: pd.DataFrame,
    external: pd.DataFrame,
    month_col: str = "month"
) -> pd.DataFrame:
    """
    패널 데이터에 외부 데이터 병합
    
    외부 데이터는 월별로 존재하므로 패널의 각 행에 해당 월의 외부 데이터를 조인
    
    Args:
        panel: 세그먼트-월 패널 데이터
        external: 월별 외부 데이터
        month_col: 월 컬럼명
    
    Returns:
        외부 데이터가 병합된 패널
    """
    if external is None or external.empty:
        return panel
    
    # 월 컬럼 타입 통일
    panel = panel.copy()
    external = external.copy()
    
    panel[month_col] = pd.to_datetime(panel[month_col])
    external[month_col] = pd.to_datetime(external[month_col])
    
    # 병합
    result = panel.merge(external, on=month_col, how="left")
    
    # 외부 데이터가 없는 월에 대해 이전 값으로 채우기 (forward fill)
    external_cols = [c for c in external.columns if c != month_col]
    for col in external_cols:
        if col in result.columns:
            result[col] = result.groupby("segment_id")[col].ffill()
    
    return result


def add_macro_lag_features(
    df: pd.DataFrame,
    macro_cols: Optional[List[str]] = None,
    lags: List[int] = [1, 2, 3],
) -> pd.DataFrame:
    """
    거시경제 변수에 대한 Lag 피처 생성
    
    거시경제 변수는 세그먼트와 무관하게 월별로 동일하므로
    단순 시프트로 Lag 생성
    
    Args:
        df: 패널 데이터
        macro_cols: 거시경제 변수 컬럼 리스트 (None이면 자동 감지)
        lags: Lag 기간 리스트
    
    Returns:
        Lag 피처가 추가된 DataFrame
    """
    if macro_cols is None:
        # 자동 감지: 모든 세그먼트에서 동일한 값을 가진 컬럼
        macro_cols = [
            "ESI", "fed_rate", "usd_krw_mean", "usd_krw_eom", "usd_krw_mom_pct",
            "wti_price", "brent_price", "wti_mom_pct", 
            "export_amt", "import_amt", "trade_balance",
            "holiday_cnt", "business_cnt", "has_seol", "has_chuseok",
            "call_rate", "koribor_3m", "govt_bond", "corp_bond_aa",
            "credit_spread_aa", "term_spread"
        ]
    
    out = df.copy()
    
    # 월별 유니크 데이터로 Lag 계산 후 다시 조인
    month_unique = out[["month"]].drop_duplicates().sort_values("month")
    
    for col in macro_cols:
        if col not in out.columns:
            continue
        
        # 월별 첫 번째 값 추출 (동일해야 함)
        monthly_val = out.groupby("month")[col].first().reset_index()
        monthly_val = monthly_val.sort_values("month")
        
        for lag in lags:
            monthly_val[f"{col}__lag{lag}"] = monthly_val[col].shift(lag)
        
        # 원래 데이터에 조인 (원본 컬럼 제외)
        lag_cols = [c for c in monthly_val.columns if "__lag" in c]
        out = out.merge(monthly_val[["month"] + lag_cols], on="month", how="left")
    
    return out


def fill_missing_values(df: pd.DataFrame, method: str = "smart") -> pd.DataFrame:
    """
    결측치 보완 함수
    
    Args:
        df: 외부 데이터 DataFrame
        method: 보완 방법
            - "smart": 변수 특성에 따라 적절한 방법 적용
            - "ffill": 이전 값으로 채우기
            - "interpolate": 선형 보간
    
    Returns:
        결측치가 보완된 DataFrame
    """
    result = df.copy()
    result = result.sort_values("month")
    
    if method == "smart":
        # YoY 변수들: 처음 12개월은 계산 불가하므로 NaN 유지하거나 0으로
        yoy_cols = [c for c in result.columns if "yoy" in c.lower()]
        for col in yoy_cols:
            # 12개월 이후부터는 이전 값으로 채움
            result[col] = result[col].ffill()
        
        # MA 변수들: 앞부분만 결측이므로 ffill
        ma_cols = [c for c in result.columns if "ma3" in c.lower() or "ma6" in c.lower()]
        for col in ma_cols:
            result[col] = result[col].ffill()
        
        # MoM 변수들: 첫 번째만 결측
        mom_cols = [c for c in result.columns if "mom" in c.lower()]
        for col in mom_cols:
            result[col] = result[col].fillna(0)
        
        # 금리 및 가격 데이터: 선형 보간
        price_cols = [c for c in result.columns if any(x in c for x in 
                      ["rate", "price", "bond", "spread", "koribor", "cd_", "cp_", "call_"])]
        for col in price_cols:
            if col in result.columns and result[col].dtype in ["float64", "int64"]:
                result[col] = result[col].interpolate(method='linear', limit_direction='both')
        
        # 남은 숫자형 결측치는 ffill + bfill
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            result[col] = result[col].ffill().bfill()
    
    elif method == "ffill":
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            result[col] = result[col].ffill()
    
    elif method == "interpolate":
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            result[col] = result[col].interpolate(method='linear', limit_direction='both')
    
    return result


def validate_external_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    외부 데이터 품질 검증
    
    Returns:
        검증 결과 딕셔너리
    """
    result = {
        "is_valid": True,
        "warnings": [],
        "errors": [],
        "stats": {}
    }
    
    # 1. 날짜 연속성 확인
    if "month" in df.columns:
        months = pd.to_datetime(df["month"]).sort_values()
        expected_months = pd.date_range(months.min(), months.max(), freq="MS")
        missing_months = set(expected_months) - set(months)
        if missing_months:
            result["warnings"].append(f"누락된 월: {sorted(missing_months)}")
    
    # 2. 결측치 비율 확인
    missing_pct = df.isnull().sum() / len(df) * 100
    high_missing = missing_pct[missing_pct > 30]
    if len(high_missing) > 0:
        result["warnings"].append(f"결측치 30% 초과 컬럼: {dict(high_missing)}")
    
    # 3. 이상치 확인 (3-sigma rule)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_info = {}
    for col in numeric_cols:
        if col == "month":
            continue
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 0:
            outliers = ((df[col] - mean_val).abs() > 3 * std_val).sum()
            if outliers > 0:
                outlier_info[col] = outliers
    if outlier_info:
        result["warnings"].append(f"이상치 포함 컬럼: {outlier_info}")
    
    # 4. 기본 통계
    result["stats"] = {
        "row_count": len(df),
        "col_count": len(df.columns),
        "date_range": f"{df['month'].min()} ~ {df['month'].max()}" if "month" in df.columns else "N/A",
        "total_missing": df.isnull().sum().sum(),
        "missing_pct": df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    }
    
    # 5. 필수 컬럼 확인
    required_cols = ["month", "ESI", "fed_rate", "usd_krw_mean"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        result["errors"].append(f"필수 컬럼 누락: {missing_required}")
        result["is_valid"] = False
    
    return result


def get_external_feature_groups() -> Dict[str, List[str]]:
    """
    외부 변수들을 카테고리별로 그룹화하여 반환
    
    모델 학습 시 feature selection에 활용 가능
    """
    return {
        "sentiment": [
            "ESI",
        ],
        "fed_rate": [
            "fed_rate", "fed_rate_eom", "fed_rate_mom_diff", "fed_rate_mom_pct",
            "fed_rate_yoy_diff", "fed_rate_yoy_pct", "fed_rate_ma3", "fed_rate_ma6",
            "fed_rate_vol_3m", "policy_stance_num", "rate_level_num",
            "fed_rate_cumulative_change"
        ],
        "forex": [
            "usd_krw_mean", "usd_krw_std", "usd_krw_min", "usd_krw_max",
            "usd_krw_eom", "usd_krw_mom_pct", "usd_krw_range", "usd_krw_range_pct"
        ],
        "oil": [
            "wti_price", "brent_price", "wti_mom_pct", "wti_yoy_pct",
            "wti_ma3", "wti_ma6", "wti_vol_3m", "brent_mom_pct", "brent_yoy_pct",
            "brent_ma3", "brent_vol_3m", "wti_brent_spread"
        ],
        "trade": [
            "export_amt", "import_amt", "trade_balance", "export_cnt", "import_cnt",
            "export_mom_pct", "import_mom_pct", "balance_mom_pct",
            "export_yoy_pct", "import_yoy_pct", "balance_yoy_pct",
            "export_ma3", "import_ma3", "export_ma6", "import_ma6"
        ],
        "calendar": [
            "holiday_cnt", "weekend_cnt", "business_cnt", "total_days",
            "seol_cnt", "chuseok_cnt", "holiday_ratio", "business_ratio",
            "has_seol", "has_chuseok"
        ],
        "kr_rate": [
            "call_rate", "call_rate_broker", "call_rate_bank_fin", "kofr",
            "koribor_3m", "koribor_6m", "koribor_12m", "cd_rate", "cp_rate",
            "govt_bond", "corp_bond_aa", "corp_bond_bbb",
            "credit_spread_aa", "credit_spread_bbb", "credit_spread_bbb_aa",
            "call_rate_mom_diff", "call_rate_ma3", "term_spread"
        ],
        # 신규 추가 그룹들
        "cpi": [
            "cpi_total", "cpi_mom", "cpi_yoy", "cpi_ma3"
        ],
        "ppi": [
            "ppi_total", "ppi_mom", "ppi_yoy"
        ],
        "employment": [
            "unemployment_rate", "unemployment_mom", "unemployment_yoy",
            "employment_rate", "employment_mom", "employment_yoy"
        ],
        "retail": [
            "retail_total", "retail_mom", "retail_yoy", "retail_ma3"
        ],
        "industry": [
            "industry_index_total", "industry_index_manufacturing", "industry_index_services"
        ]
    }


def create_composite_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    외부 변수들을 조합한 복합 지표 생성
    
    금융 시장 분석에 유용한 복합 지표들을 계산
    """
    result = df.copy()
    
    # 1. 통화정책 방향성 지수 (Fed + 국내 콜금리 통합)
    if "fed_rate_mom_diff" in result.columns and "call_rate_mom_diff" in result.columns:
        result["monetary_policy_direction"] = (
            result["fed_rate_mom_diff"].fillna(0) + 
            result["call_rate_mom_diff"].fillna(0)
        )
    
    # 2. 금리차 지표 (한미 금리차)
    if "call_rate" in result.columns and "fed_rate" in result.columns:
        result["kr_us_rate_diff"] = result["call_rate"] - result["fed_rate"]
    
    # 3. 위험 선호 지표 (신용스프레드 + 환율 변동성)
    if "credit_spread_aa" in result.columns and "usd_krw_std" in result.columns:
        # 표준화 후 합산
        spread_norm = (result["credit_spread_aa"] - result["credit_spread_aa"].mean()) / result["credit_spread_aa"].std()
        vol_norm = (result["usd_krw_std"] - result["usd_krw_std"].mean()) / result["usd_krw_std"].std()
        result["risk_aversion_index"] = spread_norm.fillna(0) + vol_norm.fillna(0)
    
    # 4. 무역 모멘텀
    if "export_mom_pct" in result.columns and "import_mom_pct" in result.columns:
        result["trade_momentum"] = result["export_mom_pct"].fillna(0) - result["import_mom_pct"].fillna(0)
    
    # 5. 에너지 가격 압력
    if "wti_mom_pct" in result.columns and "brent_mom_pct" in result.columns:
        result["energy_price_pressure"] = (
            result["wti_mom_pct"].fillna(0) + result["brent_mom_pct"].fillna(0)
        ) / 2
    
    # 6. 경기 사이클 지표 (ESI + 무역수지 + 금리)
    if all(c in result.columns for c in ["ESI", "trade_balance", "call_rate"]):
        esi_norm = (result["ESI"] - result["ESI"].mean()) / result["ESI"].std()
        balance_norm = (result["trade_balance"] - result["trade_balance"].mean()) / result["trade_balance"].std()
        result["business_cycle_index"] = esi_norm.fillna(0) + balance_norm.fillna(0)
    
    # 7. FX 방향성 (환율 모멘텀 + 수출입 밸런스)
    if "usd_krw_mom_pct" in result.columns and "trade_balance" in result.columns:
        result["fx_pressure"] = result["usd_krw_mom_pct"].fillna(0)
    
    return result


# 테스트 코드
if __name__ == "__main__":
    from pathlib import Path
    
    data_dir = Path(r"C:\Users\campus4D052\Desktop\최종 프로젝트\외부 데이터")
    
    print("=" * 60)
    print("외부 데이터 로더 종합 테스트")
    print("=" * 60)
    
    # 전체 로드
    all_data = load_all_external_data(data_dir)
    
    if all_data is not None:
        # 검증
        print("\n[데이터 검증]")
        validation = validate_external_data(all_data)
        print(f"  유효성: {validation['is_valid']}")
        print(f"  통계: {validation['stats']}")
        if validation['warnings']:
            print(f"  경고: {validation['warnings']}")
        if validation['errors']:
            print(f"  오류: {validation['errors']}")
        
        # 결측치 보완
        print("\n[결측치 보완]")
        filled = fill_missing_values(all_data, method="smart")
        missing_before = all_data.isnull().sum().sum()
        missing_after = filled.isnull().sum().sum()
        print(f"  보완 전 결측치: {missing_before}")
        print(f"  보완 후 결측치: {missing_after}")
        
        # 복합 지표 생성
        print("\n[복합 지표 생성]")
        enhanced = create_composite_indicators(filled)
        new_cols = [c for c in enhanced.columns if c not in filled.columns]
        print(f"  생성된 복합 지표: {new_cols}")
        
        # Feature 그룹
        print("\n[Feature 그룹]")
        groups = get_external_feature_groups()
        for group_name, cols in groups.items():
            available = [c for c in cols if c in enhanced.columns]
            print(f"  {group_name}: {len(available)}/{len(cols)}개 사용 가능")

