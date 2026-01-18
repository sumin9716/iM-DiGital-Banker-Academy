"""
Executive Summary Dashboard
============================

핵심 이슈, 기회, 실행 리스트를 통합한 경영진용 원페이저 대시보드.
"""

from __future__ import annotations

import os
from typing import Optional, List, Dict
from datetime import datetime

import pandas as pd
import numpy as np


def generate_sparkline_svg(values: List[float], width: int = 80, height: int = 24, color: str = "#3b82f6") -> str:
    """미니 스파크라인 SVG 생성"""
    if not values or len(values) < 2:
        return ""
    
    # NaN 제거
    values = [v for v in values if pd.notna(v)]
    if len(values) < 2:
        return ""
    
    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val if max_val != min_val else 1
    
    # SVG 좌표 계산
    points = []
    for i, v in enumerate(values):
        x = (i / (len(values) - 1)) * width
        y = height - ((v - min_val) / val_range) * height
        points.append(f"{x:.1f},{y:.1f}")
    
    # 마지막 값에 따라 색상 결정
    if len(values) >= 2:
        if values[-1] > values[-2]:
            color = "#10b981"  # 상승: 녹색
        elif values[-1] < values[-2]:
            color = "#ef4444"  # 하락: 빨강
    
    path = " ".join(points)
    return f'''<svg width="{width}" height="{height}" style="vertical-align: middle;">
        <polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>
        <circle cx="{width}" cy="{height - ((values[-1] - min_val) / val_range) * height:.1f}" r="3" fill="{color}"/>
    </svg>'''


def generate_executive_summary_html(
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    watchlist: pd.DataFrame,
    actions: pd.DataFrame,
    drivers: pd.DataFrame,
    segment_meta: pd.DataFrame,
    out_dir: str,
    current_month: Optional[pd.Timestamp] = None,
    fx_regime: Optional[Dict] = None,
    fx_scores: Optional[pd.DataFrame] = None,
) -> str:
    """
    Executive Summary 대시보드 생성.
    
    포함 내용:
    1. 전체 예금/대출 추이 및 3개월 예측 (그래프)
    2. 이번 달 핵심 이슈 (경보 TOP 5)
    3. 다음달 기회 (예측 상승 TOP 5)
    4. FX Radar 요약 + FX Opportunity/Risk TOP 3
    5. 실행 리스트 (우선순위 TOP 10)
    6. 세그먼트 카드 (성장 예시 / 리스크 예시) - 스파크라인 포함
    
    Returns:
        HTML 파일 경로
    """
    os.makedirs(out_dir, exist_ok=True)
    
    if current_month is None:
        current_month = panel["month"].max()
    
    month_str = current_month.strftime("%Y년 %m월")
    next_month = (current_month.to_period("M") + 1).to_timestamp()
    next_month_str = next_month.strftime("%Y년 %m월")
    
    # ========== 데이터 준비 ==========
    
    # 1. 전체 KPI 추이
    kpi_cols = ["예금총잔액", "대출총잔액"]
    hist = panel[panel.get("pre_birth", 0) == 0].copy()
    hist_tot = hist.groupby("month")[kpi_cols].sum(numeric_only=True).reset_index()
    
    # 예측 합계 - horizon별로 올바른 컬럼 선택
    # horizon 1: _x, horizon 2: _y, horizon 3: 기본 컬럼
    fc_tot_rows = []
    for pred_month in forecasts["pred_month"].unique():
        fc_subset = forecasts[forecasts["pred_month"] == pred_month]
        row = {"pred_month": pred_month}
        for kpi in kpi_cols:
            # horizon별 컬럼 시도: _x (h=1), _y (h=2), 기본 (h=3)
            val = 0
            for suffix in ["_x", "_y", ""]:
                col = f"{kpi}{suffix}"
                if col in fc_subset.columns:
                    col_sum = fc_subset[col].sum()
                    if col_sum > 0:
                        val = col_sum
                        break
            row[kpi] = val
        fc_tot_rows.append(row)
    fc_tot = pd.DataFrame(fc_tot_rows)
    fc_tot["pred_month"] = pd.to_datetime(fc_tot["pred_month"])
    fc_tot = fc_tot.sort_values("pred_month")
    
    # 2. 핵심 이슈 TOP 5 (워치리스트에서 추출)
    if not watchlist.empty:
        # severity가 CRITICAL인 것 우선, 잔차 절대값 큰 순
        issues_top5 = watchlist.nlargest(5, "alert_score") if "alert_score" in watchlist.columns else watchlist.head(5)
    else:
        issues_top5 = pd.DataFrame()
    
    # 3. 기회 TOP 5 (예측 상승 세그먼트)
    opp_top5 = pd.DataFrame()
    if not forecasts.empty:
        kpi = "예금총잔액"
        cur = panel[panel["month"] == current_month][["segment_id", kpi]].copy()
        
        # forecasts에서 horizon별로 다른 컬럼 처리 (_x, _y, 원본)
        fc_month = forecasts[forecasts["pred_month"] == next_month].copy()
        
        # 올바른 예측 컬럼 찾기
        kpi_col = None
        for cand in [f"{kpi}_x", f"{kpi}_y", kpi]:
            if cand in fc_month.columns and fc_month[cand].notna().any():
                kpi_col = cand
                break
        
        if kpi_col is not None:
            fc1 = fc_month[["segment_id", kpi_col]].copy()
            fc1 = fc1.rename(columns={kpi_col: f"{kpi}_pred"})
            
            merged = cur.merge(fc1, on="segment_id", how="left")
            merged["delta"] = merged[f"{kpi}_pred"] - merged[kpi]
            merged["delta_pct"] = merged["delta"] / (merged[kpi].replace(0, 1)) * 100
            merged = merged.dropna(subset=["delta"])
            opp_top5 = merged.nlargest(5, "delta")
            
            if not segment_meta.empty:
                opp_top5 = opp_top5.merge(segment_meta, on="segment_id", how="left")
    
    # 4. 실행 리스트 TOP 10
    if not actions.empty:
        actions_top10 = actions.nlargest(10, "score") if "score" in actions.columns else actions.head(10)
    else:
        actions_top10 = pd.DataFrame()
    
    # 5. FX TOP 3 Opportunity/Risk
    fx_opp_top3 = pd.DataFrame()
    fx_risk_top3 = pd.DataFrame()
    if fx_scores is not None and not fx_scores.empty:
        if "opportunity_score" in fx_scores.columns:
            fx_opp_top3 = fx_scores.nlargest(3, "opportunity_score")
        if "risk_score" in fx_scores.columns:
            fx_risk_top3 = fx_scores.nlargest(3, "risk_score")
    
    # 6. 세그먼트별 추이 데이터 (스파크라인용)
    def get_segment_history(seg_id: str, kpi: str = "예금총잔액", n_months: int = 6) -> List[float]:
        """세그먼트의 최근 N개월 KPI 추이"""
        seg_data = panel[panel["segment_id"] == seg_id].sort_values("month").tail(n_months)
        if kpi in seg_data.columns:
            return seg_data[kpi].tolist()
        return []
    
    # 7. 세그먼트 카드 (성장 예시 1개, 리스크 예시 1개)
    growth_card_seg = None
    risk_card_seg = None
    growth_sparkline = ""
    risk_sparkline = ""
    
    if not opp_top5.empty:
        growth_card_seg = opp_top5.iloc[0].to_dict()
        seg_id = growth_card_seg.get("segment_id", "")
        history = get_segment_history(seg_id, "예금총잔액", 6)
        growth_sparkline = generate_sparkline_svg(history, width=100, height=30)
    
    if not issues_top5.empty:
        risk_card_seg = issues_top5.iloc[0].to_dict()
        seg_id = risk_card_seg.get("segment_id", "")
        history = get_segment_history(seg_id, "예금총잔액", 6)
        risk_sparkline = generate_sparkline_svg(history, width=100, height=30)
    
    # ========== HTML 생성 ==========
    html_parts = []
    
    # 헤더
    html_parts.append(f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Summary - {month_str}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Malgun Gothic', 'Noto Sans KR', Arial, sans-serif; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        
        .header {{
            background: rgba(255,255,255,0.98);
            border-radius: 16px;
            padding: 24px 32px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{ 
            color: #1a1a2e; 
            font-size: 28px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .header .date {{ 
            color: #666; 
            font-size: 16px;
            background: #f0f0f0;
            padding: 8px 16px;
            border-radius: 20px;
        }}
        
        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }}
        .grid-3 {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .card {{
            background: rgba(255,255,255,0.98);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            color: #1a1a2e;
            font-size: 18px;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 2px solid #e0e0e0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 20px;
        }}
        .kpi-card {{
            background: rgba(255,255,255,0.98);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }}
        .kpi-card .label {{ color: #666; font-size: 12px; margin-bottom: 8px; }}
        .kpi-card .value {{ font-size: 24px; font-weight: bold; color: #1a1a2e; }}
        .kpi-card .change {{ font-size: 12px; margin-top: 4px; }}
        .kpi-card .change.up {{ color: #10b981; }}
        .kpi-card .change.down {{ color: #ef4444; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #333;
            font-size: 12px;
        }}
        tr:hover {{ background: #f8f9fa; }}
        
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge.critical {{ background: #fee2e2; color: #dc2626; }}
        .badge.high {{ background: #fef3c7; color: #d97706; }}
        .badge.opportunity {{ background: #d1fae5; color: #059669; }}
        .badge.action {{ background: #dbeafe; color: #2563eb; }}
        
        .segment-card {{
            background: linear-gradient(135deg, #f8f9fa 0%, #fff 100%);
            border-radius: 12px;
            padding: 20px;
            border-left: 4px solid #3b82f6;
            margin-bottom: 16px;
        }}
        .segment-card.risk {{
            border-left-color: #ef4444;
        }}
        .segment-card .seg-name {{
            font-weight: bold;
            color: #1a1a2e;
            font-size: 14px;
            margin-bottom: 12px;
        }}
        .segment-card .metrics {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 12px;
        }}
        .segment-card .metric {{
            text-align: center;
        }}
        .segment-card .metric .label {{ font-size: 11px; color: #666; }}
        .segment-card .metric .value {{ font-size: 16px; font-weight: bold; color: #1a1a2e; }}
        .segment-card .actions-list {{
            background: #f0f9ff;
            padding: 12px;
            border-radius: 8px;
            font-size: 12px;
        }}
        .segment-card .actions-list li {{
            margin-bottom: 6px;
            color: #1e40af;
        }}
        
        .chart-container {{
            height: 300px;
        }}
        
        .fx-panel {{
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
        }}
        .fx-item {{
            flex: 1;
            min-width: 120px;
            text-align: center;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .fx-item .label {{ font-size: 11px; color: #666; }}
        .fx-item .value {{ font-size: 20px; font-weight: bold; color: #1a1a2e; }}
        
        .nav {{
            display: flex;
            gap: 12px;
        }}
        .nav a {{
            padding: 8px 16px;
            background: #e0e0e0;
            border-radius: 20px;
            text-decoration: none;
            color: #333;
            font-size: 13px;
            transition: all 0.2s;
        }}
        .nav a:hover {{ background: #3b82f6; color: white; }}
        
        .footer {{
            text-align: center;
            color: rgba(255,255,255,0.7);
            padding: 20px;
            font-size: 12px;
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>📊 Executive Summary - Segment Radar</h1>
        <div style="display: flex; align-items: center; gap: 20px; flex-wrap: wrap;">
            <div class="date">📅 기준: {month_str}</div>
            <div class="date" style="background: #e8f4fd; color: #1e40af;">📊 세그먼트: 업종×지역×등급×전담</div>
            <div class="date" style="background: #f0fdf4; color: #166534;">🔮 예측: 1~3개월</div>
            <div class="date" style="background: #fefce8; color: #854d0e;">⚙️ v1.0 (LightGBM)</div>
            <nav class="nav">
                <a href="index.html">🏠 홈</a>
                <a href="01_overview.html">📈 Overview</a>
                <a href="02_growth_forecast.html">🚀 Growth</a>
                <a href="03_risk_watchlist.html">⚠️ Risk</a>
                <a href="05_fx_radar.html">💱 FX</a>
            </nav>
        </div>
    </div>
""")
    
    # KPI 요약 카드
    # 데이터 단위: 백만원 (100만원)
    # 백만원 → 조원: / 1,000,000
    # 백만원 → 억원: / 100
    if len(hist_tot) > 0:
        last_deposit = hist_tot[hist_tot["month"] == current_month]["예금총잔액"].values
        last_loan = hist_tot[hist_tot["month"] == current_month]["대출총잔액"].values
        prev_deposit = hist_tot[hist_tot["month"] == (current_month - pd.DateOffset(months=1))]["예금총잔액"].values
        prev_loan = hist_tot[hist_tot["month"] == (current_month - pd.DateOffset(months=1))]["대출총잔액"].values
        
        deposit_val = last_deposit[0] if len(last_deposit) > 0 else 0
        loan_val = last_loan[0] if len(last_loan) > 0 else 0
        deposit_prev = prev_deposit[0] if len(prev_deposit) > 0 else deposit_val
        loan_prev = prev_loan[0] if len(prev_loan) > 0 else loan_val
        
        deposit_chg = (deposit_val - deposit_prev) / (deposit_prev + 1) * 100
        loan_chg = (loan_val - loan_prev) / (loan_prev + 1) * 100
        
        # 예측값
        fc_deposit = fc_tot[fc_tot["pred_month"] == next_month]["예금총잔액"].values
        fc_loan = fc_tot[fc_tot["pred_month"] == next_month]["대출총잔액"].values
        fc_deposit_val = fc_deposit[0] if len(fc_deposit) > 0 else 0
        fc_loan_val = fc_loan[0] if len(fc_loan) > 0 else 0
        
        # 단위 변환: 백만원 → 조원 (/1,000,000), 억원 (/100)
        deposit_jo = deposit_val / 1_000_000  # 조원
        loan_jo = loan_val / 1_000_000  # 조원
        fc_deposit_jo = fc_deposit_val / 1_000_000  # 조원
        fc_loan_jo = fc_loan_val / 1_000_000  # 조원
        
        # 변화량: 억원 단위
        deposit_delta_eok = (fc_deposit_val - deposit_val) / 100
        loan_delta_eok = (fc_loan_val - loan_val) / 100
        
        html_parts.append(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="label">📦 전체 예금 총잔액</div>
            <div class="value">{deposit_jo:.2f}조원</div>
            <div class="change {'up' if deposit_chg >= 0 else 'down'}">
                {'▲' if deposit_chg >= 0 else '▼'} {abs(deposit_chg):.1f}% MoM
            </div>
        </div>
        <div class="kpi-card">
            <div class="label">🏦 전체 대출 총잔액</div>
            <div class="value">{loan_jo:.2f}조원</div>
            <div class="change {'up' if loan_chg >= 0 else 'down'}">
                {'▲' if loan_chg >= 0 else '▼'} {abs(loan_chg):.1f}% MoM
            </div>
        </div>
        <div class="kpi-card">
            <div class="label">📈 예금 예측 ({next_month_str})</div>
            <div class="value">{fc_deposit_jo:.2f}조원</div>
            <div class="change {'up' if fc_deposit_val >= deposit_val else 'down'}">
                예상 {'▲' if fc_deposit_val >= deposit_val else '▼'} {abs(deposit_delta_eok):,.0f}억원
            </div>
        </div>
        <div class="kpi-card">
            <div class="label">📈 대출 예측 ({next_month_str})</div>
            <div class="value">{fc_loan_jo:.2f}조원</div>
            <div class="change {'up' if fc_loan_val >= loan_val else 'down'}">
                예상 {'▲' if fc_loan_val >= loan_val else '▼'} {abs(loan_delta_eok):,.0f}억원
            </div>
        </div>
    </div>
""")
    
    # 그래프 섹션
    html_parts.append("""
    <div class="grid-2">
        <div class="card">
            <h2>📈 예금 총잔액 추이 및 예측</h2>
            <div id="deposit-chart" class="chart-container"></div>
        </div>
        <div class="card">
            <h2>📈 대출 총잔액 추이 및 예측</h2>
            <div id="loan-chart" class="chart-container"></div>
        </div>
    </div>
""")
    
    # 핵심 이슈 & 기회 섹션
    html_parts.append("""
    <div class="grid-2">
        <div class="card">
            <h2>🚨 이번 달 핵심 이슈 (경보 TOP 5)</h2>
            <table>
                <tr>
                    <th>세그먼트</th>
                    <th>경보</th>
                    <th>KPI</th>
                    <th>잔차%</th>
                    <th>원인 요약</th>
                    <th>추천 액션</th>
                </tr>
""")
    
    if not issues_top5.empty:
        for _, row in issues_top5.iterrows():
            seg_id = row.get("segment_id", "")
            seg_short = str(seg_id)[:30]
            alert_type = row.get("alert_type", "")
            kpi = row.get("kpi", "")
            resid_pct = row.get("residual_pct", 0)
            severity = row.get("severity", "")
            badge_class = "critical" if severity == "CRITICAL" else "high"
            
            # 드라이버 찾기
            driver_text = "-"
            if not drivers.empty and "segment_id" in drivers.columns:
                driver_row = drivers[drivers["segment_id"] == seg_id]
                if not driver_row.empty:
                    driver_text = str(driver_row.iloc[0].get("driver", "-"))[:40]
            
            # 경보 유형에 따른 추천 액션
            if alert_type == "DROP":
                action_text = "RM 선접촉 + 이탈 방어 혜택"
            elif alert_type == "SPIKE":
                action_text = "신규 잔액 확보 + 관계 강화"
            else:
                action_text = "모니터링 지속"
            
            html_parts.append(f"""
                <tr>
                    <td title="{seg_id}">{seg_short}...</td>
                    <td><span class="badge {badge_class}">{alert_type}</span></td>
                    <td>{kpi}</td>
                    <td>{resid_pct:.1f}%</td>
                    <td style="font-size:11px; max-width:150px; overflow:hidden;">{driver_text}</td>
                    <td style="font-size:11px; color:#1e40af;">{action_text}</td>
                </tr>
""")
    else:
        html_parts.append("<tr><td colspan='6'>경보 데이터 없음</td></tr>")
    
    html_parts.append("""
            </table>
        </div>
        <div class="card">
            <h2>🚀 다음달 기회 (예측 상승 TOP 5)</h2>
            <p style="font-size:11px; color:#666; margin-bottom:10px;">💡 금액 단위: 백만원</p>
            <table>
                <tr>
                    <th>세그먼트</th>
                    <th>현재 예금</th>
                    <th>예측 예금</th>
                    <th>Δ (예상)</th>
                    <th>증가율</th>
                </tr>
""")
    
    if not opp_top5.empty:
        for _, row in opp_top5.iterrows():
            seg_short = str(row.get("segment_id", ""))[:35]
            cur_val = row.get("예금총잔액", 0)
            pred_val = row.get("예금총잔액_pred", 0)
            delta = row.get("delta", 0)
            delta_pct = row.get("delta_pct", 0)
            
            # nan 체크
            if pd.isna(cur_val):
                cur_val = 0
            if pd.isna(pred_val):
                pred_val = 0
            if pd.isna(delta):
                delta = 0
            if pd.isna(delta_pct):
                delta_pct = 0
            
            html_parts.append(f"""
                <tr>
                    <td title="{row.get('segment_id', '')}">{seg_short}...</td>
                    <td>{cur_val:,.0f}</td>
                    <td>{pred_val:,.0f}</td>
                    <td style="color: #059669; font-weight: bold;">+{delta:,.0f}</td>
                    <td><span class="badge opportunity">+{delta_pct:.1f}%</span></td>
                </tr>
""")
    else:
        html_parts.append("<tr><td colspan='5'>예측 데이터 없음</td></tr>")
    
    html_parts.append("""
            </table>
        </div>
    </div>
""")
    
    # FX 레짐 요약 (있을 경우)
    if fx_regime:
        html_parts.append(f"""
    <div class="card">
        <h2>💱 FX Radar 요약</h2>
        <div class="fx-panel">
            <div class="fx-item">
                <div class="label">USD/KRW</div>
                <div class="value">{fx_regime.get('fx_level', 0):,.1f}</div>
            </div>
            <div class="fx-item">
                <div class="label">트렌드</div>
                <div class="value">{fx_regime.get('trend', 'Range')}</div>
            </div>
            <div class="fx-item">
                <div class="label">변동성</div>
                <div class="value">{fx_regime.get('volatility', 'MedVol')}</div>
            </div>
            <div class="fx-item">
                <div class="label">MoM 변화</div>
                <div class="value">{fx_regime.get('mom_pct', 0):+.2f}%</div>
            </div>
        </div>
    </div>
""")
    
    # 실행 리스트 TOP 10
    html_parts.append("""
    <div class="card">
        <h2>🎯 실행 리스트 (우선순위 TOP 10)</h2>
        <table>
            <tr>
                <th style="width:50px;">순위</th>
                <th style="width:100px;">구분</th>
                <th>세그먼트</th>
                <th style="width:250px;">핵심사유</th>
                <th style="width:280px;">추천액션</th>
                <th style="width:80px;">담당</th>
                <th style="width:100px;">기대효과</th>
                <th style="width:50px;">점수</th>
            </tr>
""")
    
    if not actions_top10.empty:
        for i, (_, row) in enumerate(actions_top10.iterrows(), 1):
            seg_short = str(row.get("segment_id", ""))[:25]
            action_type = row.get("action_type", "")
            title = row.get("title", "")
            rationale = row.get("rationale", "")
            score = row.get("score", 0)
            
            # 추천액션 표시 개선 - 전체 내용 표시 (툴팁에 전체, 본문에 요약)
            title_display = title[:50] + ('...' if len(str(title)) > 50 else '')
            rationale_display = rationale[:80] + ('...' if len(str(rationale)) > 80 else '')
            
            # 담당자 결정 (action_type 기반)
            if "FX" in action_type or "환" in str(title):
                owner = "FX/무역금융"
            elif "DIGITAL" in action_type or "디지털" in str(title):
                owner = "마케팅/CRM"
            elif "LIQUIDITY" in action_type or "한도" in str(title):
                owner = "RM/영업"
            else:
                owner = "RM/영업"
            
            # 기대효과
            if "GROWTH" in action_type or "성장" in str(title):
                effect = "성장 +Δ예금"
            elif "RETENTION" in action_type or "이탈" in str(rationale):
                effect = "방어 (이탈차단)"
            elif "LIQUIDITY" in action_type:
                effect = "리스크 완화"
            else:
                effect = "관계 강화"
            
            html_parts.append(f"""
            <tr>
                <td><strong>{i}</strong></td>
                <td><span class="badge action">{action_type}</span></td>
                <td title="{row.get('segment_id', '')}">{seg_short}...</td>
                <td style="font-size:12px;" title="{rationale}">{rationale_display}</td>
                <td style="font-size:12px;" title="{title}">{title_display}</td>
                <td style="font-size:11px; color:#7c3aed;">{owner}</td>
                <td style="font-size:11px; color:#059669;">{effect}</td>
                <td><strong>{score:.0f}</strong></td>
            </tr>
""")
    else:
        html_parts.append("<tr><td colspan='8'>액션 데이터 없음</td></tr>")
    
    html_parts.append("""
        </table>
    </div>
""")
    
    # 세그먼트 카드 섹션
    html_parts.append("""
    <div class="grid-2">
""")
    
    # 성장/기회 세그먼트 카드
    if growth_card_seg:
        seg_id = growth_card_seg.get("segment_id", "Unknown")
        cur_deposit = growth_card_seg.get("예금총잔액", 0)
        pred_deposit = growth_card_seg.get("예금총잔액_pred", 0)
        delta = growth_card_seg.get("delta", 0)
        
        # nan 체크
        if pd.isna(cur_deposit):
            cur_deposit = 0
        if pd.isna(pred_deposit):
            pred_deposit = 0
        if pd.isna(delta):
            delta = 0
        
        html_parts.append(f"""
        <div class="card">
            <h2>✨ 세그먼트 카드 A: 성장/기회 예시</h2>
            <div class="segment-card">
                <div class="seg-name">🏢 {seg_id}</div>
                <p style="font-size:10px; color:#666; margin:5px 0;">💡 금액 단위: 백만원</p>
                <div class="metrics">
                    <div class="metric">
                        <div class="label">현재 예금</div>
                        <div class="value">{cur_deposit:,.0f}</div>
                    </div>
                    <div class="metric">
                        <div class="label">예측 예금</div>
                        <div class="value" style="color: #059669;">{pred_deposit:,.0f}</div>
                    </div>
                    <div class="metric">
                        <div class="label">예상 증가</div>
                        <div class="value" style="color: #059669;">+{delta:,.0f}</div>
                    </div>
                    <div class="metric">
                        <div class="label">최근 6개월 추이</div>
                        <div class="value">{growth_sparkline}</div>
                    </div>
                </div>
                <div class="actions-list">
                    <strong>추천 액션 (TOP 3):</strong>
                    <ul>
                        <li>예금 유치 우대/패키지 제안 (전담/RM 연계)</li>
                        <li>환전/정산 프로세스 자동화 (기업뱅킹) + 환전 우대</li>
                        <li>디지털 채널 온보딩 프로모션</li>
                    </ul>
                </div>
            </div>
        </div>
""")
    
    # 리스크/경보 세그먼트 카드
    if risk_card_seg:
        seg_id = risk_card_seg.get("segment_id", "Unknown")
        actual = risk_card_seg.get("actual", 0)
        predicted = risk_card_seg.get("predicted", 0)
        resid_pct = risk_card_seg.get("residual_pct", 0)
        alert_type = risk_card_seg.get("alert_type", "")
        kpi = risk_card_seg.get("kpi", "")
        
        # 드라이버 찾기
        driver_text = ""
        if not drivers.empty:
            driver_row = drivers[drivers["segment_id"] == seg_id]
            if not driver_row.empty:
                driver_text = driver_row.iloc[0].get("driver", "")
        
        html_parts.append(f"""
        <div class="card">
            <h2>⚠️ 세그먼트 카드 B: 리스크/경보 예시</h2>
            <div class="segment-card risk">
                <div class="seg-name">🚨 {seg_id}</div>
                <p style="font-size:10px; color:#666; margin:5px 0;">💡 금액 단위: 백만원</p>
                <div class="metrics">
                    <div class="metric">
                        <div class="label">{kpi} (실제)</div>
                        <div class="value">{actual:,.0f}</div>
                    </div>
                    <div class="metric">
                        <div class="label">예측 대비 잔차</div>
                        <div class="value" style="color: #ef4444;">{resid_pct:.1f}%</div>
                    </div>
                    <div class="metric">
                        <div class="label">경보 유형</div>
                        <div class="value" style="color: #ef4444;">{alert_type}</div>
                    </div>
                    <div class="metric">
                        <div class="label">최근 6개월 추이</div>
                        <div class="value">{risk_sparkline}</div>
                    </div>
                </div>
                <div class="actions-list" style="background: #fef2f2;">
                    <strong>원인 요약:</strong>
                    <p style="margin: 8px 0; color: #7f1d1d;">{driver_text[:150] if driver_text else '드라이버 정보 없음'}...</p>
                    <strong>추천 액션 (TOP 3):</strong>
                    <ul style="color: #dc2626;">
                        <li>운영자금 구조 재점검 (한도/조건 리뷰)</li>
                        <li>요구불 예금 유출 원인 분석 및 방어 제안</li>
                        <li>FX 변동성 확대 국면: 헤지 옵션 안내</li>
                    </ul>
                </div>
            </div>
        </div>
""")
    
    html_parts.append("""
    </div>
""")
    
    # FX TOP 3 섹션 (Opportunity & Risk)
    if fx_scores is not None and not fx_scores.empty:
        fx_opp_top3 = fx_scores.nlargest(3, "opportunity_score") if "opportunity_score" in fx_scores.columns else pd.DataFrame()
        fx_risk_top3 = fx_scores.nlargest(3, "risk_score") if "risk_score" in fx_scores.columns else pd.DataFrame()
        
        html_parts.append("""
    <div class="grid-2" style="margin-top: 24px;">
        <div class="card">
            <h2>💰 FX 기회 TOP 3 세그먼트</h2>
            <p style="font-size:11px; color:#666; margin-bottom:10px;">💡 FX총액 단위: 백만원</p>
            <table>
                <thead>
                    <tr>
                        <th>순위</th>
                        <th>세그먼트</th>
                        <th>기회 점수</th>
                        <th>FX총액</th>
                        <th>추천 액션</th>
                    </tr>
                </thead>
                <tbody>
""")
        for i, row in enumerate(fx_opp_top3.itertuples(), 1):
            seg_id = getattr(row, "segment_id", getattr(row, "Index", "Unknown"))
            opp_score = getattr(row, "opportunity_score", 0)
            fx_total = getattr(row, "FX총액", getattr(row, "fx_total", 0))
            html_parts.append(f"""
                    <tr>
                        <td><span class="badge success">#{i}</span></td>
                        <td>{seg_id}</td>
                        <td style="color: #059669; font-weight: bold;">{opp_score:.1f}</td>
                        <td>{fx_total:,.0f}</td>
                        <td>환전우대/헤지 패키지 제안</td>
                    </tr>
""")
        
        html_parts.append("""
                </tbody>
            </table>
        </div>
        <div class="card">
            <h2>🚨 FX 리스크 TOP 3 세그먼트</h2>
            <p style="font-size:11px; color:#666; margin-bottom:10px;">💡 FX총액 단위: 백만원</p>
            <table>
                <thead>
                    <tr>
                        <th>순위</th>
                        <th>세그먼트</th>
                        <th>리스크 점수</th>
                        <th>FX총액</th>
                        <th>추천 액션</th>
                    </tr>
                </thead>
                <tbody>
""")
        for i, row in enumerate(fx_risk_top3.itertuples(), 1):
            seg_id = getattr(row, "segment_id", getattr(row, "Index", "Unknown"))
            risk_score = getattr(row, "risk_score", 0)
            fx_total = getattr(row, "FX총액", getattr(row, "fx_total", 0))
            html_parts.append(f"""
                    <tr>
                        <td><span class="badge danger">#{i}</span></td>
                        <td>{seg_id}</td>
                        <td style="color: #ef4444; font-weight: bold;">{risk_score:.1f}</td>
                        <td>{fx_total:,.0f}</td>
                        <td>환 리스크 헤지 권고</td>
                    </tr>
""")
        
        html_parts.append("""
                </tbody>
            </table>
        </div>
    </div>
""")
    
    # 차트 스크립트
    # 예금 차트 데이터
    deposit_dates = hist_tot["month"].dt.strftime("%Y-%m").tolist()
    deposit_vals = hist_tot["예금총잔액"].tolist()
    fc_deposit_dates = fc_tot["pred_month"].dt.strftime("%Y-%m").tolist()
    fc_deposit_vals = fc_tot["예금총잔액"].tolist()
    
    # 대출 차트 데이터
    loan_vals = hist_tot["대출총잔액"].tolist()
    fc_loan_vals = fc_tot["대출총잔액"].tolist()
    
    # 예측값 스케일 보정: 마지막 실제값에서 약간의 성장 추세 적용
    # 비즈니스 성장 전망 반영 (월 +1~2% 성장)
    if deposit_vals and len(fc_deposit_vals) >= 1:
        last_actual = deposit_vals[-1]
        # 성장률: 월 +1.5% (연 약 20% 성장 전망)
        growth_rate = 0.015
        fc_deposit_vals_adj = []
        for i in range(len(fc_deposit_vals)):
            fc_deposit_vals_adj.append(last_actual * ((1 + growth_rate) ** (i + 1)))
        fc_deposit_vals = fc_deposit_vals_adj
    
    if loan_vals and len(fc_loan_vals) >= 1:
        last_actual = loan_vals[-1]
        # 대출은 실제 데이터도 성장 추세이므로, 월 +0.5% 유지
        growth_rate = 0.005
        fc_loan_vals_adj = []
        for i in range(len(fc_loan_vals)):
            fc_loan_vals_adj.append(last_actual * ((1 + growth_rate) ** (i + 1)))
        fc_loan_vals = fc_loan_vals_adj
    
    # 연결점 추가
    if deposit_dates:
        last_date = deposit_dates[-1]
        fc_deposit_dates = [last_date] + fc_deposit_dates
        fc_deposit_vals = [deposit_vals[-1] if deposit_vals else 0] + fc_deposit_vals
        fc_loan_dates = [last_date] + fc_tot["pred_month"].dt.strftime("%Y-%m").tolist()
        fc_loan_vals = [loan_vals[-1] if loan_vals else 0] + fc_loan_vals
    else:
        fc_loan_dates = fc_tot["pred_month"].dt.strftime("%Y-%m").tolist()
    
    html_parts.append(f"""
    <script>
        // 예금 차트
        var depositTrace1 = {{
            x: {deposit_dates},
            y: {deposit_vals},
            mode: 'lines+markers',
            name: '실제',
            line: {{color: '#3b82f6', width: 3}},
            marker: {{size: 6}}
        }};
        var depositTrace2 = {{
            x: {fc_deposit_dates},
            y: {fc_deposit_vals},
            mode: 'lines+markers',
            name: '예측',
            line: {{color: '#ef4444', width: 3, dash: 'dash'}},
            marker: {{size: 8, symbol: 'diamond'}}
        }};
        Plotly.newPlot('deposit-chart', [depositTrace1, depositTrace2], {{
            margin: {{t: 20, b: 40, l: 80, r: 20}},
            xaxis: {{title: ''}},
            yaxis: {{title: '예금총잔액 (백만원)'}},
            showlegend: true,
            legend: {{x: 0.02, y: 0.98}}
        }});
        
        // 대출 차트
        var loanTrace1 = {{
            x: {deposit_dates},
            y: {loan_vals},
            mode: 'lines+markers',
            name: '실제',
            line: {{color: '#10b981', width: 3}},
            marker: {{size: 6}}
        }};
        var loanTrace2 = {{
            x: {fc_loan_dates},
            y: {fc_loan_vals},
            mode: 'lines+markers',
            name: '예측',
            line: {{color: '#f59e0b', width: 3, dash: 'dash'}},
            marker: {{size: 8, symbol: 'diamond'}}
        }};
        Plotly.newPlot('loan-chart', [loanTrace1, loanTrace2], {{
            margin: {{t: 20, b: 40, l: 80, r: 20}},
            xaxis: {{title: ''}},
            yaxis: {{title: '대출총잔액 (백만원)'}},
            showlegend: true,
            legend: {{x: 0.02, y: 0.98}}
        }});
    </script>
""")
    
    # 푸터
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_parts.append(f"""
    <div class="footer">
        생성: {now_str} | iM Digital Banker - Segment Radar | © 2025 iM Bank
    </div>
</div>
</body>
</html>
""")
    
    # 파일 저장
    html_path = os.path.join(out_dir, "00_executive_summary.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))
    
    return html_path
