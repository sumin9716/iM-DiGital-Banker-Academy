from __future__ import annotations

import os
from typing import Dict, Optional, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def save_overview_dashboard(
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    out_dir: str,
    kpi_cols: List[str],
) -> str:
    """
    Create an overview HTML showing:
    - historical total KPI trends
    - next 1~3 month forecasts (total across segments)
    - actual의 마지막 점과 forecast의 시작점이 연결됨
    """
    os.makedirs(out_dir, exist_ok=True)

    hist = panel[panel.get("pre_birth", 0) == 0].copy()
    hist_tot = hist.groupby("month")[kpi_cols].sum(numeric_only=True).reset_index()

    # forecast totals
    fc_tot = forecasts.groupby("pred_month")[kpi_cols].sum(numeric_only=True).reset_index().rename(columns={"pred_month": "month"})
    
    # actual의 마지막 점을 forecast의 시작점으로 추가 (연결을 위해)
    last_actual = hist_tot[hist_tot["month"] == hist_tot["month"].max()].copy()
    
    # forecast 시작점에 actual 마지막 값 추가
    fc_with_start = pd.concat([last_actual, fc_tot], axis=0, ignore_index=True)
    fc_with_start["type"] = "forecast"
    
    hist_tot2 = hist_tot.copy()
    hist_tot2["type"] = "actual"

    combined = pd.concat([hist_tot2, fc_with_start], axis=0, ignore_index=True)

    figs = []
    for k in kpi_cols:
        fig = px.line(combined, x="month", y=k, color="type", title=f"전체 {k} 추이 (actual vs forecast)")
        # actual: solid line, forecast: dashed line
        fig.update_traces(line=dict(dash="dash"), selector=dict(name="forecast"))
        fig.update_traces(line=dict(dash="solid"), selector=dict(name="actual"))
        figs.append(fig)

    # Combine into single HTML
    html_path = os.path.join(out_dir, "01_overview.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'></head><body>")
        f.write("<h1>Segment Radar — Overview</h1>")
        for fig in figs:
            f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write("</body></html>")
    return html_path


def save_watchlist_dashboard(
    watchlist: pd.DataFrame,
    out_dir: str,
    title: str = "Risk Radar — Watchlist",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    wl = watchlist.copy()
    wl["month"] = wl["month"].astype(str)
    
    # 개선된 HTML 대시보드
    html_parts = []
    html_parts.append("""
    <html>
    <head>
        <meta charset='utf-8'>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background: #f5f5f5; margin: 0; padding: 20px; }
            .container { max-width: 1400px; margin: 0 auto; }
            h1 { color: #1a237e; border-bottom: 3px solid #3f51b5; padding-bottom: 10px; }
            h2 { color: #303f9f; margin-top: 30px; }
            .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
            .summary-card { background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .summary-card .value { font-size: 32px; font-weight: bold; color: #1a237e; }
            .summary-card .label { color: #666; margin-top: 5px; }
            .alert-table { width: 100%; border-collapse: collapse; background: white; margin: 20px 0; }
            .alert-table th { background: #d32f2f; color: white; padding: 12px; text-align: left; }
            .alert-table td { padding: 12px; border-bottom: 1px solid #eee; }
            .alert-table tr:hover { background: #ffebee; }
            .severity-critical { background: #ffcdd2; font-weight: bold; }
            .severity-high { background: #fff3e0; }
            .severity-medium { background: #e3f2fd; }
            .alert-card { background: white; border-left: 4px solid #d32f2f; padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0; }
            .alert-card .segment { font-weight: bold; font-size: 16px; color: #c62828; }
            .alert-card .meta { color: #888; font-size: 12px; margin: 5px 0; }
            .alert-card .explanation { background: #fff8e1; padding: 10px; border-radius: 4px; margin-top: 10px; }
            .alert-card .driver { color: #555; font-size: 13px; margin-top: 8px; padding: 8px; background: #fafafa; border-left: 3px solid #ff9800; }
        </style>
    </head>
    <body>
    <div class='container'>
    """)
    
    html_parts.append(f"<h1>⚠️ {title}</h1>")
    
    # 요약 통계
    html_parts.append("<div class='summary-grid'>")
    total_alerts = len(wl)
    
    # severity 컬럼이 있으면 분석
    critical = 0
    high = 0
    if 'severity' in wl.columns:
        # severity를 numeric으로 변환 시도
        try:
            wl['severity_num'] = pd.to_numeric(wl['severity'], errors='coerce').fillna(0)
            critical = len(wl[wl['severity_num'] >= 0.8])
            high = len(wl[(wl['severity_num'] >= 0.5) & (wl['severity_num'] < 0.8)])
        except Exception:
            pass
    
    html_parts.append(f"""
        <div class='summary-card'>
            <div class='value' style='color: #d32f2f;'>{total_alerts}</div>
            <div class='label'>총 알림 건수</div>
        </div>
        <div class='summary-card'>
            <div class='value' style='color: #c62828;'>{critical}</div>
            <div class='label'>🚨 긴급 알림</div>
        </div>
        <div class='summary-card'>
            <div class='value' style='color: #e65100;'>{high}</div>
            <div class='label'>⚠️ 주의 알림</div>
        </div>
    """)
    html_parts.append("</div>")
    
    # 상세 알림 카드
    html_parts.append("<h2>📋 상세 알림 목록</h2>")
    
    for _, row in wl.iterrows():
        segment = row.get('segment_id', str(row.get('segment', '')))
        month = row.get('month', '')
        kpi = row.get('kpi', row.get('target', ''))
        severity_raw = row.get('severity', 0)
        # severity를 float로 변환
        try:
            severity = float(severity_raw) if severity_raw is not None else 0.0
        except (ValueError, TypeError):
            severity = 0.0
        explanation = row.get('explanation', row.get('reason', ''))
        driver_summary = row.get('driver_summary', '')
        
        # severity 기반 스타일
        if severity >= 0.8:
            sev_icon = '🚨'
            sev_text = '긴급'
        elif severity >= 0.5:
            sev_icon = '⚠️'
            sev_text = '주의'
        else:
            sev_icon = '💡'
            sev_text = '참고'
        
        html_parts.append(f"""
        <div class='alert-card'>
            <div class='segment'>{sev_icon} {segment}</div>
            <div class='meta'>
                📅 {month} | 📊 KPI: {kpi} | 심각도: {sev_text} ({severity:.1%})
            </div>
        """)
        
        if explanation:
            html_parts.append(f"""
            <div class='explanation'>
                <b>📋 핵심 사유:</b> {explanation}
            </div>
            """)
        
        if driver_summary:
            html_parts.append(f"""
            <div class='driver'>
                <b>💬 원인 분석:</b> {driver_summary}
            </div>
            """)
        
        html_parts.append("</div>")
    
    # 테이블 형태로도 제공
    html_parts.append("<h2>📊 전체 데이터 테이블</h2>")
    
    table = go.Figure(
        data=[go.Table(
            header=dict(values=list(wl.columns), fill_color='#3f51b5', font=dict(color='white')),
            cells=dict(values=[wl[c] for c in wl.columns], fill_color='white')
        )]
    )
    table.update_layout(title="", margin=dict(l=0, r=0, t=0, b=0))
    html_parts.append(table.to_html(full_html=False, include_plotlyjs="cdn"))
    
    html_parts.append("</div></body></html>")

    html_path = os.path.join(out_dir, "03_risk_watchlist.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    return html_path


def save_growth_dashboard(
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    out_dir: str,
    kpi: str = "예금총잔액",
    horizon: int = 1,
    top_n: int = 50,
) -> str:
    """
    Growth Forecast tab (MVP):
    - Top segments by predicted delta for next month/horizon
    """
    os.makedirs(out_dir, exist_ok=True)

    last_month = panel["month"].max()
    cur = panel[panel["month"] == last_month][["segment_id", kpi]].copy()
    pred_month = (last_month.to_period("M") + horizon).to_timestamp()
    
    # Handle column naming issues from merge (예금총잔액_x, 예금총잔액_y, 예금총잔액)
    fc_subset = forecasts[forecasts["pred_month"] == pred_month].copy()
    
    # Find the correct KPI column based on horizon
    # horizon=1 -> _x suffix, horizon=2 -> _y suffix, horizon=3 -> no suffix
    kpi_col_candidates = [f"{kpi}_x", f"{kpi}_y", kpi]
    kpi_col = None
    for cand in kpi_col_candidates:
        if cand in fc_subset.columns and fc_subset[cand].notna().any():
            kpi_col = cand
            break
    
    if kpi_col is None:
        # fallback: try to find any column containing the kpi name
        for col in fc_subset.columns:
            if kpi in col and fc_subset[col].notna().any():
                kpi_col = col
                break
    
    if kpi_col is None:
        print(f"Warning: No valid forecast column found for {kpi}")
        kpi_col = kpi  # fallback
    
    fc = fc_subset[["segment_id", kpi_col]].copy()
    fc = fc.rename(columns={kpi_col: f"{kpi}__pred"})

    merged = cur.merge(fc, on="segment_id", how="left")
    merged["delta"] = merged[f"{kpi}__pred"] - merged[kpi]
    merged = merged.dropna(subset=["delta"])  # Remove rows with null delta
    merged = merged.sort_values("delta", ascending=False).head(top_n)

    fig = px.bar(merged, x="segment_id", y="delta", title=f"{kpi} 예측 상승 TOP {top_n} (h={horizon})")
    fig.update_layout(xaxis_title="segment_id", yaxis_title="predicted delta")

    table = go.Figure(
        data=[go.Table(
            header=dict(values=list(merged.columns)),
            cells=dict(values=[merged[c] for c in merged.columns])
        )]
    )
    table.update_layout(title=f"{kpi} 예측 상승 리스트 (h={horizon})")

    html_path = os.path.join(out_dir, "02_growth_forecast.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'></head><body>")
        f.write("<h1>Growth Forecast</h1>")
        f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write(table.to_html(full_html=False, include_plotlyjs=False))
        f.write("</body></html>")
    return html_path


def save_fx_radar_dashboard(
    fx_scores: pd.DataFrame,
    fx_regime_result,
    out_dir: str,
    top_n: int = 30,
) -> str:
    """
    FX Radar 대시보드 탭:
    - 환율 레짐 패널 (현재 레짐, 트렌드, 변동성)
    - FX Opportunity Top 세그먼트
    - FX Risk/Alert Top 세그먼트
    - 레짐 기반 추천 액션 카드
    
    Args:
        fx_scores: FX 세그먼트별 점수 DataFrame
        fx_regime_result: FXRegimeResult instance
        out_dir: 출력 디렉토리
        top_n: 표시할 세그먼트 수
    
    Returns:
        HTML 파일 경로
    """
    os.makedirs(out_dir, exist_ok=True)
    
    html_parts = []
    html_parts.append("""
    <html>
    <head>
        <meta charset='utf-8'>
        <style>
            body { font-family: 'Malgun Gothic', Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1400px; margin: 0 auto; }
            h1 { color: #1a237e; border-bottom: 3px solid #3f51b5; padding-bottom: 10px; }
            h2 { color: #303f9f; margin-top: 30px; }
            .regime-panel { 
                display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0;
            }
            .regime-card {
                background: white; border-radius: 12px; padding: 20px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); flex: 1; min-width: 200px;
            }
            .regime-card h3 { margin: 0 0 10px 0; color: #666; font-size: 14px; }
            .regime-card .value { font-size: 28px; font-weight: bold; color: #1a237e; }
            .regime-card .subvalue { font-size: 14px; color: #888; margin-top: 5px; }
            .uptrend { color: #d32f2f !important; }
            .downtrend { color: #388e3c !important; }
            .highvol { background: #fff3e0 !important; }
            .score-table { width: 100%; border-collapse: collapse; margin: 20px 0; background: white; }
            .score-table th { background: #3f51b5; color: white; padding: 12px; text-align: left; }
            .score-table td { padding: 10px; border-bottom: 1px solid #eee; }
            .score-table tr:hover { background: #f5f5f5; }
            .grade-A { background: #ffcdd2; font-weight: bold; }
            .grade-B { background: #fff9c4; }
            .grade-C { background: #e3f2fd; }
            .action-card {
                background: white; border-left: 4px solid #3f51b5; padding: 15px;
                margin: 10px 0; border-radius: 0 8px 8px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .action-card .segment { font-weight: bold; color: #1a237e; font-size: 16px; }
            .action-card .meta { color: #888; font-size: 12px; margin: 5px 0; }
            .action-card .actions { color: #333; margin-top: 10px; line-height: 1.6; }
            .action-card .actions .action-item { 
                background: #f8f9fa; padding: 8px 12px; margin: 5px 0; 
                border-radius: 4px; display: flex; align-items: center;
            }
            .action-card .actions .action-item .urgency { 
                display: inline-block; padding: 2px 8px; border-radius: 10px; 
                font-size: 11px; margin-right: 10px; font-weight: bold;
            }
            .action-card .actions .urgency-critical { background: #ffcdd2; color: #c62828; }
            .action-card .actions .urgency-high { background: #fff3e0; color: #e65100; }
            .action-card .actions .urgency-medium { background: #e3f2fd; color: #1565c0; }
            .action-card .actions .urgency-low { background: #e8f5e9; color: #2e7d32; }
            .action-card .reason { 
                color: #666; font-size: 13px; margin-top: 10px; padding: 10px;
                background: #fafafa; border-radius: 4px; border-left: 3px solid #9e9e9e;
            }
            .action-card .cause { 
                color: #555; font-size: 12px; margin-top: 8px; font-style: italic;
            }
            .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        </style>
    </head>
    <body>
    <div class="container">
        <h1>📈 FX Opportunity & Risk Radar</h1>
    """)
    
    # 1. 레짐 패널
    html_parts.append("<h2>🌐 현재 환율 레짐</h2>")
    html_parts.append("<div class='regime-panel'>")
    
    if fx_regime_result is not None and hasattr(fx_regime_result, 'fx_monthly'):
        fx_df = fx_regime_result.fx_monthly
        if len(fx_df) > 0:
            latest = fx_df.iloc[-1]
            
            # 환율 레벨
            fx_level = latest.get('fx_level', 0)
            html_parts.append(f"""
            <div class='regime-card'>
                <h3>USD/KRW 환율</h3>
                <div class='value'>{fx_level:,.1f}</div>
                <div class='subvalue'>월 평균</div>
            </div>
            """)
            
            # 트렌드
            trend = latest.get('fx_trend_regime', 'Range')
            trend_class = 'uptrend' if trend == 'Uptrend' else ('downtrend' if trend == 'Downtrend' else '')
            trend_icon = '📈' if trend == 'Uptrend' else ('📉' if trend == 'Downtrend' else '➡️')
            html_parts.append(f"""
            <div class='regime-card'>
                <h3>트렌드</h3>
                <div class='value {trend_class}'>{trend_icon} {trend}</div>
                <div class='subvalue'>{'원화 약세' if trend == 'Uptrend' else ('원화 강세' if trend == 'Downtrend' else '횡보')}</div>
            </div>
            """)
            
            # 변동성
            vol = latest.get('fx_vol_regime', 'MedVol')
            vol_class = 'highvol' if vol == 'HighVol' else ''
            vol_icon = '🔥' if vol == 'HighVol' else ('❄️' if vol == 'LowVol' else '〰️')
            html_parts.append(f"""
            <div class='regime-card {vol_class}'>
                <h3>변동성</h3>
                <div class='value'>{vol_icon} {vol}</div>
                <div class='subvalue'>{'고변동성 주의' if vol == 'HighVol' else ('안정' if vol == 'LowVol' else '보통')}</div>
            </div>
            """)
            
            # 모멘텀
            mom_pct = latest.get('fx_mom_pct', 0) * 100
            mom_icon = '⬆️' if mom_pct > 0 else '⬇️'
            html_parts.append(f"""
            <div class='regime-card'>
                <h3>월간 변동</h3>
                <div class='value'>{mom_icon} {mom_pct:+.2f}%</div>
                <div class='subvalue'>전월 대비</div>
            </div>
            """)
    
    html_parts.append("</div>")
    
    # 2. FX Opportunity TOP 세그먼트
    html_parts.append("<h2>🎯 FX Opportunity TOP 세그먼트</h2>")
    
    if not fx_scores.empty:
        opp_top = fx_scores.nlargest(top_n, 'opportunity_score')
        
        html_parts.append("<table class='score-table'>")
        html_parts.append("""
        <tr>
            <th>순위</th>
            <th>세그먼트</th>
            <th>FX총액</th>
            <th>기회점수</th>
            <th>리스크점수</th>
            <th>등급</th>
            <th>추천 액션</th>
        </tr>
        """)
        
        for i, (_, row) in enumerate(opp_top.iterrows(), 1):
            grade = row.get('priority_grade', 'C')
            grade_class = f'grade-{grade}'
            fx_amt = row.get('FX총액', 0)
            # 단위: 백만원 -> 억원 (/100)
            fx_eok = fx_amt / 100
            fx_display = f"{fx_eok:,.1f}억" if fx_eok >= 1 else f"{fx_amt:,.0f}백만"
            
            # 추천 액션 첫 번째 항목만 추출 (| 구분)
            actions_raw = row.get('recommended_actions', '')
            first_action = actions_raw.split('|')[0].strip() if actions_raw else '-'
            # 80자로 늘림
            action_display = first_action[:80] + ('...' if len(first_action) > 80 else '')
            
            html_parts.append(f"""
            <tr class='{grade_class}'>
                <td>{i}</td>
                <td>{row.get('segment_id', '')[:30]}...</td>
                <td>{fx_display}</td>
                <td><b>{row.get('opportunity_score', 0):.1f}</b></td>
                <td>{row.get('risk_score', 0):.1f}</td>
                <td><b>{grade}</b></td>
                <td title='{actions_raw[:200]}'>{action_display}</td>
            </tr>
            """)
        
        html_parts.append("</table>")
    else:
        html_parts.append("<p>FX 활성 세그먼트가 없습니다.</p>")
    
    # 3. FX Risk TOP 세그먼트
    html_parts.append("<h2>⚠️ FX Risk TOP 세그먼트</h2>")
    
    if not fx_scores.empty:
        risk_top = fx_scores.nlargest(top_n, 'risk_score')
        
        html_parts.append("<table class='score-table'>")
        html_parts.append("""
        <tr>
            <th>순위</th>
            <th>세그먼트</th>
            <th>FX총액</th>
            <th>리스크점수</th>
            <th>기회점수</th>
            <th>등급</th>
            <th>리스크 요인</th>
        </tr>
        """)
        
        for i, (_, row) in enumerate(risk_top.iterrows(), 1):
            grade = row.get('priority_grade', 'C')
            grade_class = f'grade-{grade}'
            fx_amt = row.get('FX총액', 0)
            # 단위: 백만원 -> 억원 (/100)
            fx_eok = fx_amt / 100
            fx_display = f"{fx_eok:,.1f}억" if fx_eok >= 1 else f"{fx_amt:,.0f}백만"
            
            # 리스크 요인 첫 번째 항목 추출
            risk_factors_raw = row.get('risk_factors', '')
            first_factor = risk_factors_raw.split('|')[0].strip() if risk_factors_raw else '-'
            factor_display = first_factor[:80] + ('...' if len(first_factor) > 80 else '')
            
            html_parts.append(f"""
            <tr class='{grade_class}'>
                <td>{i}</td>
                <td>{row.get('segment_id', '')[:30]}...</td>
                <td>{fx_display}</td>
                <td><b>{row.get('risk_score', 0):.1f}</b></td>
                <td>{row.get('opportunity_score', 0):.1f}</td>
                <td><b>{grade}</b></td>
                <td title='{risk_factors_raw[:200]}'>{factor_display}</td>
            </tr>
            """)
        
        html_parts.append("</table>")
    
    # 4. 레짐 기반 추천 액션 카드
    html_parts.append("<h2>💡 레짐 기반 추천 액션 (TOP 10)</h2>")
    
    if not fx_scores.empty:
        action_top = fx_scores[fx_scores['priority_grade'].isin(['A', 'B'])].head(10)
        
        for _, row in action_top.iterrows():
            seg_id = row.get('segment_id', '')[:40]
            actions = row.get('recommended_actions', '')
            opp = row.get('opportunity_score', 0)
            risk = row.get('risk_score', 0)
            grade = row.get('priority_grade', 'C')
            
            # 액션 문자열 파싱 (| 구분)
            action_items = [a.strip() for a in actions.split('|') if a.strip()]
            
            # 긴급도 아이콘에서 urgency 레벨 추출
            def get_urgency_class(action_text):
                if '🚨' in action_text or '⚡' in action_text:
                    return 'urgency-critical'
                elif '⚠️' in action_text or '📈' in action_text or '📉' in action_text:
                    return 'urgency-high'
                elif '💡' in action_text or '🔔' in action_text:
                    return 'urgency-medium'
                else:
                    return 'urgency-low'
            
            html_parts.append(f"""
            <div class='action-card'>
                <div class='segment'>🏢 {seg_id}</div>
                <div class='meta'>
                    📊 기회점수: <b>{opp:.0f}</b> | 리스크점수: <b>{risk:.0f}</b> | 등급: <b>{grade}</b>
                </div>
                <div class='actions'>
            """)
            
            # 각 액션 아이템 개별 표시
            for action in action_items[:3]:  # 최대 3개
                urg_class = get_urgency_class(action)
                urg_label = {'urgency-critical': '긴급', 'urgency-high': '높음', 
                             'urgency-medium': '중간', 'urgency-low': '낮음'}.get(urg_class, '')
                html_parts.append(f"""
                    <div class='action-item'>
                        <span class='urgency {urg_class}'>{urg_label}</span>
                        <span>{action}</span>
                    </div>
                """)
            
            # 핵심 사유 표시
            reason = row.get('action_reason', row.get('risk_factors', ''))
            if reason:
                html_parts.append(f"""
                    <div class='reason'>
                        <b>📋 핵심 사유:</b> {reason[:200]}
                    </div>
                """)
            
            # 원인 요약 표시  
            cause = row.get('cause_summary', row.get('driver_summary', ''))
            if cause:
                html_parts.append(f"""
                    <div class='cause'>
                        💬 {cause[:150]}
                    </div>
                """)
            
            html_parts.append("</div></div>")
    
    # 5. 요약 통계
    html_parts.append("<h2>📊 FX 세그먼트 요약</h2>")
    html_parts.append("<div class='stat-grid'>")
    
    if not fx_scores.empty:
        total_segments = len(fx_scores)
        grade_a = len(fx_scores[fx_scores['priority_grade'] == 'A'])
        grade_b = len(fx_scores[fx_scores['priority_grade'] == 'B'])
        high_opp = len(fx_scores[fx_scores['opportunity_score'] >= 70])
        high_risk = len(fx_scores[fx_scores['risk_score'] >= 70])
        
        html_parts.append(f"""
        <div class='regime-card'>
            <h3>총 FX 활성 세그먼트</h3>
            <div class='value'>{total_segments:,}</div>
        </div>
        <div class='regime-card'>
            <h3>A등급 (최우선)</h3>
            <div class='value' style='color: #d32f2f;'>{grade_a}</div>
        </div>
        <div class='regime-card'>
            <h3>B등급 (우선)</h3>
            <div class='value' style='color: #f57c00;'>{grade_b}</div>
        </div>
        <div class='regime-card'>
            <h3>고기회 (70+)</h3>
            <div class='value' style='color: #388e3c;'>{high_opp}</div>
        </div>
        <div class='regime-card'>
            <h3>고리스크 (70+)</h3>
            <div class='value' style='color: #d32f2f;'>{high_risk}</div>
        </div>
        """)
    
    html_parts.append("</div>")
    
    # 닫기
    html_parts.append("""
    </div>
    </body>
    </html>
    """)
    
    html_path = os.path.join(out_dir, "05_fx_radar.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))
    
    return html_path
