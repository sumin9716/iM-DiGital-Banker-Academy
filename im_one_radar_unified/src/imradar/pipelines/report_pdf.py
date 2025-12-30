from __future__ import annotations

import os
from typing import List
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def generate_monthly_onepager(
    month: str,
    top_alerts: pd.DataFrame,
    top_growth: pd.DataFrame,
    top_actions: pd.DataFrame,
    out_path: str,
    title: str = "세그먼트 운영 레이더 — 월간 1페이지 리포트",
):
    """
    Generate a simple 1-page PDF report (A4).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4

    y = height - 20 * mm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, y, title)
    y -= 8 * mm
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, y, f"기준월: {month}")
    y -= 12 * mm

    def _section(header: str, df: pd.DataFrame, y: float, max_rows: int = 8) -> float:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(20 * mm, y, header)
        y -= 6 * mm
        c.setFont("Helvetica", 9)
        if df is None or df.empty:
            c.drawString(22 * mm, y, "- (데이터 없음)")
            return y - 10 * mm

        cols = [col for col in df.columns if col not in {"sev_rank"}][:6]
        show = df[cols].head(max_rows)

        for i, (_, r) in enumerate(show.iterrows()):
            line = " | ".join([f"{col}:{str(r[col])}" for col in cols])
            if len(line) > 120:
                line = line[:120] + "..."
            c.drawString(22 * mm, y, f"- {line}")
            y -= 5 * mm
            if y < 30 * mm:
                break
        return y - 6 * mm

    y = _section("1) 이번 달 핵심 경보 (TOP)", top_alerts, y)
    y = _section("2) 다음 달 기회 (예측 상승 TOP)", top_growth, y)
    y = _section("3) 실행 리스트 (Next Best Action TOP)", top_actions, y)

    c.showPage()
    c.save()
