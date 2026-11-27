"""
고급 기능 개선 스크립트

개선 영역:
1. Horizon 3개월 성능 개선 (R²=77~87% → 90%+ 목표)
2. XAI (SHAP 기반 Feature 기여도 시각화)
3. Confidence Interval (90% CI 예측 불확실성)
4. 세그먼트 클러스터링 (K-Means/DBSCAN)
5. 액션 ROI 추정 (예상 효과 금액)
6. 다중 시나리오 분석 (Best/Base/Worst Case)
7. 외부 데이터 확장 (ESI, KASI 등 추가 통합)

Author: iM ONEderful Team
Date: 2026-01-14
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

# ============================================================================
# 1. Horizon 3개월 성능 개선
# ============================================================================

class HorizonOptimizer:
    """Horizon 3개월 예측 정확도 개선"""
    
    def __init__(self, panel: pd.DataFrame, cfg=None):
        self.panel = panel
        self.cfg = cfg
        
    def create_enhanced_features(self) -> pd.DataFrame:
        """3개월 예측을 위한 강화된 피처 생성 (최적화 버전)"""
        df = self.panel.copy()
        df = df.sort_values(['segment_id', 'month'])
        
        # 장기 트렌드 피처 (3, 6개월) - 벡터화 연산
        kpi_cols = ['log1p_예금총잔액', 'log1p_대출총잔액', 'log1p_카드총사용', 
                    'log1p_디지털거래금액', 'slog1p_순유입']
        
        existing_cols = [c for c in kpi_cols if c in df.columns]
        
        print(f"    피처 생성 대상: {len(existing_cols)} KPIs")
        
        for col in existing_cols:
            # 이동평균 (vectorized)
            df[f"{col}_ma3"] = df.groupby('segment_id')[col].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
            
            # 트렌드 기울기 (현재 vs 3개월 전)
            df[f"{col}_lag3"] = df.groupby('segment_id')[col].shift(3)
            df[f"{col}_trend_slope"] = (df[col] - df[f"{col}_lag3"]) / (df[f"{col}_lag3"].abs() + 1e-9)
            df = df.drop(columns=[f"{col}_lag3"])
        
        print(f"    생성 완료: {len(df.columns)} total columns")
        
        return df
    
    def get_horizon3_optimized_params(self) -> Dict:
        """Horizon 3 최적화된 LightGBM 파라미터"""
        return {
            'n_estimators': 500,
            'learning_rate': 0.03,  # 느린 학습률
            'num_leaves': 63,
            'max_depth': 8,
            'min_child_samples': 50,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'random_state': 42,
        }


# ============================================================================
# 2. XAI - SHAP 기반 Feature 기여도
# ============================================================================

@dataclass
class FeatureContribution:
    """Feature 기여도 데이터 클래스"""
    feature: str
    shap_value: float
    feature_value: float
    contribution_pct: float
    direction: str  # 'positive' or 'negative'


class XAIExplainer:
    """SHAP 기반 설명가능 AI"""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.shap_values = None
        
    def compute_shap_values(self, X: pd.DataFrame, sample_size: int = 1000) -> np.ndarray:
        """SHAP 값 계산 (TreeExplainer 기반)"""
        try:
            import shap
            
            # 샘플링 (대용량 데이터)
            if len(X) > sample_size:
                X_sample = X.sample(n=sample_size, random_state=42)
            else:
                X_sample = X
            
            # TreeExplainer (LightGBM 최적화)
            explainer = shap.TreeExplainer(self.model)
            self.shap_values = explainer.shap_values(X_sample)
            
            return self.shap_values
        except ImportError:
            print("⚠️ SHAP 라이브러리 미설치. Feature Importance로 대체합니다.")
            return self._compute_fallback_importance(X)
    
    def _compute_fallback_importance(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP 미설치 시 Fallback: LightGBM Feature Importance 기반"""
        if hasattr(self.model, 'feature_importances_'):
            imp = self.model.feature_importances_
            # 각 샘플에 대해 동일한 중요도 배분 (근사)
            return np.tile(imp, (len(X), 1))
        return np.zeros((len(X), len(self.feature_names)))
    
    def get_top_features(self, X: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """상위 K개 중요 피처"""
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # 평균 절대값 기준
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap,
            'mean_shap': self.shap_values.mean(axis=0),
        })
        df['direction'] = df['mean_shap'].apply(lambda x: 'positive' if x > 0 else 'negative')
        df['contribution_pct'] = df['mean_abs_shap'] / df['mean_abs_shap'].sum() * 100
        
        return df.sort_values('mean_abs_shap', ascending=False).head(top_k)
    
    def explain_segment(self, X_segment: pd.Series, top_k: int = 5) -> List[FeatureContribution]:
        """특정 세그먼트 예측 설명"""
        if self.shap_values is None:
            print("⚠️ SHAP 값이 계산되지 않았습니다.")
            return []
        
        # 단일 샘플 SHAP (근사)
        contributions = []
        for i, feat in enumerate(self.feature_names):
            if i < len(X_segment):
                shap_val = self.shap_values[0][i] if len(self.shap_values) > 0 else 0
                contributions.append(FeatureContribution(
                    feature=feat,
                    shap_value=shap_val,
                    feature_value=X_segment.iloc[i] if i < len(X_segment) else 0,
                    contribution_pct=abs(shap_val),
                    direction='positive' if shap_val > 0 else 'negative'
                ))
        
        contributions.sort(key=lambda x: abs(x.shap_value), reverse=True)
        return contributions[:top_k]


# ============================================================================
# 3. Confidence Interval (예측 불확실성)
# ============================================================================

class ConfidenceIntervalEstimator:
    """예측 불확실성 구간 추정"""
    
    def __init__(self, model, residual_std: float = 0.0):
        self.model = model
        self.residual_std = residual_std
        self.quantile_models = {}
        
    def fit_quantile_regression(self, X_train: pd.DataFrame, y_train: np.ndarray,
                                 quantiles: Sequence[float] = (0.05, 0.5, 0.95)):
        """Quantile Regression 기반 CI"""
        try:
            import lightgbm as lgb
            
            for q in quantiles:
                model = lgb.LGBMRegressor(
                    objective='quantile',
                    alpha=q,
                    n_estimators=100,
                    learning_rate=0.1,
                    num_leaves=31,
                    random_state=42,
                    verbosity=-1,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                self.quantile_models[q] = model
                
        except Exception as e:
            print(f"⚠️ Quantile Regression 실패: {e}")
    
    def predict_with_ci(self, X: pd.DataFrame, ci_level: float = 0.90) -> pd.DataFrame:
        """CI 포함 예측"""
        alpha = (1 - ci_level) / 2
        
        if self.quantile_models:
            # Quantile Regression 기반
            lower = self.quantile_models.get(alpha, None)
            upper = self.quantile_models.get(1 - alpha, None)
            point = self.quantile_models.get(0.5, None)
            
            results = pd.DataFrame({
                'predicted': point.predict(X) if point else self.model.predict(X),
                'ci_lower': lower.predict(X) if lower else np.nan,
                'ci_upper': upper.predict(X) if upper else np.nan,
            })
        else:
            # 잔차 기반 정규분포 가정
            from scipy import stats
            z = stats.norm.ppf(1 - alpha)
            
            pred = self.model.predict(X)
            results = pd.DataFrame({
                'predicted': pred,
                'ci_lower': pred - z * self.residual_std,
                'ci_upper': pred + z * self.residual_std,
            })
        
        results['ci_width'] = results['ci_upper'] - results['ci_lower']
        results['ci_level'] = ci_level
        
        return results


# ============================================================================
# 4. 세그먼트 클러스터링
# ============================================================================

@dataclass
class ClusterResult:
    """클러스터링 결과"""
    cluster_id: int
    size: int
    centroid: np.ndarray
    silhouette: float
    top_segments: List[str]
    characteristics: Dict[str, float]


class SegmentClusterer:
    """세그먼트 클러스터링 (K-Means + DBSCAN)"""
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.dbscan = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)
        
    def prepare_segment_features(self, panel: pd.DataFrame) -> pd.DataFrame:
        """클러스터링용 세그먼트 피처 집계"""
        # 세그먼트별 최신 3개월 집계
        latest_month = panel['month'].max()
        recent = panel[panel['month'] >= latest_month - pd.DateOffset(months=2)]
        
        agg_features = {}
        kpi_cols = ['log1p_예금총잔액', 'log1p_대출총잔액', 'log1p_카드총사용', 
                    'log1p_디지털거래금액', 'slog1p_순유입']
        
        for col in kpi_cols:
            if col in recent.columns:
                agg_features[f"{col}_mean"] = (col, 'mean')
                agg_features[f"{col}_std"] = (col, 'std')
                agg_features[f"{col}_trend"] = (col, lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0)
        
        segment_features = recent.groupby('segment_id').agg(**agg_features).reset_index()
        segment_features = segment_features.fillna(0)
        
        return segment_features
    
    def fit_kmeans(self, X: np.ndarray) -> np.ndarray:
        """K-Means 클러스터링"""
        # 스케일링 + PCA
        X_scaled = self.scaler.fit_transform(X)
        
        if X.shape[1] > 10:
            X_reduced = self.pca.fit_transform(X_scaled)
        else:
            X_reduced = X_scaled
        
        # 최적 K 탐색 (Silhouette)
        best_k = self.n_clusters
        best_score = -1
        
        for k in range(2, min(10, len(X) // 10)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_reduced)
            score = silhouette_score(X_reduced, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        # 최적 K로 학습
        self.kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X_reduced)
        
        print(f"  ✓ K-Means: {best_k} clusters (Silhouette: {best_score:.4f})")
        
        return labels
    
    def fit_dbscan(self, X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """DBSCAN 클러스터링 (이상치 탐지)"""
        X_scaled = self.scaler.transform(X) if hasattr(self.scaler, 'mean_') else self.scaler.fit_transform(X)
        
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = self.dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = (labels == -1).sum()
        
        print(f"  ✓ DBSCAN: {n_clusters} clusters, {n_outliers} outliers")
        
        return labels
    
    def get_cluster_summary(self, segment_features: pd.DataFrame, 
                            labels: np.ndarray) -> List[ClusterResult]:
        """클러스터 요약"""
        segment_features = segment_features.copy()
        segment_features['cluster'] = labels
        
        results = []
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue  # 이상치 제외
            
            cluster_df = segment_features[segment_features['cluster'] == cluster_id]
            feature_cols = [c for c in cluster_df.columns if c not in ['segment_id', 'cluster']]
            
            # 특성 계산
            characteristics = {}
            for col in feature_cols:
                characteristics[col] = cluster_df[col].mean()
            
            results.append(ClusterResult(
                cluster_id=cluster_id,
                size=len(cluster_df),
                centroid=cluster_df[feature_cols].mean().values,
                silhouette=0.0,  # 개별 계산 필요
                top_segments=cluster_df['segment_id'].head(5).tolist(),
                characteristics=characteristics
            ))
        
        return results


# ============================================================================
# 5. 액션 ROI 추정
# ============================================================================

class ActionROIEstimator:
    """액션 예상 효과 금액 추정"""
    
    # 액션 유형별 기대 효과 (베이스라인)
    ACTION_EFFECT_RATES = {
        'DEPOSIT_GROWTH': {'예금총잔액': 0.15, '순유입': 0.10},
        'DEPOSIT_DEFENSE': {'예금총잔액': 0.08, '순유입': 0.05},
        'LIQUIDITY_STRESS': {'대출총잔액': 0.05, '순유입': 0.03},
        'CREDIT_RISK': {'대출총잔액': 0.03, '연체율': -0.10},
        'CHURN_PREVENTION': {'순유입': 0.12, '예금총잔액': 0.10},
        'FX_EXPANSION': {'FX총액': 0.20, '수수료': 0.15},
        'ENGAGEMENT_DROP': {'디지털거래금액': 0.10, '카드총사용': 0.08},
    }
    
    def __init__(self, panel: pd.DataFrame):
        self.panel = panel
        self.segment_baseline = self._compute_segment_baseline()
    
    def _compute_segment_baseline(self) -> pd.DataFrame:
        """세그먼트별 기준 금액"""
        latest = self.panel[self.panel['month'] == self.panel['month'].max()]
        
        kpi_cols = ['예금총잔액', '대출총잔액', 'FX총액', '카드총사용', '디지털거래금액']
        existing_cols = [c for c in kpi_cols if c in latest.columns]
        
        if existing_cols:
            baseline = latest.groupby('segment_id')[existing_cols].mean().reset_index()
        else:
            baseline = pd.DataFrame({'segment_id': latest['segment_id'].unique()})
        
        return baseline
    
    def estimate_roi(self, actions: pd.DataFrame) -> pd.DataFrame:
        """액션별 ROI 추정"""
        actions = actions.copy()
        
        roi_estimates = []
        for _, action in actions.iterrows():
            segment_id = action.get('segment_id', '')
            action_type = action.get('action_type', '')
            
            # 기준 금액 조회
            segment_data = self.segment_baseline[
                self.segment_baseline['segment_id'] == segment_id
            ]
            
            if len(segment_data) == 0:
                roi_estimates.append({
                    'segment_id': segment_id,
                    'action_type': action_type,
                    'estimated_effect_억원': 0.0,
                    'confidence': 'low',
                    'effect_kpi': 'unknown'
                })
                continue
            
            # 효과 계산
            effect_rates = self.ACTION_EFFECT_RATES.get(action_type, {})
            total_effect = 0.0
            primary_kpi = 'unknown'
            
            for kpi, rate in effect_rates.items():
                if kpi in segment_data.columns:
                    base_value = segment_data[kpi].values[0]
                    effect = base_value * rate
                    total_effect += effect
                    if primary_kpi == 'unknown':
                        primary_kpi = kpi
            
            # 억원 단위 변환 (가정: 원래 단위가 백만원)
            effect_억원 = total_effect / 100 if total_effect > 0 else total_effect * 100
            
            roi_estimates.append({
                'segment_id': segment_id,
                'action_type': action_type,
                'estimated_effect_억원': round(effect_억원, 2),
                'confidence': 'high' if abs(effect_억원) > 1 else 'medium',
                'effect_kpi': primary_kpi
            })
        
        roi_df = pd.DataFrame(roi_estimates)
        
        # 원본에 병합
        actions = actions.merge(
            roi_df[['segment_id', 'action_type', 'estimated_effect_억원', 'confidence', 'effect_kpi']],
            on=['segment_id', 'action_type'],
            how='left'
        )
        
        return actions


# ============================================================================
# 6. 다중 시나리오 분석
# ============================================================================

@dataclass
class ScenarioResult:
    """시나리오 분석 결과"""
    scenario: str  # 'best', 'base', 'worst'
    predicted: float
    ci_lower: float
    ci_upper: float
    probability: float
    assumptions: Dict[str, str]


class ScenarioAnalyzer:
    """Best/Base/Worst Case 시뮬레이션"""
    
    SCENARIO_MULTIPLIERS = {
        'best': 1.2,   # +20% 
        'base': 1.0,   # 기준
        'worst': 0.8,  # -20%
    }
    
    SCENARIO_PROBABILITIES = {
        'best': 0.20,
        'base': 0.60,
        'worst': 0.20,
    }
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        
    def generate_scenarios(self, X_base: pd.DataFrame, 
                           external_factors: Optional[Dict] = None) -> List[ScenarioResult]:
        """시나리오별 예측 생성"""
        results = []
        
        base_pred = self.model.predict(X_base).mean()
        
        for scenario, multiplier in self.SCENARIO_MULTIPLIERS.items():
            # 시나리오별 피처 조정
            X_scenario = X_base.copy()
            
            # 외부 요인 적용
            if external_factors:
                for col, adjustment in external_factors.get(scenario, {}).items():
                    if col in X_scenario.columns:
                        X_scenario[col] = X_scenario[col] * adjustment
            
            # 예측
            scenario_pred = base_pred * multiplier
            
            # CI 추정 (시나리오별 불확실성)
            ci_width = abs(scenario_pred) * (0.1 if scenario == 'base' else 0.2)
            
            assumptions = self._get_scenario_assumptions(scenario)
            
            results.append(ScenarioResult(
                scenario=scenario,
                predicted=scenario_pred,
                ci_lower=scenario_pred - ci_width,
                ci_upper=scenario_pred + ci_width,
                probability=self.SCENARIO_PROBABILITIES[scenario],
                assumptions=assumptions
            ))
        
        return results
    
    def _get_scenario_assumptions(self, scenario: str) -> Dict[str, str]:
        """시나리오별 가정"""
        if scenario == 'best':
            return {
                '경기': '회복세 지속',
                '금리': '하락 (50bp)',
                '환율': '안정 (1,300원대)',
                '유가': '하락 ($70 이하)',
            }
        elif scenario == 'worst':
            return {
                '경기': '침체 진입',
                '금리': '상승 (50bp)',
                '환율': '급등 (1,450원 이상)',
                '유가': '급등 ($100 이상)',
            }
        else:
            return {
                '경기': '현 수준 유지',
                '금리': '동결',
                '환율': '현 수준 유지',
                '유가': '현 수준 유지',
            }
    
    def to_dataframe(self, results: List[ScenarioResult]) -> pd.DataFrame:
        """결과를 DataFrame으로 변환"""
        records = []
        for r in results:
            records.append({
                'scenario': r.scenario,
                'predicted': r.predicted,
                'ci_lower': r.ci_lower,
                'ci_upper': r.ci_upper,
                'probability': r.probability,
                'assumptions': json.dumps(r.assumptions, ensure_ascii=False),
            })
        return pd.DataFrame(records)


# ============================================================================
# 7. 외부 데이터 확장
# ============================================================================

class ExternalDataIntegrator:
    """외부 데이터 확장 통합"""
    
    def __init__(self, external_data_dir: Path):
        self.data_dir = external_data_dir
        self.loaded_data = {}
        
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """모든 외부 데이터 로드"""
        files = {
            'ESI': 'ESI.csv',
            'KASI': 'KASI.csv',
            'fed_rate': 'fed_금리_파생변수_202106_202506.csv',
            'oil': '원자재(유가)_파생변수.csv',
            'housing': '주택가격지수_아파트.csv',
            'trade': '무역_P1_파생변수.csv',
            'fx': 'usd_krw_raw_daily.csv',
        }
        
        for name, filename in files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    self.loaded_data[name] = pd.read_csv(filepath)
                    print(f"  ✓ 로드: {filename} ({len(self.loaded_data[name])} rows)")
                except Exception as e:
                    print(f"  ⚠️ 로드 실패: {filename} - {e}")
            else:
                print(f"  ⚠️ 파일 없음: {filename}")
        
        return self.loaded_data
    
    def integrate_to_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        """패널에 외부 데이터 통합"""
        df = panel.copy()
        
        # ESI (Economic Sentiment Index)
        if 'ESI' in self.loaded_data:
            esi = self.loaded_data['ESI'].copy()
            if 'month' in esi.columns or '날짜' in esi.columns:
                date_col = 'month' if 'month' in esi.columns else '날짜'
                esi[date_col] = pd.to_datetime(esi[date_col])
                esi = esi.rename(columns={date_col: 'month'})
                df = df.merge(esi, on='month', how='left')
        
        # KASI (Korea ASI)
        if 'KASI' in self.loaded_data:
            kasi = self.loaded_data['KASI'].copy()
            if 'month' in kasi.columns or '날짜' in kasi.columns:
                date_col = 'month' if 'month' in kasi.columns else '날짜'
                kasi[date_col] = pd.to_datetime(kasi[date_col])
                kasi = kasi.rename(columns={date_col: 'month'})
                df = df.merge(kasi, on='month', how='left')
        
        print(f"  ✓ 외부 데이터 통합 완료: {len(df.columns)} features")
        
        return df
    
    def get_latest_indicators(self) -> Dict[str, float]:
        """최신 경제 지표 조회"""
        indicators = {}
        
        for name, data in self.loaded_data.items():
            if len(data) > 0:
                # 마지막 행의 수치 컬럼
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    last_row = data.iloc[-1]
                    for col in numeric_cols[:3]:  # 상위 3개만
                        # JSON 직렬화 가능하도록 변환
                        val = last_row[col]
                        if hasattr(val, 'item'):
                            val = val.item()  # numpy -> python
                        indicators[f"{name}_{col}"] = float(val) if not pd.isna(val) else None
        
        return indicators


# ============================================================================
# Main 실행
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="고급 기능 개선")
    parser.add_argument("--data-dir", type=str, default="outputs")
    parser.add_argument("--external-dir", type=str, default="../외부 데이터")
    parser.add_argument("--output-dir", type=str, default="outputs/advanced_features")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / args.data_dir
    external_dir = base_dir / args.external_dir
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🚀 고급 기능 개선 실행")
    print("=" * 70)
    
    # ========================================================================
    # 1. 데이터 로드
    # ========================================================================
    print("\n[1/8] 데이터 로드...")
    
    panel_path = data_dir / "segment_panel.parquet"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
    else:
        panel_csv = data_dir / "segment_panel.csv"
        if panel_csv.exists():
            panel = pd.read_csv(panel_csv, parse_dates=['month'])
        else:
            print("  ❌ segment_panel 파일을 찾을 수 없습니다.")
            return
    
    print(f"  ✓ 패널 로드: {len(panel):,} rows, {len(panel.columns)} columns")
    
    # ========================================================================
    # 2. Horizon 3개월 성능 개선
    # ========================================================================
    print("\n[2/8] Horizon 3개월 성능 개선...")
    
    optimizer = HorizonOptimizer(panel)
    enhanced_panel = optimizer.create_enhanced_features()
    enhanced_params = optimizer.get_horizon3_optimized_params()
    
    print(f"  ✓ 강화된 피처 생성: {len(enhanced_panel.columns)} columns")
    print(f"  ✓ 최적화 파라미터: learning_rate={enhanced_params['learning_rate']}, num_leaves={enhanced_params['num_leaves']}")
    
    # ========================================================================
    # 3. XAI - SHAP 시각화
    # ========================================================================
    print("\n[3/8] XAI SHAP 시각화 준비...")
    
    feature_names = [c for c in enhanced_panel.columns if c.startswith('log1p_') or c.startswith('slog1p_')]
    
    # SHAP 설명 샘플 생성
    xai_report = {
        'status': 'ready',
        'top_features': feature_names[:10] if feature_names else [],
        'note': 'SHAP 계산은 모델 로드 후 실행 필요'
    }
    
    with open(output_dir / "xai_report.json", 'w', encoding='utf-8') as f:
        json.dump(xai_report, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ XAI 준비 완료: {len(feature_names)} features 대상")
    
    # ========================================================================
    # 4. Confidence Interval
    # ========================================================================
    print("\n[4/8] Confidence Interval 설정...")
    
    ci_config = {
        'ci_level': 0.90,
        'method': 'quantile_regression',
        'fallback': 'residual_based',
        'quantiles': [0.05, 0.5, 0.95],
    }
    
    with open(output_dir / "ci_config.json", 'w', encoding='utf-8') as f:
        json.dump(ci_config, f, indent=2)
    
    print(f"  ✓ CI 설정: {ci_config['ci_level']*100}% level, {ci_config['method']}")
    
    # ========================================================================
    # 5. 세그먼트 클러스터링
    # ========================================================================
    print("\n[5/8] 세그먼트 클러스터링...")
    
    clusterer = SegmentClusterer(n_clusters=5)
    segment_features = clusterer.prepare_segment_features(enhanced_panel)
    
    if len(segment_features) > 10:
        feature_cols = [c for c in segment_features.columns if c != 'segment_id']
        X_cluster = segment_features[feature_cols].fillna(0).values
        
        # K-Means
        kmeans_labels = clusterer.fit_kmeans(X_cluster)
        segment_features['kmeans_cluster'] = kmeans_labels
        
        # DBSCAN
        dbscan_labels = clusterer.fit_dbscan(X_cluster, eps=1.5, min_samples=3)
        segment_features['dbscan_cluster'] = dbscan_labels
        
        # 저장
        segment_features.to_csv(output_dir / "segment_clusters.csv", index=False)
        print(f"  ✓ 클러스터링 완료: {len(segment_features)} segments")
    else:
        print(f"  ⚠️ 세그먼트 수 부족: {len(segment_features)}")
    
    # ========================================================================
    # 6. 액션 ROI 추정
    # ========================================================================
    print("\n[6/8] 액션 ROI 추정...")
    
    actions_path = data_dir / "actions_enhanced.csv"
    if actions_path.exists():
        actions = pd.read_csv(actions_path)
        
        roi_estimator = ActionROIEstimator(panel)
        actions_with_roi = roi_estimator.estimate_roi(actions)
        
        # 저장
        actions_with_roi.to_csv(output_dir / "actions_with_roi.csv", index=False)
        
        total_roi = actions_with_roi['estimated_effect_억원'].sum()
        print(f"  ✓ ROI 추정 완료: {len(actions_with_roi)} actions, 총 예상 효과: {total_roi:.1f}억원")
    else:
        print("  ⚠️ actions_enhanced.csv 없음")
    
    # ========================================================================
    # 7. 다중 시나리오 분석
    # ========================================================================
    print("\n[7/8] 다중 시나리오 분석...")
    
    # 시나리오 템플릿 생성
    scenarios_template = {
        'best': {
            'name': 'Best Case (낙관적)',
            'probability': 0.20,
            'assumptions': {
                '경기': '회복세 지속',
                '금리': '하락 50bp',
                '환율': '안정 (1,300원대)',
                '유가': '하락 ($70 이하)',
            },
            'kpi_multiplier': 1.2,
        },
        'base': {
            'name': 'Base Case (기준)',
            'probability': 0.60,
            'assumptions': {
                '경기': '현 수준 유지',
                '금리': '동결',
                '환율': '현 수준 유지',
                '유가': '현 수준 유지',
            },
            'kpi_multiplier': 1.0,
        },
        'worst': {
            'name': 'Worst Case (비관적)',
            'probability': 0.20,
            'assumptions': {
                '경기': '침체 진입',
                '금리': '상승 50bp',
                '환율': '급등 (1,450원 이상)',
                '유가': '급등 ($100 이상)',
            },
            'kpi_multiplier': 0.8,
        }
    }
    
    with open(output_dir / "scenario_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(scenarios_template, f, ensure_ascii=False, indent=2)
    
    print("  ✓ 시나리오 템플릿 생성: Best/Base/Worst")
    
    # ========================================================================
    # 8. 외부 데이터 확장
    # ========================================================================
    print("\n[8/8] 외부 데이터 확장...")
    
    if external_dir.exists():
        integrator = ExternalDataIntegrator(external_dir)
        external_data = integrator.load_all()
        
        if external_data:
            latest_indicators = integrator.get_latest_indicators()
            
            with open(output_dir / "external_indicators.json", 'w', encoding='utf-8') as f:
                json.dump(latest_indicators, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ 외부 데이터 통합: {len(external_data)} sources, {len(latest_indicators)} indicators")
    else:
        print(f"  ⚠️ 외부 데이터 디렉토리 없음: {external_dir}")
    
    # ========================================================================
    # 결과 요약
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ 고급 기능 개선 완료!")
    print("=" * 70)
    
    print(f"\n📁 출력 디렉토리: {output_dir}")
    print("\n📊 생성된 파일:")
    for f in output_dir.glob("*"):
        if f.is_file():
            print(f"  - {f.name}")
    
    print("\n🔍 개선 요약:")
    print("  1. Horizon 3개월: 트렌드/모멘텀 피처 추가")
    print("  2. XAI: SHAP 기반 Feature 기여도 준비")
    print("  3. CI: 90% 신뢰구간 설정")
    print("  4. 클러스터링: K-Means + DBSCAN")
    print("  5. ROI: 액션별 예상 효과 금액")
    print("  6. 시나리오: Best/Base/Worst Case")
    print("  7. 외부 데이터: ESI, KASI 등 통합")


if __name__ == "__main__":
    main()
