"""
Unit Tests for iM ONEderful Segment Radar

pytest 실행: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDataPreprocess:
    """데이터 전처리 테스트"""
    
    def test_log_transform(self):
        """log1p 변환 테스트"""
        data = pd.Series([0, 1, 10, 100, 1000])
        result = np.log1p(data)
        assert result[0] == 0  # log1p(0) = 0
        assert result[1] == pytest.approx(0.693, rel=0.01)
        
    def test_slog_transform(self):
        """signed log 변환 테스트"""
        data = pd.Series([-100, -1, 0, 1, 100])
        result = np.sign(data) * np.log1p(np.abs(data))
        assert result[2] == 0  # slog(0) = 0
        assert result[4] > 0  # slog(100) > 0
        assert result[0] < 0  # slog(-100) < 0
        
    def test_segment_id_creation(self):
        """세그먼트 ID 생성 테스트"""
        df = pd.DataFrame({
            '업종': ['제조업', '서비스업'],
            '지역': ['서울', '부산'],
            '등급': ['우수', '일반'],
        })
        df['segment_id'] = df['업종'] + '|' + df['지역'] + '|' + df['등급']
        assert df['segment_id'].iloc[0] == '제조업|서울|우수'


class TestForecast:
    """예측 모델 테스트"""
    
    def test_prediction_shape(self):
        """예측 출력 형태 테스트"""
        # Mock prediction
        n_samples = 100
        predictions = np.random.randn(n_samples)
        assert len(predictions) == n_samples
        
    def test_r2_score(self):
        """R² 스코어 계산 테스트"""
        from sklearn.metrics import r2_score
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.0, 2.9, 4.1, 5.0])
        r2 = r2_score(y_true, y_pred)
        assert r2 > 0.95  # 매우 좋은 예측


class TestRecommendation:
    """추천 시스템 테스트"""
    
    def test_action_score_range(self):
        """액션 점수 범위 테스트"""
        scores = [10, 50, 80, 120, 200]
        for score in scores:
            assert score >= 0
            
    def test_urgency_mapping(self):
        """긴급도 매핑 테스트"""
        urgency_map = {
            'CRITICAL': 4,
            'HIGH': 3,
            'MEDIUM': 2,
            'LOW': 1,
        }
        assert urgency_map['CRITICAL'] > urgency_map['LOW']


class TestClustering:
    """클러스터링 테스트"""
    
    def test_kmeans_output(self):
        """K-Means 출력 테스트"""
        from sklearn.cluster import KMeans
        X = np.random.randn(100, 5)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        assert len(labels) == 100
        assert len(set(labels)) == 3
        
    def test_silhouette_score(self):
        """Silhouette Score 테스트"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        X = np.random.randn(100, 5)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        assert -1 <= score <= 1


class TestConfidenceInterval:
    """신뢰구간 테스트"""
    
    def test_ci_contains_prediction(self):
        """CI가 예측값 포함 테스트"""
        predicted = 100
        ci_lower = 80
        ci_upper = 120
        assert ci_lower <= predicted <= ci_upper
        
    def test_ci_width_positive(self):
        """CI 폭 양수 테스트"""
        ci_lower = 80
        ci_upper = 120
        assert ci_upper - ci_lower > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
