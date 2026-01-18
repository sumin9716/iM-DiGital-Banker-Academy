"""
임베딩 기반 추천 시스템

Features:
1. 세그먼트 임베딩 (ALS Matrix Factorization)
2. 코사인 유사도 기반 유사 세그먼트 탐색
3. KPI 패턴 기반 협업 필터링
4. 하이브리드 추천 (Content + Collaborative)
5. PCA 차원 축소 및 특성 가중치 적용 (NEW)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# KPI별 유사도 가중치 설정
# ============================================================================
# 중요도에 따라 유사도 계산 시 가중치 부여
# - 예금, 순유입: 가장 중요 (고객 가치 핵심 지표)
# - 카드, 디지털: 행동 패턴 지표
# - FX: 특수 거래 지표
KPI_SIMILARITY_WEIGHTS = {
    "예금총잔액": 2.0,
    "대출총잔액": 1.5,
    "순유입": 2.0,
    "카드총사용": 1.0,
    "디지털거래금액": 1.2,
    "FX총액": 0.8,
    # 파생 변수
    "한도소진율": 1.5,
    "디지털비중": 1.0,
    "자동이체비중": 0.8,
}

# 변화율 컬럼 (추세 유사성에 중요)
CHANGE_FEATURE_WEIGHTS = {
    "예금총잔액_pct_chg": 1.5,
    "대출총잔액_pct_chg": 1.2,
    "순유입_pct_chg": 1.5,
    "카드총사용_pct_chg": 1.0,
}


@dataclass
class SimilarSegment:
    """유사 세그먼트 정보"""
    segment_id: str
    similarity: float
    shared_patterns: List[str]
    recommended_actions: List[str]


@dataclass
class SegmentEmbedding:
    """세그먼트 임베딩 결과"""
    segment_id: str
    embedding: np.ndarray
    cluster_id: int = -1


class ALSEmbedder:
    """
    ALS (Alternating Least Squares) 기반 세그먼트 임베딩
    
    세그먼트-KPI 행렬을 저차원 latent factor로 분해하여
    세그먼트 간 유사성을 파악
    """
    
    def __init__(
        self,
        n_factors: int = 32,
        n_iterations: int = 20,
        regularization: float = 0.1,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.random_state = random_state
        
        self.segment_factors: Optional[np.ndarray] = None
        self.kpi_factors: Optional[np.ndarray] = None
        self.segment_ids: Optional[List[str]] = None
        self.kpi_names: Optional[List[str]] = None
        self.scaler = StandardScaler()
        
    def fit(self, panel: pd.DataFrame, kpi_cols: List[str]) -> 'ALSEmbedder':
        """
        세그먼트-KPI 패널 데이터로 ALS 임베딩 학습
        
        Args:
            panel: segment_id, month, KPI 컬럼이 있는 패널
            kpi_cols: 임베딩에 사용할 KPI 컬럼명 리스트
        """
        # 최신 월 데이터만 사용하거나 평균 사용
        latest_month = panel['month'].max()
        recent_data = panel[panel['month'] >= latest_month - pd.DateOffset(months=6)]
        
        # 세그먼트별 KPI 평균
        agg_data = recent_data.groupby('segment_id')[kpi_cols].mean().reset_index()
        
        self.segment_ids = agg_data['segment_id'].tolist()
        self.kpi_names = kpi_cols
        
        # KPI 매트릭스 (세그먼트 x KPI)
        X = agg_data[kpi_cols].fillna(0).values
        
        # 정규화
        X_scaled = self.scaler.fit_transform(X)
        
        # ALS 수행
        self.segment_factors, self.kpi_factors = self._als(
            X_scaled, 
            self.n_factors, 
            self.n_iterations, 
            self.regularization
        )
        
        return self
    
    def _als(
        self, 
        R: np.ndarray, 
        k: int, 
        n_iter: int, 
        reg: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alternating Least Squares 알고리즘
        
        R ≈ U @ V.T
        - U: 세그먼트 latent factors (n_segments x k)
        - V: KPI latent factors (n_kpis x k)
        """
        np.random.seed(self.random_state)
        n_segments, n_kpis = R.shape
        
        # 초기화
        U = np.random.normal(0, 0.1, (n_segments, k))
        V = np.random.normal(0, 0.1, (n_kpis, k))
        
        for iteration in range(n_iter):
            # Fix V, solve for U
            for i in range(n_segments):
                U[i] = np.linalg.solve(
                    V.T @ V + reg * np.eye(k),
                    V.T @ R[i]
                )
            
            # Fix U, solve for V
            for j in range(n_kpis):
                V[j] = np.linalg.solve(
                    U.T @ U + reg * np.eye(k),
                    U.T @ R[:, j]
                )
        
        return U, V
    
    def get_embedding(self, segment_id: str) -> Optional[np.ndarray]:
        """특정 세그먼트의 임베딩 벡터 반환"""
        if self.segment_factors is None or segment_id not in self.segment_ids:
            return None
        idx = self.segment_ids.index(segment_id)
        return self.segment_factors[idx]
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """모든 세그먼트 임베딩 반환"""
        if self.segment_factors is None:
            return {}
        return {
            seg_id: self.segment_factors[i] 
            for i, seg_id in enumerate(self.segment_ids)
        }


class SegmentSimilarityEngine:
    """
    세그먼트 유사도 엔진
    
    코사인 유사도 기반으로 유사 세그먼트를 찾고
    성공적인 액션 패턴을 추천
    
    개선사항:
    - PCA 차원 축소로 노이즈 제거
    - KPI별 가중치 적용
    - RobustScaler로 이상치 영향 최소화
    """
    
    def __init__(self, embedder: Optional[ALSEmbedder] = None):
        self.embedder = embedder
        self.similarity_matrix: Optional[np.ndarray] = None
        self.segment_ids: Optional[List[str]] = None
        self.pca: Optional[PCA] = None
        self.feature_names: Optional[List[str]] = None
        
        # 성공 액션 이력 (세그먼트별)
        self.success_actions: Dict[str, List[Dict]] = {}
        
    def build_similarity_matrix(
        self, 
        panel: pd.DataFrame, 
        kpi_cols: List[str],
        use_embedding: bool = True,
        use_pca: bool = True,
        pca_variance_ratio: float = 0.95,
        use_weighted: bool = True,
    ) -> 'SegmentSimilarityEngine':
        """
        유사도 행렬 구축 (개선 버전)
        
        Args:
            panel: 세그먼트 패널 데이터
            kpi_cols: KPI 컬럼
            use_embedding: ALS 임베딩 사용 여부 (False면 원본 피처 사용)
            use_pca: PCA 차원 축소 적용 여부
            pca_variance_ratio: PCA 설명 분산 비율 (0.95 = 95%)
            use_weighted: KPI별 가중치 적용 여부
        """
        if use_embedding and self.embedder is not None:
            # 임베딩 기반 유사도
            embeddings = self.embedder.get_all_embeddings()
            self.segment_ids = list(embeddings.keys())
            emb_matrix = np.array([embeddings[s] for s in self.segment_ids])
            self.similarity_matrix = cosine_similarity(emb_matrix)
        else:
            # ============== 개선된 KPI 기반 유사도 ==============
            latest_month = panel['month'].max()
            recent = panel[panel['month'] >= latest_month - pd.DateOffset(months=6)]
            
            # 핵심 특성만 선택 (KPI + 변화율 + 파생변수)
            # 원본 KPI보다 변화율/비율 특성에 더 집중 (변별력 향상)
            core_kpi_cols = [c for c in kpi_cols if c in recent.columns]
            
            # 변화율 컬럼 (가장 중요 - 패턴 유사성)
            change_cols = [c for c in recent.columns if '_pct_chg' in c or '_chg' in c]
            
            # 파생변수 (비율 지표)
            derived_cols = [c for c in ['한도소진율', '디지털비중', '자동이체비중'] if c in recent.columns]
            
            # 최종 특성: 변화율 우선, 파생변수, KPI 순
            all_feature_cols = change_cols + derived_cols + core_kpi_cols
            all_feature_cols = list(dict.fromkeys(all_feature_cols))  # 중복 제거
            
            self.feature_names = all_feature_cols
            
            agg = recent.groupby('segment_id')[all_feature_cols].mean().fillna(0)
            self.segment_ids = agg.index.tolist()
            
            X = agg.values
            
            # 1. KPI별 가중치 적용
            if use_weighted:
                weights = np.array([
                    KPI_SIMILARITY_WEIGHTS.get(col, CHANGE_FEATURE_WEIGHTS.get(col, 1.0))
                    for col in all_feature_cols
                ])
                X = X * weights
            
            # 2. 로그 스케일 + RobustScaler로 분포 정규화
            # 금융 데이터는 log-normal 분포가 많으므로 로그 변환
            X_log = np.sign(X) * np.log1p(np.abs(X))
            
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_log)
            
            # 3. PCA 차원 축소 (노이즈 제거 및 변별력 향상)
            if use_pca and X_scaled.shape[1] > 5:
                # 최소 5개 주성분 보장 (변별력 확보)
                min_components = min(10, X_scaled.shape[1] // 3)
                
                # 먼저 전체 PCA로 분산 분석
                pca_full = PCA()
                pca_full.fit(X_scaled)
                
                # 목표 분산 비율에 도달하는 최소 성분 수 계산
                cumsum = np.cumsum(pca_full.explained_variance_ratio_)
                n_for_variance = np.argmax(cumsum >= pca_variance_ratio) + 1
                
                # 최소 주성분 수 보장
                n_components = max(min_components, n_for_variance)
                n_components = min(n_components, X_scaled.shape[1])
                
                self.pca = PCA(n_components=n_components)
                X_transformed = self.pca.fit_transform(X_scaled)
                
                print(f"     PCA: {X_scaled.shape[1]}개 특성 → {n_components}개 주성분 "
                      f"(설명분산 {self.pca.explained_variance_ratio_.sum():.1%})")
            else:
                X_transformed = X_scaled
            
            # 4. 유클리디안 거리를 유사도로 변환 (변별력 향상)
            from sklearn.metrics.pairwise import euclidean_distances
            
            dist_matrix = euclidean_distances(X_transformed)
            
            # 거리를 유사도로 변환: sim = 1 / (1 + dist)
            # 또는 가우시안 커널: sim = exp(-dist^2 / (2 * sigma^2))
            sigma = np.median(dist_matrix[dist_matrix > 0])  # 중앙값을 sigma로 사용
            self.similarity_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2 + 1e-9))
            
            # 유사도 분포 로깅
            upper_tri = self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)]
            print(f"     유사도 분포: min={upper_tri.min():.3f}, "
                  f"median={np.median(upper_tri):.3f}, max={upper_tri.max():.3f}")
        
        return self
    
    def find_similar_segments(
        self, 
        segment_id: str, 
        top_k: int = 10,
        min_similarity: float = 0.5,
    ) -> List[SimilarSegment]:
        """
        유사 세그먼트 탐색
        
        Args:
            segment_id: 대상 세그먼트
            top_k: 반환할 유사 세그먼트 수
            min_similarity: 최소 유사도 임계값
        """
        if self.similarity_matrix is None or segment_id not in self.segment_ids:
            return []
        
        idx = self.segment_ids.index(segment_id)
        similarities = self.similarity_matrix[idx]
        
        # 자기 자신 제외
        similarities[idx] = -1
        
        # Top-K 인덱스
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i in top_indices:
            sim = similarities[i]
            if sim < min_similarity:
                continue
            
            similar_seg = self.segment_ids[i]
            
            # 공유 패턴 분석
            shared = self._analyze_shared_patterns(segment_id, similar_seg)
            
            # 추천 액션 (유사 세그먼트의 성공 액션 참조)
            actions = self.success_actions.get(similar_seg, [])
            action_titles = [a.get('title', '') for a in actions[:3]]
            
            results.append(SimilarSegment(
                segment_id=similar_seg,
                similarity=float(sim),
                shared_patterns=shared,
                recommended_actions=action_titles,
            ))
        
        return results
    
    def _analyze_shared_patterns(
        self, 
        seg1: str, 
        seg2: str
    ) -> List[str]:
        """두 세그먼트 간 공유 패턴 분석"""
        patterns = []
        
        # 세그먼트 ID 파싱 (업종|지역|등급|전담여부)
        parts1 = seg1.split('|')
        parts2 = seg2.split('|')
        
        if len(parts1) >= 4 and len(parts2) >= 4:
            if parts1[0] == parts2[0]:
                patterns.append(f"동일 업종: {parts1[0]}")
            if parts1[1] == parts2[1] and parts1[1] != 'nan':
                patterns.append(f"동일 지역: {parts1[1]}")
            if parts1[2] == parts2[2]:
                patterns.append(f"동일 등급: {parts1[2]}")
            if parts1[3] == parts2[3]:
                patterns.append(f"전담여부: {parts1[3]}")
        
        return patterns
    
    def register_success_action(
        self, 
        segment_id: str, 
        action: Dict
    ) -> None:
        """성공한 액션 등록 (협업 필터링용)"""
        if segment_id not in self.success_actions:
            self.success_actions[segment_id] = []
        self.success_actions[segment_id].append(action)


class HybridRecommender:
    """
    하이브리드 추천 시스템
    
    1. Content-based: 세그먼트 특성 기반 규칙
    2. Collaborative: 유사 세그먼트 성공 액션
    3. Embedding-based: ALS 임베딩 유사도
    """
    
    def __init__(
        self,
        n_factors: int = 32,
        content_weight: float = 0.4,
        collab_weight: float = 0.3,
        embedding_weight: float = 0.3,
    ):
        self.n_factors = n_factors
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        self.embedding_weight = embedding_weight
        
        self.embedder: Optional[ALSEmbedder] = None
        self.similarity_engine: Optional[SegmentSimilarityEngine] = None
        self.is_fitted = False
        
    def fit(
        self, 
        panel: pd.DataFrame, 
        kpi_cols: List[str],
        success_actions: Optional[Dict[str, List[Dict]]] = None,
    ) -> 'HybridRecommender':
        """
        하이브리드 추천 모델 학습
        
        Args:
            panel: 세그먼트 패널 데이터
            kpi_cols: KPI 컬럼 리스트
            success_actions: 세그먼트별 성공 액션 이력
        """
        # ALS 임베딩 학습
        self.embedder = ALSEmbedder(n_factors=self.n_factors)
        self.embedder.fit(panel, kpi_cols)
        
        # 유사도 엔진 구축
        self.similarity_engine = SegmentSimilarityEngine(self.embedder)
        self.similarity_engine.build_similarity_matrix(panel, kpi_cols, use_embedding=True)
        
        # 성공 액션 등록
        if success_actions:
            for seg_id, actions in success_actions.items():
                for action in actions:
                    self.similarity_engine.register_success_action(seg_id, action)
        
        self.is_fitted = True
        return self
    
    def recommend(
        self,
        segment_id: str,
        current_kpis: Dict[str, float],
        content_actions: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        하이브리드 추천 생성
        
        Args:
            segment_id: 대상 세그먼트
            current_kpis: 현재 KPI 값
            content_actions: Content-based 추천 액션 (규칙 기반)
            top_k: 반환할 추천 수
        
        Returns:
            하이브리드 점수로 랭킹된 추천 액션 리스트
        """
        if not self.is_fitted:
            return content_actions[:top_k]
        
        # 유사 세그먼트 탐색
        similar_segments = self.similarity_engine.find_similar_segments(
            segment_id, top_k=20, min_similarity=0.3
        )
        
        # 협업 필터링 액션 수집
        collab_actions = []
        for sim_seg in similar_segments:
            for action_title in sim_seg.recommended_actions:
                if action_title:
                    collab_actions.append({
                        'title': action_title,
                        'source_segment': sim_seg.segment_id,
                        'similarity': sim_seg.similarity,
                        'score': sim_seg.similarity * 100,
                    })
        
        # 하이브리드 점수 계산
        hybrid_actions = []
        
        # Content-based 액션에 점수 부여
        for action in content_actions:
            hybrid_score = action.get('score', 50) * self.content_weight
            
            # 유사 세그먼트에서도 추천된 경우 가산점
            action_type = action.get('action_type', '')
            for ca in collab_actions:
                if action_type in ca.get('title', ''):
                    hybrid_score += ca['similarity'] * 50 * self.collab_weight
                    break
            
            action['hybrid_score'] = hybrid_score
            action['recommendation_source'] = 'content + collab'
            hybrid_actions.append(action)
        
        # 순수 협업 필터링 액션 추가
        for ca in collab_actions[:3]:
            if not any(ca['title'] in str(a.get('title', '')) for a in hybrid_actions):
                ca['hybrid_score'] = ca['score'] * self.collab_weight
                ca['recommendation_source'] = f"similar: {ca['source_segment'][:30]}..."
                hybrid_actions.append(ca)
        
        # 점수순 정렬
        hybrid_actions.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
        
        return hybrid_actions[:top_k]
    
    def get_segment_embedding(self, segment_id: str) -> Optional[np.ndarray]:
        """세그먼트 임베딩 반환"""
        if self.embedder:
            return self.embedder.get_embedding(segment_id)
        return None
    
    def find_similar_segments(
        self, 
        segment_id: str, 
        top_k: int = 5
    ) -> List[SimilarSegment]:
        """유사 세그먼트 탐색"""
        if self.similarity_engine:
            return self.similarity_engine.find_similar_segments(segment_id, top_k)
        return []


# === 유틸리티 함수 ===

def compute_segment_similarity_report(
    panel: pd.DataFrame,
    kpi_cols: List[str],
    target_segments: Optional[List[str]] = None,
    top_k: int = 5,
    use_pca: bool = True,
    min_similarity: float = 0.3,
) -> pd.DataFrame:
    """
    세그먼트 유사도 리포트 생성 (개선 버전)
    
    Args:
        panel: 세그먼트 패널
        kpi_cols: KPI 컬럼
        target_segments: 분석 대상 세그먼트 (None이면 전체)
        top_k: 각 세그먼트별 반환할 유사 세그먼트 수
        use_pca: PCA 차원 축소 적용 여부
        min_similarity: 최소 유사도 임계값 (변별력 확보)
    
    Returns:
        유사도 리포트 DataFrame
    """
    print("  └─ Computing improved similarity with PCA and weighted features...")
    
    # 임베딩 학습
    embedder = ALSEmbedder(n_factors=32)
    embedder.fit(panel, kpi_cols)
    
    # 유사도 엔진 (PCA + 가중치 적용)
    engine = SegmentSimilarityEngine(embedder)
    engine.build_similarity_matrix(
        panel, 
        kpi_cols, 
        use_embedding=False,  # PCA 기반 사용
        use_pca=use_pca,
        pca_variance_ratio=0.80,  # 80% 분산 유지 (더 많은 주성분 유지)
        use_weighted=True,
    )
    
    if target_segments is None:
        target_segments = engine.segment_ids[:100] if engine.segment_ids else []
    
    results = []
    for seg_id in target_segments:
        # min_similarity를 0.1로 낮춰서 더 다양한 범위 포함
        similar = engine.find_similar_segments(seg_id, top_k=top_k, min_similarity=0.1)
        for rank, sim_seg in enumerate(similar, 1):
            results.append({
                'segment_id': seg_id,
                'rank': rank,
                'similar_segment': sim_seg.segment_id,
                'similarity': sim_seg.similarity,
                'shared_patterns': ' | '.join(sim_seg.shared_patterns),
            })
    
    if results:
        df = pd.DataFrame(results)
        # 유사도 분포 통계
        print(f"     유사도 결과: {len(df)}개 쌍, "
              f"평균={df['similarity'].mean():.3f}, "
              f"min={df['similarity'].min():.3f}, max={df['similarity'].max():.3f}")
        return df
    
    return pd.DataFrame(columns=['segment_id', 'rank', 'similar_segment', 'similarity', 'shared_patterns'])
    
    return pd.DataFrame(results)


def enrich_actions_with_collaborative(
    actions: pd.DataFrame,
    panel: pd.DataFrame,
    kpi_cols: List[str],
) -> pd.DataFrame:
    """
    액션 DataFrame에 협업 필터링 정보 추가
    
    Args:
        actions: 기존 액션 DataFrame
        panel: 세그먼트 패널
        kpi_cols: KPI 컬럼
    
    Returns:
        유사 세그먼트 정보가 추가된 액션 DataFrame
    """
    if actions.empty:
        return actions
    
    # 임베딩 학습
    embedder = ALSEmbedder(n_factors=32)
    embedder.fit(panel, kpi_cols)
    
    engine = SegmentSimilarityEngine(embedder)
    engine.build_similarity_matrix(panel, kpi_cols, use_embedding=True)
    
    # 각 액션에 유사 세그먼트 정보 추가
    similar_info = []
    for _, row in actions.iterrows():
        seg_id = row.get('segment_id', '')
        similar = engine.find_similar_segments(seg_id, top_k=3, min_similarity=0.5)
        
        if similar:
            top_sim = similar[0]
            similar_info.append({
                'top_similar_segment': top_sim.segment_id[:40],
                'similarity_score': top_sim.similarity,
                'shared_patterns': ' | '.join(top_sim.shared_patterns[:2]),
            })
        else:
            similar_info.append({
                'top_similar_segment': '',
                'similarity_score': 0.0,
                'shared_patterns': '',
            })
    
    sim_df = pd.DataFrame(similar_info)
    return pd.concat([actions.reset_index(drop=True), sim_df], axis=1)
