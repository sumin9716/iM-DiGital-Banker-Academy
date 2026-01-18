"""
추천 엔진 패키지

Modules:
- action_engine: NBA(Next Best Action) 생성
- cause_analyzer: 원인 분석 및 요약
- embedding_recommender: ALS 임베딩 기반 협업 필터링
"""

from imradar.recommend.action_engine import (
    ActionEngine,
    RecommendedAction,
    SegmentProfile,
    ActionCategory,
    Urgency,
    build_segment_profile,
    generate_recommended_actions_for_segment,
    generate_action_reason,
)

from imradar.recommend.cause_analyzer import (
    CauseAnalysis,
    analyze_cause,
    generate_cause_summary,
    generate_watchlist_driver_summary,
    generate_risk_factor_summary,
    enrich_watchlist_with_drivers,
    format_feature_name,
    interpret_contribution,
)

from imradar.recommend.embedding_recommender import (
    ALSEmbedder,
    SegmentSimilarityEngine,
    HybridRecommender,
    SimilarSegment,
    SegmentEmbedding,
    compute_segment_similarity_report,
    enrich_actions_with_collaborative,
)


__all__ = [
    # Action Engine
    "ActionEngine",
    "RecommendedAction",
    "SegmentProfile",
    "ActionCategory",
    "Urgency",
    "build_segment_profile",
    "generate_recommended_actions_for_segment",
    "generate_action_reason",
    
    # Cause Analyzer
    "CauseAnalysis",
    "analyze_cause",
    "generate_cause_summary",
    "generate_watchlist_driver_summary",
    "generate_risk_factor_summary",
    "enrich_watchlist_with_drivers",
    "format_feature_name",
    "interpret_contribution",
]
