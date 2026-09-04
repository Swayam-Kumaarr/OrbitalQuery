"""
Tests for the Semantic EO Query Layer.

Verifies:
  1. Concept detection from natural language queries
  2. Threshold consistency (single authoritative source)
  3. Spatial/temporal intent extraction
  4. Unsupported concept handling
  5. Query explanation completeness
  6. Generalization (no hard-coded locations)
"""

import pytest
from app.services.semantic_concepts import (
    SEMANTIC_CONCEPTS,
    detect_semantic_concept,
    get_concept,
    get_concept_for_phenomenon,
    list_semantic_concepts,
    SignalRule,
)
from app.services.query_to_plan import build_analysis_plan
from app.services.change_detection import PHENOMENON_CONFIG


# ══════════════════════════════════════════════════════════════════
# 1. Concept Detection from Natural Language
# ══════════════════════════════════════════════════════════════════

class TestConceptDetection:
    """Verify keyword-based concept detection works for real queries."""

    def test_urban_expansion_delhi(self):
        """'Delhi urban sprawl 2019 vs 2025' → URBAN_EXPANSION."""
        concept = detect_semantic_concept("Delhi urban sprawl 2019 vs 2025")
        assert concept == "URBAN_EXPANSION"

    def test_urban_expansion_hyderabad(self):
        """'Hyderabad urban expansion 2021 vs 2025' → URBAN_EXPANSION."""
        concept = detect_semantic_concept("Hyderabad urban expansion 2021 vs 2025")
        assert concept == "URBAN_EXPANSION"

    def test_forest_change(self):
        """'Amazon deforestation 2020 vs 2024' → VEGETATION_CHANGE."""
        concept = detect_semantic_concept("Amazon deforestation 2020 vs 2024")
        assert concept == "VEGETATION_CHANGE"

    def test_water_change(self):
        """'Lake shrinking 2018 vs 2023' → WATER_CHANGE."""
        concept = detect_semantic_concept("Lake shrinking 2018 vs 2023")
        assert concept == "WATER_CHANGE"

    def test_burn_severity(self):
        """'Wildfire burn severity California 2023 vs 2024' → BURN_CHANGE."""
        concept = detect_semantic_concept("Wildfire burn severity California 2023 vs 2024")
        assert concept == "BURN_CHANGE"

    def test_snow_change(self):
        """'Himalayan glacier retreat 2018 vs 2025' → SNOW_CHANGE."""
        concept = detect_semantic_concept("Himalayan glacier retreat 2018 vs 2025")
        assert concept == "SNOW_CHANGE"

    def test_ambiguous_query_returns_none(self):
        """Ambiguous query with no matching keywords → None."""
        concept = detect_semantic_concept("What happened here?")
        assert concept is None

    def test_unsupported_query_returns_none(self):
        """Query about an unsupported phenomenon → None."""
        concept = detect_semantic_concept("air quality monitoring in Mumbai")
        assert concept is None

    def test_concept_generalizes_beyond_specific_locations(self):
        """Concept detection must work for ANY location, not just Delhi/Hyderabad."""
        # Test with various locations
        for location in ["Jaipur", "Mumbai", "Chennai", "Kolkata", "Berlin", "Tokyo"]:
            concept = detect_semantic_concept(f"{location} urban expansion 2020 vs 2025")
            assert concept == "URBAN_EXPANSION", f"Failed for {location}"

    def test_concept_scores_longest_keyword_match(self):
        """When multiple keywords match, longest match wins."""
        # 'deforestation' (13 chars) > 'vegetation' (10 chars)
        concept = detect_semantic_concept("deforestation in forest area")
        assert concept == "VEGETATION_CHANGE"


# ══════════════════════════════════════════════════════════════════
# 2. Threshold Consistency
# ══════════════════════════════════════════════════════════════════

class TestThresholdConsistency:
    """
    There must be ONE authoritative threshold source.
    semantic_concepts.py → signal_rules → plan → runtime.
    """

    def test_ndbi_threshold_single_source(self):
        """NDBI threshold in semantic_concepts must match PHENOMENON_CONFIG."""
        concept = SEMANTIC_CONCEPTS["URBAN_EXPANSION"]
        ndbi_rule = [r for r in concept.signal_rules if r.index_name == "NDBI"][0]
        pheno_config = PHENOMENON_CONFIG.get("urban_expansion", {})
        assert ndbi_rule.threshold == pheno_config.get("threshold"), (
            f"NDBI threshold mismatch: semantic_concepts={ndbi_rule.threshold}, "
            f"PHENOMENON_CONFIG={pheno_config.get('threshold')}"
        )

    def test_ndvi_threshold_single_source(self):
        """NDVI threshold in semantic_concepts must match PHENOMENON_CONFIG."""
        concept = SEMANTIC_CONCEPTS["URBAN_EXPANSION"]
        ndvi_rule = [r for r in concept.signal_rules if r.index_name == "NDVI"][0]
        pheno_config = PHENOMENON_CONFIG.get("urban_expansion", {})
        assert ndvi_rule.threshold == pheno_config.get("ndvi_decrease_threshold"), (
            f"NDVI threshold mismatch: semantic_concepts={ndvi_rule.threshold}, "
            f"PHENOMENON_CONFIG={pheno_config.get('ndvi_decrease_threshold')}"
        )

    def test_plan_carries_thresholds_from_signal_rules(self):
        """Analysis plan must carry thresholds from signal rules."""
        result = build_analysis_plan("Delhi urban sprawl 2019 vs 2025")
        assert result["status"] == "ok"
        plan = result["plan"]
        signal_rules = plan["multi_signal"]["rules"]
        assert len(signal_rules) == 2

        # NDBI rule
        ndbi_rule = [r for r in signal_rules if r["index_name"] == "NDBI"][0]
        assert ndbi_rule["threshold"] == 0.12
        assert ndbi_rule["direction"] == "increase"

        # NDVI rule
        ndvi_rule = [r for r in signal_rules if r["index_name"] == "NDVI"][0]
        assert ndvi_rule["threshold"] == 0.08
        assert ndvi_rule["direction"] == "decrease"

    def test_all_concepts_have_thresholds(self):
        """Every signal rule must have a threshold."""
        for concept_id, concept in SEMANTIC_CONCEPTS.items():
            for rule in concept.signal_rules:
                assert rule.threshold is not None and rule.threshold > 0, (
                    f"{concept_id}/{rule.index_name} has invalid threshold: {rule.threshold}"
                )

    def test_no_duplicate_threshold_sources(self):
        """No two different threshold values for the same signal in the same concept."""
        for concept_id, concept in SEMANTIC_CONCEPTS.items():
            seen = {}
            for rule in concept.signal_rules:
                key = (rule.index_name, rule.direction)
                if key in seen:
                    assert seen[key] == rule.threshold, (
                        f"{concept_id}: conflicting thresholds for {key}: "
                        f"{seen[key]} vs {rule.threshold}"
                    )
                seen[key] = rule.threshold


# ══════════════════════════════════════════════════════════════════
# 3. Spatial/Temporal Intent
# ══════════════════════════════════════════════════════════════════

class TestSpatialTemporalIntent:
    """Verify the plan carries spatial and temporal intent."""

    def test_plan_has_temporal_range(self):
        """Plan must have start_date and end_date."""
        result = build_analysis_plan("Delhi urban sprawl 2019 vs 2025")
        plan = result["plan"]
        assert plan["start_date"] is not None
        assert plan["end_date"] is not None

    def test_plan_has_spatial_bbox(self):
        """Plan must have bbox for the AOI."""
        result = build_analysis_plan("Delhi urban sprawl 2019 vs 2025")
        plan = result["plan"]
        assert plan["bbox"] is not None
        assert len(plan["bbox"]) == 4

    def test_plan_has_indicators(self):
        """Plan must specify primary and all indicators."""
        result = build_analysis_plan("Delhi urban sprawl 2019 vs 2025")
        plan = result["plan"]
        assert plan["indicators"]["primary"] is not None
        assert len(plan["indicators"]["all"]) > 0

    def test_plan_has_multi_signal_config(self):
        """Plan must carry multi-signal configuration."""
        result = build_analysis_plan("Delhi urban sprawl 2019 vs 2025")
        plan = result["plan"]
        ms = plan["multi_signal"]
        assert "enabled" in ms
        assert "rules" in ms
        assert "min_agreeing_signals" in ms

    def test_plan_has_semantic_layer(self):
        """Plan must have semantic concept info."""
        result = build_analysis_plan("Delhi urban sprawl 2019 vs 2025")
        plan = result["plan"]
        assert plan["semantic"]["concept"] == "URBAN_EXPANSION"

    def test_plan_has_trace(self):
        """Plan must have a trace explaining the reasoning chain."""
        result = build_analysis_plan("Delhi urban sprawl 2019 vs 2025")
        plan = result["plan"]
        assert len(plan["trace"]) >= 3
        steps = [t["step"] for t in plan["trace"]]
        assert "user_query" in steps
        assert "semantic_concept" in steps


# ══════════════════════════════════════════════════════════════════
# 4. Unsupported Concepts
# ══════════════════════════════════════════════════════════════════

class TestUnsupportedConcepts:
    """Verify unsupported queries return clear errors."""

    def test_unsupported_phenomenon(self):
        """Unknown phenomenon returns unsupported status."""
        result = build_analysis_plan("air quality in Delhi")
        assert result["status"] == "unsupported"
        assert "suggestions" in result

    def test_unsupported_includes_suggestions(self):
        """Error message must include available concepts."""
        result = build_analysis_plan("air quality monitoring in Mumbai")
        assert result["status"] == "unsupported"
        suggestions = result.get("suggestions", [])
        assert len(suggestions) > 0

    def test_missing_aoi_returns_error(self):
        """Query without recognizable location returns error."""
        result = build_analysis_plan("urban expansion 2020 vs 2025")
        assert result["status"] == "error"

    def test_known_concepts_are_all_registered(self):
        """Every concept in SEMANTIC_CONCEPTS must have a registry_phenomenon."""
        for concept_id, concept in SEMANTIC_CONCEPTS.items():
            assert concept.registry_phenomenon, f"{concept_id} has no registry_phenomenon"

    def test_all_concepts_have_evidence_requirements(self):
        """Every concept must define evidence requirements."""
        for concept_id, concept in SEMANTIC_CONCEPTS.items():
            assert len(concept.evidence_requirements) > 0, (
                f"{concept_id} has no evidence requirements"
            )

    def test_all_concepts_have_keywords(self):
        """Every concept must have detection keywords."""
        for concept_id, concept in SEMANTIC_CONCEPTS.items():
            assert len(concept.keywords) > 0, f"{concept_id} has no keywords"

    def test_all_concepts_have_preferred_sensors(self):
        """Every concept must specify preferred sensors."""
        for concept_id, concept in SEMANTIC_CONCEPTS.items():
            assert len(concept.preferred_sensors) > 0, f"{concept_id} has no sensors"


# ══════════════════════════════════════════════════════════════════
# 5. Query Explanation
# ══════════════════════════════════════════════════════════════════

class TestQueryExplanation:
    """Verify the plan provides a complete explanation."""

    def test_plan_has_all_explanation_fields(self):
        """Plan must explain phenomenon, indicators, data, time, method."""
        result = build_analysis_plan("Delhi urban sprawl 2019 vs 2025")
        plan = result["plan"]

        # Phenomenon
        assert plan["phenomenon"] is not None
        assert plan["semantic"]["concept"] is not None

        # Indicators
        assert plan["indicators"]["primary"] is not None
        assert len(plan["indicators"]["all"]) > 0
        assert len(plan["indicators"]["formulas"]) > 0

        # Data requirements
        assert plan["sensor"] is not None
        assert plan["bands"] is not None
        assert plan["cloud_threshold"] is not None

        # Time
        assert plan["start_date"] is not None
        assert plan["end_date"] is not None

        # Method
        assert plan["comparison_strategy"] is not None
        assert plan["analysis_type"] is not None

    def test_plan_evidence_requirements_describe_why(self):
        """Evidence requirements must explain why indicators were selected."""
        result = build_analysis_plan("Delhi urban sprawl 2019 vs 2025")
        plan = result["plan"]
        evidence = plan["evidence_requirements"]
        assert len(evidence) > 0
        for ev in evidence:
            assert "description" in ev
            assert "interpretation" in ev
            assert len(ev["interpretation"]) > 0

    def test_plan_trace_documents_reasoning(self):
        """Trace must document the reasoning chain from query to plan."""
        result = build_analysis_plan("Delhi urban sprawl 2019 vs 2025")
        plan = result["plan"]
        trace = plan["trace"]
        # Must have at least: user_query, semantic_concept, data_requirements, indicators, analysis
        step_names = [t["step"] for t in trace]
        assert "user_query" in step_names
        assert "semantic_concept" in step_names
        assert "indicators" in step_names


# ══════════════════════════════════════════════════════════════════
# 6. Concept → Detector Mapping
# ══════════════════════════════════════════════════════════════════

class TestConceptDetectorMapping:
    """Verify concepts map correctly to detector configuration."""

    def test_urban_expansion_multi_signal(self):
        """URBAN_EXPANSION must use multi-signal (NDBI + NDVI)."""
        concept = SEMANTIC_CONCEPTS["URBAN_EXPANSION"]
        assert concept.multi_signal_recommended is True
        assert concept.min_agreeing_signals == 2
        assert len(concept.signal_rules) == 2

    def test_vegetation_change_single_signal(self):
        """VEGETATION_CHANGE uses single signal (NDVI)."""
        concept = SEMANTIC_CONCEPTS["VEGETATION_CHANGE"]
        assert concept.multi_signal_recommended is False
        assert concept.min_agreeing_signals == 1
        assert len(concept.signal_rules) == 1

    def test_water_change_single_signal(self):
        """WATER_CHANGE uses single signal (NDWI)."""
        concept = SEMANTIC_CONCEPTS["WATER_CHANGE"]
        assert concept.multi_signal_recommended is False
        assert concept.min_agreeing_signals == 1

    def test_burn_change_single_signal(self):
        """BURN_CHANGE uses single signal (NBR)."""
        concept = SEMANTIC_CONCEPTS["BURN_CHANGE"]
        assert concept.multi_signal_recommended is False
        assert concept.min_agreeing_signals == 1

    def test_concepts_have_correct_primary_indicators(self):
        """Each concept must have the correct primary indicator."""
        expected = {
            "URBAN_EXPANSION": "NDBI",
            "VEGETATION_CHANGE": "NDVI",
            "WATER_CHANGE": "NDWI",
            "BURN_CHANGE": "NBR",
            "SNOW_CHANGE": "NDSI",
        }
        for concept_id, expected_primary in expected.items():
            concept = SEMANTIC_CONCEPTS[concept_id]
            assert concept.primary_indicator == expected_primary, (
                f"{concept_id}: expected primary={expected_primary}, got {concept.primary_indicator}"
            )

    def test_min_agreeing_enforces_AND_logic(self):
        """URBAN_EXPANSION min_agreeing=2 means AND logic (both signals must agree)."""
        concept = SEMANTIC_CONCEPTS["URBAN_EXPANSION"]
        assert concept.min_agreeing_signals == 2
        # Both NDBI and NDVI rules must exist
        index_names = {r.index_name for r in concept.signal_rules}
        assert "NDBI" in index_names
        assert "NDVI" in index_names
