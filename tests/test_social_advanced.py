"""
Comprehensive Tests for Advanced Social Dynamics (KINSHIP-S Workstream)

Tests for:
- Social Graph
- Reputation System
- Coalition Formation
- Dominance Hierarchy
- Cultural Transmission
- Conflict Resolution
- Kinship Tracking
- Cooperation/Game Theory
"""

import pytest
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Social Graph Tests
# =============================================================================

class TestSocialGraph:
    """Tests for social_graph.py"""

    def test_add_organism(self):
        from social.social_graph import SocialGraph

        graph = SocialGraph(emit_events=False)
        node = graph.add_organism("org_1", attributes={"age": 5})

        assert graph.has_organism("org_1")
        assert node.organism_id == "org_1"
        assert node.attributes["age"] == 5

    def test_add_edge(self):
        from social.social_graph import SocialGraph, EdgeType

        graph = SocialGraph(emit_events=False)
        graph.add_organism("a")
        graph.add_organism("b")
        edge = graph.add_edge("a", "b", weight=0.8, edge_type=EdgeType.ALLIANCE)

        assert graph.has_edge("a", "b")
        assert not graph.has_edge("b", "a")  # Directed
        assert edge.weight == 0.8

    def test_bidirectional_edge(self):
        from social.social_graph import SocialGraph

        graph = SocialGraph(emit_events=False)
        graph.add_edge("a", "b", weight=0.5, bidirectional=True)

        assert graph.has_edge("a", "b")
        assert graph.has_edge("b", "a")

    def test_neighbors(self):
        from social.social_graph import SocialGraph

        graph = SocialGraph(emit_events=False)
        graph.add_edge("a", "b", bidirectional=True)
        graph.add_edge("a", "c")

        neighbors = graph.get_neighbors("a", direction="all")
        assert "b" in neighbors
        assert "c" in neighbors

    def test_shortest_path(self):
        from social.social_graph import SocialGraph

        graph = SocialGraph(emit_events=False)
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", "d")

        path = graph.find_shortest_path("a", "d")
        assert path == ["a", "b", "c", "d"]

    def test_community_detection(self):
        from social.social_graph import SocialGraph

        graph = SocialGraph(emit_events=False)
        # Create two cliques
        for i in range(3):
            for j in range(3):
                if i != j:
                    graph.add_edge(f"a{i}", f"a{j}")
                    graph.add_edge(f"b{i}", f"b{j}")

        communities = graph.detect_communities()
        assert len(communities) >= 2

    def test_centrality(self):
        from social.social_graph import SocialGraph

        graph = SocialGraph(emit_events=False)
        graph.add_edge("center", "a")
        graph.add_edge("center", "b")
        graph.add_edge("center", "c")

        metrics = graph.calculate_centrality("center")
        assert metrics.out_degree == 3
        assert metrics.degree_centrality > 0


# =============================================================================
# Reputation Tests
# =============================================================================

class TestReputation:
    """Tests for reputation.py"""

    def test_record_observation(self):
        from social.reputation import ReputationEngine, ReputationDimension

        engine = ReputationEngine(emit_events=False)
        obs = engine.record_observation(
            "observer", "target",
            ReputationDimension.TRUSTWORTHY, 0.8
        )

        assert obs.value == 0.8
        assert obs.dimension == ReputationDimension.TRUSTWORTHY

    def test_get_reputation(self):
        from social.reputation import ReputationEngine, ReputationDimension

        engine = ReputationEngine(emit_events=False)
        engine.record_observation("a", "target", ReputationDimension.TRUSTWORTHY, 0.9)
        engine.record_observation("b", "target", ReputationDimension.TRUSTWORTHY, 0.7)

        score = engine.get_reputation("target")
        assert 0.7 <= score.get_dimension(ReputationDimension.TRUSTWORTHY) <= 0.9

    def test_observer_specific_reputation(self):
        from social.reputation import ReputationEngine, ReputationDimension

        engine = ReputationEngine(emit_events=False)
        engine.record_observation("alice", "target", ReputationDimension.COMPETENT, 0.9)
        engine.record_observation("bob", "target", ReputationDimension.COMPETENT, 0.3)

        alice_view = engine.get_reputation("target", observer_id="alice")
        bob_view = engine.get_reputation("target", observer_id="bob")

        assert alice_view.get_dimension(ReputationDimension.COMPETENT) == 0.9
        assert bob_view.get_dimension(ReputationDimension.COMPETENT) == 0.3

    def test_overall_score(self):
        from social.reputation import ReputationEngine, ReputationDimension

        engine = ReputationEngine(emit_events=False)
        engine.record_observation("a", "target", ReputationDimension.TRUSTWORTHY, 0.8)
        engine.record_observation("a", "target", ReputationDimension.COOPERATIVE, 0.6)

        score = engine.get_reputation("target")
        assert score.overall_score != 0


# =============================================================================
# Coalition Tests
# =============================================================================

class TestCoalition:
    """Tests for coalition.py"""

    def test_form_coalition(self):
        from social.coalition import CoalitionManager, CoalitionType

        manager = CoalitionManager(emit_events=False)
        cid = manager.form_coalition(
            "founder",
            initial_members=["a", "b"],
            name="TestCoalition",
            purpose="testing"
        )

        coalition = manager.get_coalition(cid)
        assert coalition is not None
        assert coalition.size == 3  # founder + 2 members
        assert "founder" in coalition.members

    def test_request_membership(self):
        from social.coalition import CoalitionManager

        manager = CoalitionManager(emit_events=False)
        cid = manager.form_coalition("founder", ["a"])

        proposal_id = manager.request_membership(cid, "newcomer")
        assert proposal_id is not None

        manager.accept_membership(proposal_id)
        coalition = manager.get_coalition(cid)
        assert "newcomer" in coalition.members

    def test_coalition_stability(self):
        from social.coalition import CoalitionManager

        manager = CoalitionManager(emit_events=False)
        cid = manager.form_coalition("a", ["b", "c", "d"])

        stability = manager.calculate_stability(cid)
        assert 0 <= stability <= 1

    def test_merge_coalitions(self):
        from social.coalition import CoalitionManager

        manager = CoalitionManager(emit_events=False)
        cid1 = manager.form_coalition("a", ["b"])
        cid2 = manager.form_coalition("c", ["d"])

        merged_id = manager.merge_coalitions(cid1, cid2, "Merged")
        merged = manager.get_coalition(merged_id)

        assert merged.size == 4
        assert "a" in merged.members and "c" in merged.members


# =============================================================================
# Hierarchy Tests
# =============================================================================

class TestHierarchy:
    """Tests for hierarchy.py"""

    def test_add_organism(self):
        from social.hierarchy import HierarchyManager

        manager = HierarchyManager(emit_events=False)
        score = manager.add_organism("alpha", initial_rating=1200)

        assert score.rating == 1200

    def test_record_contest(self):
        from social.hierarchy import HierarchyManager

        manager = HierarchyManager(emit_events=False)
        manager.add_organism("a", 1000)
        manager.add_organism("b", 1000)

        contest = manager.record_contest("a", "b", winner="a")

        assert contest.winner_id == "a"
        assert manager.get_rating("a") > 1000
        assert manager.get_rating("b") < 1000

    def test_rankings(self):
        from social.hierarchy import HierarchyManager

        manager = HierarchyManager(emit_events=False)
        manager.add_organism("alpha", 1500)
        manager.add_organism("beta", 1200)
        manager.add_organism("gamma", 1000)

        rankings = manager.get_rankings()
        assert rankings[0][0] == "alpha"
        assert rankings[1][0] == "beta"
        assert rankings[2][0] == "gamma"

    def test_dominance_probability(self):
        from social.hierarchy import HierarchyManager

        manager = HierarchyManager(emit_events=False)
        manager.add_organism("strong", 1400)
        manager.add_organism("weak", 1000)

        prob = manager.dominance_probability("strong", "weak")
        assert prob > 0.5

    def test_linearity(self):
        from social.hierarchy import HierarchyManager

        manager = HierarchyManager(emit_events=False)
        manager.add_organism("a")
        manager.add_organism("b")
        manager.record_contest("a", "b", winner="a")

        linearity = manager.calculate_linearity()
        assert 0 <= linearity <= 1


# =============================================================================
# Cultural Transmission Tests
# =============================================================================

class TestCulturalTransmission:
    """Tests for cultural_transmission.py"""

    def test_create_meme(self):
        from social.cultural_transmission import CulturalEngine, MemeCategory

        engine = CulturalEngine(emit_events=False)
        meme = engine.create_meme(
            "inventor", "fire_making",
            category=MemeCategory.TECHNOLOGY,
            fitness=0.8
        )

        assert meme.content == "fire_making"
        assert meme.fitness == 0.8

    def test_transmit(self):
        from social.cultural_transmission import CulturalEngine, TransmissionMode

        engine = CulturalEngine(emit_events=False)
        meme = engine.create_meme("teacher", "knowledge")

        event = engine.transmit(
            meme.id, "teacher", "student",
            TransmissionMode.HORIZONTAL, force=True
        )

        assert event.success
        knowledge = engine.get_organism_knowledge("student")
        assert knowledge.knows_meme(meme.id)

    def test_vertical_transmission(self):
        from social.cultural_transmission import CulturalEngine

        engine = CulturalEngine(emit_events=False)
        meme = engine.create_meme("parent", "tradition")

        events = engine.vertical_transmission("parent", "child")
        assert len(events) > 0

    def test_meme_carriers(self):
        from social.cultural_transmission import CulturalEngine

        engine = CulturalEngine(emit_events=False)
        meme = engine.create_meme("creator", "idea")

        engine.transmit(meme.id, "creator", "a", force=True)
        engine.transmit(meme.id, "creator", "b", force=True)

        carriers = engine.get_meme_carriers(meme.id)
        assert len(carriers) == 3  # creator, a, b


# =============================================================================
# Conflict Resolution Tests
# =============================================================================

class TestConflictResolution:
    """Tests for conflict_resolution.py"""

    def test_register_conflict(self):
        from social.conflict_resolution import ConflictResolver, ConflictType

        resolver = ConflictResolver(emit_events=False)
        conflict = resolver.register_conflict(
            "a", "b",
            ConflictType.RESOURCE,
            resource_value=100
        )

        assert conflict.party1_id == "a"
        assert conflict.party2_id == "b"
        assert conflict.status == "active"

    def test_resolve_fight(self):
        from social.conflict_resolution import ConflictResolver, ConflictType, Strategy

        resolver = ConflictResolver(emit_events=False)
        resolver.register_organism("a", fighting_ability=0.9)
        resolver.register_organism("b", fighting_ability=0.3)

        conflict = resolver.register_conflict("a", "b", ConflictType.RESOURCE)
        outcome = resolver.resolve(conflict.id, Strategy.FIGHT)

        assert outcome is not None
        assert outcome.winner_id is not None

    def test_resolve_negotiate(self):
        from social.conflict_resolution import ConflictResolver, ConflictType, Strategy

        resolver = ConflictResolver(emit_events=False)
        conflict = resolver.register_conflict("a", "b", ConflictType.RESOURCE, resource_value=100)
        outcome = resolver.resolve(conflict.id, Strategy.SHARE)

        assert outcome.party1_gain == 50
        assert outcome.party2_gain == 50

    def test_recommend_strategy(self):
        from social.conflict_resolution import ConflictResolver, ConflictType

        resolver = ConflictResolver(emit_events=False)
        resolver.register_organism("strong", fighting_ability=0.9)
        resolver.register_organism("weak", fighting_ability=0.2)

        conflict = resolver.register_conflict("strong", "weak", ConflictType.DOMINANCE)
        strategy = resolver.recommend_strategy(conflict.id, "strong")

        assert strategy is not None


# =============================================================================
# Kinship Tests
# =============================================================================

class TestKinship:
    """Tests for kinship.py"""

    def test_register_birth(self):
        from social.kinship import KinshipTracker

        tracker = KinshipTracker(emit_events=False)
        tracker.register_organism("parent1")
        tracker.register_organism("parent2")

        child = tracker.register_birth("child", "parent1", "parent2")

        assert child.parent1_id == "parent1"
        assert child.parent2_id == "parent2"
        assert child.generation == 1

    def test_relatedness_parent_child(self):
        from social.kinship import KinshipTracker

        tracker = KinshipTracker(emit_events=False)
        tracker.register_birth("child", "parent1", "parent2")

        r = tracker.calculate_relatedness("child", "parent1")
        assert abs(r - 0.5) < 0.01  # Parent-child r = 0.5

    def test_relatedness_siblings(self):
        from social.kinship import KinshipTracker

        tracker = KinshipTracker(emit_events=False)
        tracker.register_organism("p1")
        tracker.register_organism("p2")
        tracker.register_birth("sib1", "p1", "p2")
        tracker.register_birth("sib2", "p1", "p2")

        r = tracker.calculate_relatedness("sib1", "sib2")
        assert abs(r - 0.5) < 0.01  # Full siblings r = 0.5

    def test_relatedness_half_siblings(self):
        from social.kinship import KinshipTracker

        tracker = KinshipTracker(emit_events=False)
        tracker.register_organism("p1")
        tracker.register_organism("p2")
        tracker.register_organism("p3")
        tracker.register_birth("half1", "p1", "p2")
        tracker.register_birth("half2", "p1", "p3")

        r = tracker.calculate_relatedness("half1", "half2")
        assert abs(r - 0.25) < 0.01  # Half siblings r = 0.25

    def test_incest_check(self):
        from social.kinship import KinshipTracker

        tracker = KinshipTracker(emit_events=False)
        tracker.register_birth("child", "p1", "p2")

        is_incest = tracker.is_incestuous("child", "p1")
        assert is_incest  # Parent-child mating is incestuous

    def test_get_kin(self):
        from social.kinship import KinshipTracker

        tracker = KinshipTracker(emit_events=False)
        tracker.register_birth("child", "p1", "p2")

        kin = tracker.get_kin("child")
        kin_ids = {k.organism2_id for k in kin}
        assert "p1" in kin_ids
        assert "p2" in kin_ids


# =============================================================================
# Cooperation Tests
# =============================================================================

class TestCooperation:
    """Tests for cooperation.py"""

    def test_register_organism(self):
        from social.cooperation import CooperationEngine, Strategy

        engine = CooperationEngine(emit_events=False)
        profile = engine.register_organism("player", Strategy.TIT_FOR_TAT)

        assert profile.strategy == Strategy.TIT_FOR_TAT

    def test_play_round(self):
        from social.cooperation import CooperationEngine, Strategy, Game

        engine = CooperationEngine(emit_events=False)
        engine.register_organism("p1", Strategy.ALWAYS_COOPERATE)
        engine.register_organism("p2", Strategy.ALWAYS_COOPERATE)

        result = engine.play_round("p1", "p2", Game.PRISONERS_DILEMMA)

        assert result.payoff1 == 3.0  # Mutual cooperation
        assert result.payoff2 == 3.0

    def test_tit_for_tat(self):
        from social.cooperation import CooperationEngine, Strategy, Action

        engine = CooperationEngine(emit_events=False)
        engine.register_organism("tft", Strategy.TIT_FOR_TAT)
        engine.register_organism("defector", Strategy.ALWAYS_DEFECT)

        # Round 1: TFT cooperates, defector defects
        r1 = engine.play_round("tft", "defector")
        assert r1.action1 == Action.COOPERATE
        assert r1.action2 == Action.DEFECT

        # Round 2: TFT defects (mirrors previous), defector defects
        r2 = engine.play_round("tft", "defector")
        assert r2.action1 == Action.DEFECT

    def test_public_goods_game(self):
        from social.cooperation import CooperationEngine, Strategy

        engine = CooperationEngine(emit_events=False)
        engine.register_organism("p1", Strategy.ALWAYS_COOPERATE)
        engine.register_organism("p2", Strategy.ALWAYS_COOPERATE)
        engine.register_organism("p3", Strategy.ALWAYS_DEFECT)

        payoffs = engine.play_public_goods_game(["p1", "p2", "p3"])

        # Defector (p3) should get more than cooperators
        assert payoffs["p3"] > payoffs["p1"]

    def test_altruism_hamilton_rule(self):
        from social.cooperation import CooperationEngine

        engine = CooperationEngine(emit_events=False)

        # r*B > C should return True
        result = engine.calculate_altruism_threshold(
            "actor", "recipient",
            relatedness=0.5, cost=1.0, benefit=5.0
        )
        assert result  # 0.5 * 5 = 2.5 > 1

        # r*B < C should return False
        result = engine.calculate_altruism_threshold(
            "actor", "recipient",
            relatedness=0.1, cost=1.0, benefit=5.0
        )
        assert not result  # 0.1 * 5 = 0.5 < 1

    def test_strategy_distribution(self):
        from social.cooperation import CooperationEngine, Strategy

        engine = CooperationEngine(emit_events=False)
        engine.register_organism("p1", Strategy.TIT_FOR_TAT)
        engine.register_organism("p2", Strategy.TIT_FOR_TAT)
        engine.register_organism("p3", Strategy.ALWAYS_DEFECT)

        dist = engine.get_strategy_distribution()
        assert dist["tit_for_tat"] == 2
        assert dist["always_defect"] == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple social systems"""

    def test_kinship_with_cooperation(self):
        """Test that relatedness affects altruism decisions"""
        from social.kinship import KinshipTracker
        from social.cooperation import CooperationEngine

        kinship = KinshipTracker(emit_events=False)
        coop = CooperationEngine(emit_events=False)

        # Create family
        kinship.register_birth("child", "p1", "p2")

        # Get relatedness
        r = kinship.calculate_relatedness("child", "p1")

        # Check altruism threshold
        should_help = coop.calculate_altruism_threshold(
            "child", "p1",
            relatedness=r, cost=1.0, benefit=3.0
        )
        assert should_help  # r=0.5, 0.5*3=1.5 > 1

    def test_hierarchy_with_conflict(self):
        """Test that hierarchy influences conflict outcomes"""
        from social.hierarchy import HierarchyManager
        from social.conflict_resolution import ConflictResolver, ConflictType, Strategy

        hierarchy = HierarchyManager(emit_events=False)
        conflict_res = ConflictResolver(emit_events=False)

        # Establish hierarchy
        hierarchy.add_organism("alpha", 1400)
        hierarchy.add_organism("beta", 1000)

        # Register in conflict system with abilities matching hierarchy
        conflict_res.register_organism("alpha", fighting_ability=0.8)
        conflict_res.register_organism("beta", fighting_ability=0.4)

        # Create conflict
        c = conflict_res.register_conflict("alpha", "beta", ConflictType.DOMINANCE)

        # Higher ranked should win more often via display
        strategy = conflict_res.recommend_strategy(c.id, "alpha")
        assert strategy in (Strategy.DISPLAY, Strategy.FIGHT, Strategy.NEGOTIATE)


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    """Tests for serialization/deserialization"""

    def test_social_graph_serialization(self):
        from social.social_graph import SocialGraph

        graph = SocialGraph(emit_events=False)
        graph.add_edge("a", "b", weight=0.7)
        graph.add_edge("b", "c", weight=0.5)

        data = graph.to_dict()
        restored = SocialGraph.from_dict(data)

        assert restored.has_edge("a", "b")
        assert restored.get_edge("a", "b").weight == 0.7

    def test_kinship_serialization(self):
        from social.kinship import KinshipTracker

        tracker = KinshipTracker(emit_events=False)
        tracker.register_birth("child", "p1", "p2")

        data = tracker.to_dict()
        restored = KinshipTracker.from_dict(data)

        r = restored.calculate_relatedness("child", "p1")
        assert abs(r - 0.5) < 0.01

    def test_cooperation_serialization(self):
        from social.cooperation import CooperationEngine, Strategy

        engine = CooperationEngine(emit_events=False)
        engine.register_organism("p1", Strategy.GRIM_TRIGGER)

        data = engine.to_dict()
        restored = CooperationEngine.from_dict(data)

        profile = restored.get_profile("p1")
        assert profile.strategy == Strategy.GRIM_TRIGGER


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
