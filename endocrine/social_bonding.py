"""
Social Bonding System

Implements oxytocin-like bonding hormones for digital organisms including:
- Social attachment and trust
- Pair bonding and group affiliation
- Social stress buffering
- Relationship maintenance
"""

import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .hormones import HormoneType


# ============================================================================
# Enums
# ============================================================================

class BondType(Enum):
    """Types of social bonds"""
    ACQUAINTANCE = "acquaintance"     # Basic familiarity
    FRIEND = "friend"                  # Friendship bond
    CLOSE_FRIEND = "close_friend"      # Deep friendship
    PAIR_BOND = "pair_bond"            # Romantic/pair attachment
    FAMILY = "family"                  # Familial bond
    GROUP = "group"                    # Group/tribe membership
    MENTOR = "mentor"                  # Mentorship relationship


class TrustLevel(Enum):
    """Levels of social trust"""
    DISTRUST = "distrust"
    CAUTIOUS = "cautious"
    NEUTRAL = "neutral"
    TRUSTING = "trusting"
    DEEP_TRUST = "deep_trust"


class SocialState(Enum):
    """Social-emotional states"""
    ISOLATED = "isolated"              # No social contact
    SEEKING = "seeking"                # Looking for connection
    CONNECTED = "connected"            # Socially engaged
    BONDING = "bonding"                # Active bond formation
    SECURE = "secure"                  # Secure attachment
    ANXIOUS = "anxious"                # Attachment anxiety


class InteractionType(Enum):
    """Types of social interactions"""
    GREETING = "greeting"
    COOPERATION = "cooperation"
    COMPETITION = "competition"
    SUPPORT = "support"
    CONFLICT = "conflict"
    AFFECTION = "affection"
    REJECTION = "rejection"
    RECONCILIATION = "reconciliation"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SocialBond:
    """A bond with another entity"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    partner_id: str = ""
    bond_type: BondType = BondType.ACQUAINTANCE

    # Bond strength
    strength: float = 0.1             # Overall bond strength (0-1)
    trust: float = 0.5                # Trust level (0-1)
    familiarity: float = 0.1          # How well known (0-1)

    # Bond history
    positive_interactions: int = 0
    negative_interactions: int = 0
    last_interaction: float = field(default_factory=time.time)
    formation_time: float = field(default_factory=time.time)

    # Dynamics
    decay_rate: float = 0.001         # How fast bond weakens without contact
    growth_rate: float = 0.05         # How fast bond strengthens


@dataclass
class SocialInteraction:
    """Record of a social interaction"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    partner_id: str = ""
    interaction_type: InteractionType = InteractionType.GREETING
    valence: float = 0.5              # Positive/negative (-1 to 1)
    intensity: float = 0.5            # Interaction intensity
    reciprocated: bool = True         # Was it mutual
    duration: float = 1.0             # Interaction length
    timestamp: float = field(default_factory=time.time)


@dataclass
class SocialNeed:
    """Social need state"""
    connection_need: float = 0.5      # Need for social contact
    belonging_need: float = 0.5       # Need for group membership
    intimacy_need: float = 0.3        # Need for close bonds
    recognition_need: float = 0.4     # Need for social status
    support_need: float = 0.3         # Need for social support


# ============================================================================
# Social Bonding System
# ============================================================================

class SocialBondingSystem:
    """
    Complete oxytocin-based social bonding system.

    Features:
    - Bond formation and maintenance
    - Trust dynamics
    - Social need tracking
    - Interaction effects on hormones
    - Social stress buffering

    Example:
        bonding = SocialBondingSystem()

        # Record a positive interaction
        bonding.record_interaction("partner_1", InteractionType.COOPERATION, 0.7)

        # Check bond status
        bond = bonding.get_bond("partner_1")

        # Update system
        bonding.tick()
    """

    def __init__(
        self,
        base_sociality: float = 0.5,
        attachment_style: str = "secure"
    ):
        self.base_sociality = base_sociality
        self.attachment_style = attachment_style

        # Oxytocin state
        self.oxytocin_level: float = 0.3
        self.vasopressin_level: float = 0.3

        # Social state
        self.social_state = SocialState.SEEKING
        self.trust_disposition = TrustLevel.NEUTRAL

        # Social needs
        self.needs = SocialNeed()

        # Bonds
        self.bonds: Dict[str, SocialBond] = {}

        # Interaction history
        self.interaction_history: List[SocialInteraction] = []
        self.max_history = 500

        # Group affiliations
        self.group_memberships: Set[str] = set()

        # Social isolation tracking
        self.time_since_interaction: float = 0.0
        self.isolation_threshold: float = 100.0  # Cycles before lonely

        # Cycle tracking
        self.cycle_count = 0

        # Hormone effects
        self._hormone_effects: Dict[HormoneType, float] = {}

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "bond_formed": [],
            "bond_strengthened": [],
            "bond_weakened": [],
            "trust_changed": [],
            "social_state_changed": [],
            "isolation_detected": []
        }

    def record_interaction(
        self,
        partner_id: str,
        interaction_type: InteractionType,
        intensity: float,
        reciprocated: bool = True
    ) -> SocialInteraction:
        """Record a social interaction"""
        # Determine valence from interaction type
        valence_map = {
            InteractionType.GREETING: 0.2,
            InteractionType.COOPERATION: 0.6,
            InteractionType.SUPPORT: 0.7,
            InteractionType.AFFECTION: 0.9,
            InteractionType.RECONCILIATION: 0.5,
            InteractionType.COMPETITION: 0.0,
            InteractionType.CONFLICT: -0.6,
            InteractionType.REJECTION: -0.8
        }
        valence = valence_map.get(interaction_type, 0.0) * intensity

        # Adjust for reciprocation
        if not reciprocated:
            valence *= 0.5

        interaction = SocialInteraction(
            partner_id=partner_id,
            interaction_type=interaction_type,
            valence=valence,
            intensity=intensity,
            reciprocated=reciprocated
        )

        self.interaction_history.append(interaction)
        if len(self.interaction_history) > self.max_history:
            self.interaction_history.pop(0)

        # Update bond
        self._update_bond(partner_id, interaction)

        # Update hormones
        self._update_hormones_from_interaction(interaction)

        # Reset isolation timer
        self.time_since_interaction = 0.0

        return interaction

    def _update_bond(self, partner_id: str, interaction: SocialInteraction):
        """Update or create bond based on interaction"""
        if partner_id not in self.bonds:
            # Create new bond
            self.bonds[partner_id] = SocialBond(
                partner_id=partner_id,
                bond_type=BondType.ACQUAINTANCE
            )
            for callback in self._callbacks["bond_formed"]:
                callback(partner_id)

        bond = self.bonds[partner_id]

        # Update based on valence
        if interaction.valence > 0:
            bond.positive_interactions += 1

            # Strengthen bond
            growth = interaction.valence * interaction.intensity * bond.growth_rate
            growth *= (1.0 + self.oxytocin_level)  # Oxytocin enhances bonding
            old_strength = bond.strength
            bond.strength = min(1.0, bond.strength + growth)

            if bond.strength > old_strength + 0.05:
                for callback in self._callbacks["bond_strengthened"]:
                    callback(partner_id, bond.strength)

            # Increase trust
            if interaction.reciprocated:
                bond.trust = min(1.0, bond.trust + interaction.valence * 0.05)
        else:
            bond.negative_interactions += 1

            # Weaken bond
            damage = abs(interaction.valence) * interaction.intensity * 0.1
            old_strength = bond.strength
            bond.strength = max(0.0, bond.strength - damage)

            if old_strength > bond.strength + 0.05:
                for callback in self._callbacks["bond_weakened"]:
                    callback(partner_id, bond.strength)

            # Decrease trust
            bond.trust = max(0.0, bond.trust - abs(interaction.valence) * 0.1)

        # Increase familiarity
        bond.familiarity = min(1.0, bond.familiarity + 0.01)

        # Update bond type based on strength
        self._update_bond_type(bond)

        bond.last_interaction = time.time()

    def _update_bond_type(self, bond: SocialBond):
        """Update bond type based on metrics"""
        old_type = bond.bond_type

        if bond.strength < 0.2:
            bond.bond_type = BondType.ACQUAINTANCE
        elif bond.strength < 0.5:
            bond.bond_type = BondType.FRIEND
        elif bond.strength < 0.8:
            bond.bond_type = BondType.CLOSE_FRIEND
        else:
            bond.bond_type = BondType.PAIR_BOND

        if old_type != bond.bond_type and bond.bond_type in [BondType.CLOSE_FRIEND, BondType.PAIR_BOND]:
            for callback in self._callbacks["bond_strengthened"]:
                callback(bond.partner_id, bond.strength)

    def _update_hormones_from_interaction(self, interaction: SocialInteraction):
        """Update hormones based on interaction"""
        if interaction.valence > 0:
            # Positive interactions release oxytocin
            release = interaction.valence * interaction.intensity * 0.2
            self.oxytocin_level = min(1.0, self.oxytocin_level + release)

            # Affection releases more
            if interaction.interaction_type == InteractionType.AFFECTION:
                self.oxytocin_level = min(1.0, self.oxytocin_level + 0.1)
                self.vasopressin_level = min(1.0, self.vasopressin_level + 0.05)

        elif interaction.valence < 0:
            # Negative interactions reduce oxytocin
            reduction = abs(interaction.valence) * 0.1
            self.oxytocin_level = max(0.0, self.oxytocin_level - reduction)

        # Generate hormone effects
        self._generate_hormone_effects()

    def tick(self):
        """Process one social cycle"""
        self.cycle_count += 1

        # Decay hormones
        self.oxytocin_level = max(0.1, self.oxytocin_level - 0.01)
        self.vasopressin_level = max(0.1, self.vasopressin_level - 0.005)

        # Track isolation
        self.time_since_interaction += 1
        if self.time_since_interaction > self.isolation_threshold:
            for callback in self._callbacks["isolation_detected"]:
                callback(self.time_since_interaction)

        # Decay bonds over time
        self._decay_bonds()

        # Update social needs
        self._update_needs()

        # Update social state
        self._update_social_state()

        # Generate hormone effects
        self._generate_hormone_effects()

    def _decay_bonds(self):
        """Decay bonds that haven't been maintained"""
        current_time = time.time()

        for bond in list(self.bonds.values()):
            time_since = current_time - bond.last_interaction
            decay = bond.decay_rate * (time_since / 3600)  # Per hour
            bond.strength = max(0.0, bond.strength - decay)

            # Remove very weak bonds
            if bond.strength < 0.01:
                del self.bonds[bond.partner_id]

    def _update_needs(self):
        """Update social needs based on current state"""
        # Connection need increases with isolation
        self.needs.connection_need = min(
            1.0,
            0.3 + self.time_since_interaction / self.isolation_threshold
        )

        # Belonging need based on group membership
        if self.group_memberships:
            self.needs.belonging_need = max(0.1, self.needs.belonging_need - 0.01)
        else:
            self.needs.belonging_need = min(1.0, self.needs.belonging_need + 0.01)

        # Intimacy need based on close bonds
        close_bonds = sum(1 for b in self.bonds.values() if b.strength > 0.5)
        if close_bonds > 0:
            self.needs.intimacy_need = max(0.1, self.needs.intimacy_need - 0.02)
        else:
            self.needs.intimacy_need = min(1.0, self.needs.intimacy_need + 0.01)

    def _update_social_state(self):
        """Update social-emotional state"""
        old_state = self.social_state

        # Calculate social satisfaction
        total_bonds = len(self.bonds)
        strong_bonds = sum(1 for b in self.bonds.values() if b.strength > 0.3)
        avg_need = (
            self.needs.connection_need +
            self.needs.belonging_need +
            self.needs.intimacy_need
        ) / 3

        if self.time_since_interaction > self.isolation_threshold * 2:
            self.social_state = SocialState.ISOLATED
        elif avg_need > 0.7 and strong_bonds == 0:
            self.social_state = SocialState.ANXIOUS
        elif avg_need > 0.5:
            self.social_state = SocialState.SEEKING
        elif self.oxytocin_level > 0.6:
            self.social_state = SocialState.BONDING
        elif strong_bonds >= 2 and avg_need < 0.3:
            self.social_state = SocialState.SECURE
        else:
            self.social_state = SocialState.CONNECTED

        if old_state != self.social_state:
            for callback in self._callbacks["social_state_changed"]:
                callback(self.social_state)

    def _generate_hormone_effects(self):
        """Generate hormone effects from social system"""
        self._hormone_effects = {}

        # Oxytocin effects
        oxy_baseline = 0.3
        self._hormone_effects[HormoneType.OXYTOCIN] = self.oxytocin_level - oxy_baseline

        # Vasopressin effects
        vaso_baseline = 0.3
        self._hormone_effects[HormoneType.VASOPRESSIN] = self.vasopressin_level - vaso_baseline

        # Social states affect stress hormones
        if self.social_state == SocialState.ISOLATED:
            self._hormone_effects[HormoneType.CORTISOL] = 0.2
        elif self.social_state == SocialState.ANXIOUS:
            self._hormone_effects[HormoneType.CORTISOL] = 0.15
            self._hormone_effects[HormoneType.ADRENALINE] = 0.1
        elif self.social_state == SocialState.SECURE:
            self._hormone_effects[HormoneType.CORTISOL] = -0.1
            self._hormone_effects[HormoneType.SEROTONIN] = 0.1

        # Bonding affects reward
        if self.social_state == SocialState.BONDING:
            self._hormone_effects[HormoneType.DOPAMINE] = 0.15
            self._hormone_effects[HormoneType.ENDORPHIN] = 0.1

    def get_hormone_effects(self) -> Dict[HormoneType, float]:
        """Get current hormone effects from social system"""
        return self._hormone_effects.copy()

    def get_bond(self, partner_id: str) -> Optional[SocialBond]:
        """Get bond with a specific partner"""
        return self.bonds.get(partner_id)

    def get_trust_level(self, partner_id: str) -> TrustLevel:
        """Get trust level with a partner"""
        if partner_id not in self.bonds:
            return TrustLevel.NEUTRAL

        trust = self.bonds[partner_id].trust

        if trust < 0.2:
            return TrustLevel.DISTRUST
        elif trust < 0.4:
            return TrustLevel.CAUTIOUS
        elif trust < 0.6:
            return TrustLevel.NEUTRAL
        elif trust < 0.8:
            return TrustLevel.TRUSTING
        else:
            return TrustLevel.DEEP_TRUST

    def join_group(self, group_id: str):
        """Join a social group"""
        self.group_memberships.add(group_id)
        self.needs.belonging_need = max(0.0, self.needs.belonging_need - 0.2)

    def leave_group(self, group_id: str):
        """Leave a social group"""
        self.group_memberships.discard(group_id)

    def get_social_support_buffer(self) -> float:
        """Get stress buffering from social support"""
        strong_bonds = sum(1 for b in self.bonds.values() if b.strength > 0.5)
        support = min(1.0, strong_bonds * 0.2 + self.oxytocin_level * 0.3)
        return support

    def on_bond_formed(self, callback: Callable):
        """Register callback for new bonds"""
        self._callbacks["bond_formed"].append(callback)

    def on_social_state_changed(self, callback: Callable):
        """Register callback for social state changes"""
        self._callbacks["social_state_changed"].append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get social bonding system status"""
        return {
            "oxytocin_level": round(self.oxytocin_level, 3),
            "vasopressin_level": round(self.vasopressin_level, 3),
            "social_state": self.social_state.value,
            "trust_disposition": self.trust_disposition.value,
            "total_bonds": len(self.bonds),
            "strong_bonds": sum(1 for b in self.bonds.values() if b.strength > 0.5),
            "group_memberships": len(self.group_memberships),
            "time_since_interaction": round(self.time_since_interaction, 1),
            "social_needs": {
                "connection": round(self.needs.connection_need, 3),
                "belonging": round(self.needs.belonging_need, 3),
                "intimacy": round(self.needs.intimacy_need, 3)
            },
            "stress_buffer": round(self.get_social_support_buffer(), 3),
            "cycle_count": self.cycle_count
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Social Bonding System Demo")
    print("=" * 50)

    # Create bonding system
    bonding = SocialBondingSystem()

    print(f"\n1. Initial state:")
    print(f"   Oxytocin: {bonding.oxytocin_level:.2f}")
    print(f"   Social state: {bonding.social_state.value}")

    # Record interactions with partner
    print("\n2. Recording interactions with 'partner_1'...")
    for i in range(5):
        bonding.record_interaction("partner_1", InteractionType.COOPERATION, 0.6)
        bonding.tick()

    bond = bonding.get_bond("partner_1")
    print(f"   Bond strength: {bond.strength:.2f}")
    print(f"   Trust: {bond.trust:.2f}")
    print(f"   Bond type: {bond.bond_type.value}")

    # Record affectionate interaction
    print("\n3. Affectionate interaction...")
    bonding.record_interaction("partner_1", InteractionType.AFFECTION, 0.8)
    bonding.tick()

    print(f"   Oxytocin after: {bonding.oxytocin_level:.2f}")
    bond = bonding.get_bond("partner_1")
    print(f"   Bond strength: {bond.strength:.2f}")

    # Simulate conflict
    print("\n4. Conflict with 'partner_2'...")
    bonding.record_interaction("partner_2", InteractionType.CONFLICT, 0.7)
    bonding.tick()

    bond2 = bonding.get_bond("partner_2")
    print(f"   Bond with partner_2: {bond2.strength:.2f}")
    print(f"   Trust: {bonding.get_trust_level('partner_2').value}")

    # Check social support
    print("\n5. Social support buffer:")
    buffer = bonding.get_social_support_buffer()
    print(f"   Stress buffer: {buffer:.2f}")

    # Simulate isolation
    print("\n6. Simulating isolation (no interactions)...")
    for i in range(120):
        bonding.tick()

    print(f"   Social state: {bonding.social_state.value}")
    print(f"   Time since interaction: {bonding.time_since_interaction}")

    # Final status
    print("\n7. Final status:")
    status = bonding.get_status()
    print(f"   Total bonds: {status['total_bonds']}")
    print(f"   Strong bonds: {status['strong_bonds']}")
    print(f"   Social needs: {status['social_needs']}")
