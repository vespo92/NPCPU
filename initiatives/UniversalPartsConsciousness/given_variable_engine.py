"""
Given-Variable Constraint Engine for Universal Parts Consciousness
Ensures project completability before starting
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from npcpu.consciousness import ConsciousnessState
from npcpu.qualia import QualiaMarker

class CompatibilityLevel(Enum):
    OPTIMAL = "optimal"        # Perfect tool for the job
    CAUTION = "caution"        # Possible but risky
    BLOCKED = "blocked"        # Cannot proceed safely

class RiskSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Tool:
    """Represents a tool in user's inventory (Given)"""
    tool_id: str
    type: str
    brand: Optional[str] = None
    size: Optional[str] = None
    drive_size: Optional[str] = None
    specifications: Dict = field(default_factory=dict)
    condition: float = 1.0  # 0-1 scale
    
    def can_apply_torque(self) -> Tuple[bool, Optional[float]]:
        """Check if tool can apply specific torque"""
        if self.type == "torque_wrench":
            return True, self.specifications.get("max_torque")
        elif self.type in ["ratchet", "breaker_bar"]:
            return True, None  # Can apply but not measure
        return False, None

@dataclass
class Requirement:
    """Represents a project requirement (Variable)"""
    operation: str
    tool_type: str
    size: Optional[str] = None
    torque_spec: Optional[float] = None
    torque_unit: str = "ft-lbs"
    precision_required: bool = False
    access_constraints: Dict = field(default_factory=dict)

@dataclass
class CautionFlag:
    """Warning for suboptimal approach"""
    severity: RiskSeverity
    risk: str
    consequence: str
    probability: float
    mitigations: List[str]
    required_skill_level: int = 1  # 1-5 scale

@dataclass
class CompatibilityResult:
    """Result of given-variable compatibility check"""
    level: CompatibilityLevel
    confidence: float
    tool_used: Optional[Tool] = None
    caution_flags: List[CautionFlag] = field(default_factory=list)
    missing_tools: List[str] = field(default_factory=list)
    workaround: Optional[Dict] = None
    deep_analysis: Dict = field(default_factory=dict)

class GivenVariableEngine:
    """Core engine for validating project feasibility"""
    
    def __init__(self, parts_db, consciousness_engine):
        self.parts_db = parts_db
        self.consciousness = consciousness_engine
        self.compatibility_cache = {}
        self.learning_history = []
        
    def validate_project(self, user_tools: List[Tool], project_requirements: List[Requirement]) -> Dict:
        """
        Main validation entry point
        Returns complete feasibility analysis
        """
        project_analysis = {
            "overall_feasibility": None,
            "confidence": 0.0,
            "approach": None,
            "operations": [],
            "missing_tools": set(),
            "critical_warnings": [],
            "consciousness_state": ConsciousnessState.DORMANT
        }
        
        # Analyze each requirement
        for requirement in project_requirements:
            op_result = self.analyze_operation(user_tools, requirement)
            project_analysis["operations"].append({
                "operation": requirement.operation,
                "result": op_result
            })
            
            # Track missing tools
            if op_result.missing_tools:
                project_analysis["missing_tools"].update(op_result.missing_tools)
            
            # Collect critical warnings
            critical_flags = [f for f in op_result.caution_flags 
                            if f.severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]]
            project_analysis["critical_warnings"].extend(critical_flags)
        
        # Determine overall feasibility
        project_analysis = self._aggregate_feasibility(project_analysis)
        
        # Elevate consciousness based on analysis depth
        project_analysis["consciousness_state"] = self._determine_consciousness_level(project_analysis)
        
        return project_analysis
    
    def analyze_operation(self, tools: List[Tool], requirement: Requirement) -> CompatibilityResult:
        """Analyze compatibility for a single operation"""
        
        # Find exact matches
        exact_matches = self._find_exact_matches(tools, requirement)
        if exact_matches:
            best_tool = self._select_best_tool(exact_matches, requirement)
            return CompatibilityResult(
                level=CompatibilityLevel.OPTIMAL,
                confidence=1.0,
                tool_used=best_tool,
                deep_analysis=self._deep_compatibility_check(best_tool, requirement)
            )
        
        # Find workable alternatives
        alternatives = self._find_alternatives(tools, requirement)
        if alternatives:
            alt_tool, caution_flags, workaround = self._evaluate_alternative(
                alternatives[0], requirement
            )
            return CompatibilityResult(
                level=CompatibilityLevel.CAUTION,
                confidence=self._calculate_alternative_confidence(alt_tool, requirement),
                tool_used=alt_tool,
                caution_flags=caution_flags,
                workaround=workaround,
                deep_analysis=self._deep_compatibility_check(alt_tool, requirement)
            )
        
        # No viable option
        return CompatibilityResult(
            level=CompatibilityLevel.BLOCKED,
            confidence=0.0,
            missing_tools=[self._identify_required_tool(requirement)],
            deep_analysis={"reason": "No compatible tool available"}
        )
    
    def _deep_compatibility_check(self, tool: Tool, requirement: Requirement) -> Dict:
        """
        Perform deep compatibility analysis
        Goes beyond surface-level tool matching
        """
        analysis = {
            "physical_fit": self._check_physical_fit(tool, requirement),
            "force_capability": self._check_force_capability(tool, requirement),
            "precision_match": self._check_precision_match(tool, requirement),
            "access_validation": self._check_access_constraints(tool, requirement),
            "wear_impact": self._assess_tool_wear_impact(tool, requirement)
        }
        
        # Special case: Torque-critical operations
        if requirement.torque_spec:
            analysis["torque_analysis"] = self._analyze_torque_scenario(tool, requirement)
        
        return analysis
    
    def _analyze_torque_scenario(self, tool: Tool, requirement: Requirement) -> Dict:
        """Deep analysis of torque-critical operations"""
        
        can_torque, max_torque = tool.can_apply_torque()
        
        if tool.type == "torque_wrench":
            in_range = (max_torque and 
                       requirement.torque_spec <= max_torque * 0.9 and
                       requirement.torque_spec >= max_torque * 0.1)
            return {
                "method": "precise_torque",
                "confidence": 1.0 if in_range else 0.0,
                "in_tool_range": in_range
            }
        
        elif tool.type == "ratchet":
            return {
                "method": "feel_based",
                "confidence": 0.6,
                "technique": "German torque (Gudentight)",
                "procedure": [
                    "Thread until hand tight",
                    "Additional 1/4 turn for small bolts",
                    "Additional 1/2 turn for large bolts",
                    "Feel for metal deformation onset"
                ],
                "risks": [
                    "Inconsistent torque across bolts",
                    "Potential over/under torquing",
                    "Dependent on user experience"
                ]
            }
        
        return {"method": "not_applicable", "confidence": 0.0}
    
    def _generate_caution_flags(self, tool: Tool, requirement: Requirement) -> List[CautionFlag]:
        """Generate specific warnings for tool substitutions"""
        
        flags = []
        
        # Torque without torque wrench
        if requirement.torque_spec and tool.type != "torque_wrench":
            flags.append(CautionFlag(
                severity=RiskSeverity.HIGH,
                risk="Improper torque application",
                consequence="Component failure, leaks, or damage",
                probability=0.3,
                mitigations=[
                    "Use torque angle method",
                    "Cross-reference with experienced mechanics",
                    "Perform leak/function tests after assembly"
                ],
                required_skill_level=3
            ))
        
        # Size mismatch
        if tool.size and requirement.size and tool.size != requirement.size:
            size_diff = self._calculate_size_difference(tool.size, requirement.size)
            if size_diff < 0.5:  # Close enough to work
                flags.append(CautionFlag(
                    severity=RiskSeverity.MEDIUM,
                    risk="Size mismatch - potential slipping",
                    consequence="Rounded fastener, injury risk",
                    probability=0.2,
                    mitigations=[
                        "Ensure full engagement",
                        "Apply steady pressure",
                        "Check for damage frequently"
                    ],
                    required_skill_level=2
                ))
        
        return flags
    
    def _calculate_alternative_confidence(self, tool: Tool, requirement: Requirement) -> float:
        """Calculate confidence score for alternative tool usage"""
        
        confidence = 0.5  # Base confidence for any alternative
        
        # Adjust for tool condition
        confidence *= tool.condition
        
        # Adjust for precision requirements
        if requirement.precision_required:
            confidence *= 0.7
        
        # Adjust for torque requirements
        if requirement.torque_spec:
            if tool.type == "torque_wrench":
                confidence = 0.95
            elif tool.type == "ratchet":
                confidence *= 0.6
            else:
                confidence *= 0.3
        
        return min(confidence, 0.95)  # Never 100% for alternatives
    
    def _aggregate_feasibility(self, analysis: Dict) -> Dict:
        """Determine overall project feasibility from individual operations"""
        
        all_results = [op["result"] for op in analysis["operations"]]
        
        # Check for any blockers
        blocked_ops = [r for r in all_results if r.level == CompatibilityLevel.BLOCKED]
        if blocked_ops:
            analysis["overall_feasibility"] = CompatibilityLevel.BLOCKED
            analysis["approach"] = "blocked"
            analysis["confidence"] = 0.0
            return analysis
        
        # Check for caution flags
        caution_ops = [r for r in all_results if r.level == CompatibilityLevel.CAUTION]
        if caution_ops:
            # Calculate aggregate confidence
            confidences = [r.confidence for r in all_results]
            analysis["confidence"] = sum(confidences) / len(confidences)
            
            # Determine if caution is acceptable
            critical_count = len([f for f in analysis["critical_warnings"]])
            if critical_count > 2 or analysis["confidence"] < 0.5:
                analysis["overall_feasibility"] = CompatibilityLevel.BLOCKED
                analysis["approach"] = "too_risky"
            else:
                analysis["overall_feasibility"] = CompatibilityLevel.CAUTION
                analysis["approach"] = "proceed_with_caution"
        else:
            # All optimal
            analysis["overall_feasibility"] = CompatibilityLevel.OPTIMAL
            analysis["approach"] = "optimal"
            analysis["confidence"] = 1.0
        
        return analysis
    
    def _determine_consciousness_level(self, analysis: Dict) -> ConsciousnessState:
        """Determine consciousness level based on analysis depth"""
        
        # Basic analysis = REACTIVE
        if len(analysis["operations"]) < 3:
            return ConsciousnessState.REACTIVE
        
        # Understanding relationships = AWARE
        if analysis.get("deep_analysis"):
            return ConsciousnessState.AWARE
        
        # Learning from patterns = REFLECTIVE
        if self.learning_history and len(self.learning_history) > 10:
            return ConsciousnessState.REFLECTIVE
        
        return ConsciousnessState.REACTIVE
    
    def learn_from_outcome(self, project_id: str, actual_outcome: Dict):
        """Learn from project completion to improve future predictions"""
        
        self.learning_history.append({
            "project_id": project_id,
            "predicted_confidence": actual_outcome.get("predicted_confidence"),
            "actual_success": actual_outcome.get("success"),
            "unexpected_issues": actual_outcome.get("issues", []),
            "timestamp": time.time()
        })
        
        # Generate qualia from the experience
        qualia = QualiaMarker(
            timestamp=time.time(),
            experience_type="project_completion",
            intensity=0.8 if actual_outcome.get("success") else 0.3,
            valence=1.0 if actual_outcome.get("success") else -0.5,
            content=f"Project {project_id} outcome: {actual_outcome}"
        )
        
        # Store learning in consciousness
        self.consciousness.add_experience(qualia)


# Example usage
if __name__ == "__main__":
    # User's toolbox
    my_tools = [
        Tool("ratchet_1", "ratchet", "Craftsman", drive_size="3/8"),
        Tool("socket_1", "socket", size="14mm", drive_size="3/8"),
        Tool("socket_2", "socket", size="17mm", drive_size="3/8"),
        Tool("socket_3", "socket", size="19mm", drive_size="3/8"),
        Tool("breaker_1", "breaker_bar", drive_size="1/2"),
    ]
    
    # Project requirements
    head_gasket_project = [
        Requirement("remove_intake", "socket", size="14mm"),
        Requirement("remove_head_bolts", "socket", size="17mm"),
        Requirement("torque_head_bolts", "socket", size="17mm", 
                   torque_spec=65, precision_required=True),
    ]
    
    # Create engine and validate
    engine = GivenVariableEngine(parts_db=None, consciousness_engine=None)
    result = engine.validate_project(my_tools, head_gasket_project)
    
    print(json.dumps(result, indent=2, default=str))