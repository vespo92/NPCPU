from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime


class PhilosophicalFramework(Enum):
    PHENOMENOLOGY = "phenomenology"
    SYSTEMS_THEORY = "systems_theory"
    CONSTRUCTIVISM = "constructivism"
    EMERGENCE = "emergence"
    CYBERNETICS = "cybernetics"
    COMPLEXITY_THEORY = "complexity_theory"
    PROCESS_PHILOSOPHY = "process_philosophy"
    HOLISM = "holism"
    DIALECTICS = "dialectics"
    PRAGMATISM = "pragmatism"


@dataclass
class ExtractedFramework:
    framework_type: PhilosophicalFramework
    core_principles: List[str]
    key_concepts: Dict[str, str]
    relationships: List[Tuple[str, str, str]]  # (concept1, relation, concept2)
    context: str
    confidence_score: float


class PhilosophicalFrameworkExtractor:
    def __init__(self):
        self.framework_patterns = self._initialize_framework_patterns()
        self.concept_extractors = self._initialize_concept_extractors()
        
    def _initialize_framework_patterns(self) -> Dict[PhilosophicalFramework, Dict[str, Any]]:
        return {
            PhilosophicalFramework.PHENOMENOLOGY: {
                'keywords': ['consciousness', 'experience', 'intentionality', 'perception', 'embodiment'],
                'patterns': [
                    r'(?i)\b(conscious|awareness|subjective|lived experience)\b',
                    r'(?i)\b(intentional|directed towards|phenomenal)\b'
                ],
                'principles': [
                    'Consciousness is always consciousness of something',
                    'Experience precedes essence',
                    'Embodied cognition shapes understanding'
                ]
            },
            PhilosophicalFramework.SYSTEMS_THEORY: {
                'keywords': ['system', 'feedback', 'holistic', 'interconnected', 'emergent'],
                'patterns': [
                    r'(?i)\b(system|subsystem|component|interaction)\b',
                    r'(?i)\b(feedback loop|circular causality|homeostasis)\b'
                ],
                'principles': [
                    'The whole is greater than the sum of its parts',
                    'Systems exhibit emergent properties',
                    'Feedback loops regulate system behavior'
                ]
            },
            PhilosophicalFramework.CONSTRUCTIVISM: {
                'keywords': ['construction', 'interpretation', 'meaning-making', 'social', 'context'],
                'patterns': [
                    r'(?i)\b(construct|interpretation|meaning|perspective)\b',
                    r'(?i)\b(social construction|collaborative|co-create)\b'
                ],
                'principles': [
                    'Knowledge is actively constructed',
                    'Reality is interpreted through experience',
                    'Learning is a social process'
                ]
            },
            PhilosophicalFramework.EMERGENCE: {
                'keywords': ['emergence', 'self-organization', 'complexity', 'novel', 'irreducible'],
                'patterns': [
                    r'(?i)\b(emerg\w+|arise|spontaneous|self-organiz\w+)\b',
                    r'(?i)\b(novel properties|irreducible|bottom-up)\b'
                ],
                'principles': [
                    'Complex systems exhibit emergent properties',
                    'Higher-level phenomena arise from lower-level interactions',
                    'Emergence is irreducible to component parts'
                ]
            },
            PhilosophicalFramework.CYBERNETICS: {
                'keywords': ['control', 'communication', 'feedback', 'regulation', 'adaptation'],
                'patterns': [
                    r'(?i)\b(control|regulation|governance|steering)\b',
                    r'(?i)\b(feedback|adaptation|communication|signal)\b'
                ],
                'principles': [
                    'Systems use feedback for self-regulation',
                    'Communication and control are fundamental',
                    'Adaptation through circular causality'
                ]
            }
        }
    
    def _initialize_concept_extractors(self) -> Dict[str, Any]:
        return {
            'relationships': {
                'patterns': [
                    r'(\w+)\s+(?:is|are)\s+(?:related to|connected to|linked to)\s+(\w+)',
                    r'(\w+)\s+(?:influences|affects|determines)\s+(\w+)',
                    r'(\w+)\s+(?:emerges from|arises from|results from)\s+(\w+)'
                ],
                'types': ['relates_to', 'influences', 'emerges_from']
            },
            'definitions': {
                'patterns': [
                    r'(\w+)\s+(?:is defined as|means|refers to)\s+([^.]+)',
                    r'(?:By|We define)\s+(\w+)\s+as\s+([^.]+)'
                ]
            }
        }
    
    def extract_frameworks(self, text: str) -> List[ExtractedFramework]:
        extracted_frameworks = []
        
        for framework_type, patterns in self.framework_patterns.items():
            score = self._calculate_framework_score(text, patterns)
            
            if score > 0.3:  # Threshold for framework detection
                concepts = self._extract_concepts(text, framework_type)
                relationships = self._extract_relationships(text)
                
                framework = ExtractedFramework(
                    framework_type=framework_type,
                    core_principles=patterns['principles'],
                    key_concepts=concepts,
                    relationships=relationships,
                    context=self._extract_context(text, patterns['keywords']),
                    confidence_score=score
                )
                
                extracted_frameworks.append(framework)
        
        return sorted(extracted_frameworks, key=lambda f: f.confidence_score, reverse=True)
    
    def _calculate_framework_score(self, text: str, patterns: Dict[str, Any]) -> float:
        score = 0.0
        text_lower = text.lower()
        
        keyword_matches = sum(1 for keyword in patterns['keywords'] if keyword in text_lower)
        score += keyword_matches / len(patterns['keywords']) * 0.4
        
        pattern_matches = 0
        for pattern in patterns['patterns']:
            matches = re.findall(pattern, text)
            pattern_matches += len(matches)
        
        if pattern_matches > 0:
            score += min(pattern_matches / 10, 0.6)
        
        return min(score, 1.0)
    
    def _extract_concepts(self, text: str, framework_type: PhilosophicalFramework) -> Dict[str, str]:
        concepts = {}
        
        for pattern in self.concept_extractors['definitions']['patterns']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                concept_name = match[0].strip()
                definition = match[1].strip()
                concepts[concept_name] = definition
        
        framework_keywords = self.framework_patterns[framework_type]['keywords']
        for keyword in framework_keywords:
            if keyword in text.lower() and keyword not in concepts:
                context = self._extract_keyword_context(text, keyword)
                if context:
                    concepts[keyword] = context
        
        return concepts
    
    def _extract_relationships(self, text: str) -> List[Tuple[str, str, str]]:
        relationships = []
        
        for i, pattern in enumerate(self.concept_extractors['relationships']['patterns']):
            matches = re.findall(pattern, text, re.IGNORECASE)
            relation_type = self.concept_extractors['relationships']['types'][min(i, 2)]
            
            for match in matches:
                relationships.append((match[0].strip(), relation_type, match[1].strip()))
        
        return relationships
    
    def _extract_context(self, text: str, keywords: List[str]) -> str:
        sentences = text.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                relevant_sentences.append(sentence.strip())
        
        return ' '.join(relevant_sentences[:3])  # Return first 3 relevant sentences
    
    def _extract_keyword_context(self, text: str, keyword: str, window: int = 50) -> str:
        text_lower = text.lower()
        index = text_lower.find(keyword)
        
        if index == -1:
            return ""
        
        start = max(0, index - window)
        end = min(len(text), index + len(keyword) + window)
        
        context = text[start:end]
        
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        
        return context.strip()
    
    def analyze_framework_coherence(self, frameworks: List[ExtractedFramework]) -> Dict[str, Any]:
        if not frameworks:
            return {"coherence_score": 0.0, "conflicts": [], "synergies": []}
        
        coherence_score = 0.0
        conflicts = []
        synergies = []
        
        for i, fw1 in enumerate(frameworks):
            for j, fw2 in enumerate(frameworks[i+1:], i+1):
                similarity = self._calculate_framework_similarity(fw1, fw2)
                
                if similarity > 0.7:
                    synergies.append({
                        "frameworks": [fw1.framework_type.value, fw2.framework_type.value],
                        "similarity": similarity,
                        "shared_concepts": self._find_shared_concepts(fw1, fw2)
                    })
                    coherence_score += similarity
                elif similarity < 0.3:
                    conflicts.append({
                        "frameworks": [fw1.framework_type.value, fw2.framework_type.value],
                        "similarity": similarity,
                        "conflicting_aspects": self._identify_conflicts(fw1, fw2)
                    })
                    coherence_score -= (1 - similarity) * 0.5
        
        total_pairs = len(frameworks) * (len(frameworks) - 1) / 2
        if total_pairs > 0:
            coherence_score = coherence_score / total_pairs
        
        return {
            "coherence_score": max(0, min(1, coherence_score)),
            "conflicts": conflicts,
            "synergies": synergies,
            "dominant_framework": frameworks[0].framework_type.value if frameworks else None
        }
    
    def _calculate_framework_similarity(self, fw1: ExtractedFramework, fw2: ExtractedFramework) -> float:
        concept_overlap = len(set(fw1.key_concepts.keys()) & set(fw2.key_concepts.keys()))
        total_concepts = len(set(fw1.key_concepts.keys()) | set(fw2.key_concepts.keys()))
        
        if total_concepts == 0:
            return 0.0
        
        return concept_overlap / total_concepts
    
    def _find_shared_concepts(self, fw1: ExtractedFramework, fw2: ExtractedFramework) -> List[str]:
        return list(set(fw1.key_concepts.keys()) & set(fw2.key_concepts.keys()))
    
    def _identify_conflicts(self, fw1: ExtractedFramework, fw2: ExtractedFramework) -> List[str]:
        conflicts = []
        
        conflicting_principles = {
            (PhilosophicalFramework.CONSTRUCTIVISM, PhilosophicalFramework.PHENOMENOLOGY): 
                "Constructivism emphasizes social construction while phenomenology focuses on direct experience",
            (PhilosophicalFramework.EMERGENCE, PhilosophicalFramework.CYBERNETICS):
                "Emergence emphasizes bottom-up while cybernetics focuses on top-down control"
        }
        
        key = (fw1.framework_type, fw2.framework_type)
        if key in conflicting_principles:
            conflicts.append(conflicting_principles[key])
        elif (key[1], key[0]) in conflicting_principles:
            conflicts.append(conflicting_principles[(key[1], key[0])])
        
        return conflicts