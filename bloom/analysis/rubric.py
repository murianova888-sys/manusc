# -*- coding: utf-8 -*-
"""Annotator R: reproducible rule-based classifier of cognitive level.
Not ML: deterministic rules "verb + object + specificity of the input".
Rules are checked from higher to lower levels; the first rule that fires wins.
Conventions match the manual protocol (K1..K6 in manual_audit.py)."""
import re

LEVELS = {1:"Remember",2:"Understand",3:"Apply",4:"Analyze",5:"Evaluate",6:"Create"}

CANON = [r"\bfull adder\b", r"\brsa key pair\b", r"\brepeated squaring\b",
         r"\bmerges two sorted lists\b", r"\bbijection between the natural numbers\b",
         r"\bbijection between the intervals\b", r"\bdiagonal argument\b"]
# input given explicitly: concrete values/lists -> an execution task
GIVEN_DATA = r"(=\s*[-−]?\d|\bΣm\b|\b\d+\s*,\s*\d+\s*,|\bmod \d|\(\s*\d|\bp = \d|frequencies)"
CREATE_V   = r"\b(design|devise|create|invent|compose|propose|formulate|develop|construct|generate)\b"
NOVELTY    = r"\b(original|your own|of your choice|real[- ]world|real[- ]life|realistic|scenario|policy|puzzle|conjecture|word problem|models?\b|specification|scheme|different from|such that|but (no|not|neither)|with exactly|non-isomorphic|that is not)\b"
EVAL_V     = r"\b(evaluate this|evaluate the (claim|statement|solution|simplification|argument|classic|following)|judge (whether|the)|assess (whether|the)|critique|criticiz|decide (whether|which)|defend (your|the)|justify your (answer|view|choice|verdict|conclusion)|better (method|specification|choice)|more (suitable|appropriate|efficient|illuminating)\b)"
EVAL_CUE   = r"(a student (claims|solved|simplified|translated|counts|\"proves\"|proves)|claims that|the claim that|a textbook states|critique the|is always|adequate model|a good structure|a reasonable choice|acceptable for|secure today|are consistent|name the fallacy)"
ANALYZE_V  = r"\b(analyz\w+|compare|contrast|examine|break down|distinguish|differentiate|locate where|arrange the functions)\b"
IDENT_V    = r"\b(identify (which|what|the)|which of (these|the following)|classify (each|k|e))"
APPLY_V    = r"\b(compute|calculate|solve|find|trace|simplify|convert|sort|insert|apply|use (the|a|dijkstra|kruskal|warshall|prim|contraposition|universal|set identities|a truth table|a karnaugh)|how many|in how many|check whether|determine (whether|how many|the (chromatic|truth|least))|show that|prove (by|that|it)|give a (direct proof|big-o)|translate|express|draw (a|the) (logic|hasse)|build a|write (a )?(pseudocode|recurrence|regular expression|regular|the adjacency|the negation|the symbolic)|list all|write out|evaluate the boolean)\b"
UNDERST_V  = r"\b(explain|describ\w+|interpret|summariz\w+|paraphrase|in your own words|why (are|is|does|do|not)|illustrate)\b"
REMEMBER_V = r"^\s*(define|state (the|kleene|fermat)|list (the|three)|name (three|the)|recall|write (the |down |out )?(definition|formula|recurrence|symbolic|duality|inclusion|de morgan)|what (is|are)|how many edges does)\b"

def rubric_level(q: str) -> int:
    s = q.strip(); low = s.lower()
    if any(re.search(p, low) for p in CANON):
        return 3                                   # K4: canonical artefact
    if re.search(CREATE_V, low):
        if re.search(NOVELTY, low):     return 6
        if re.search(GIVEN_DATA, s):    return 3   # K6: input given -> execution
        if re.search(EVAL_V, low) or re.search(EVAL_CUE, low): return 5
        return 6
    if re.search(EVAL_V, low) or re.search(EVAL_CUE, low): return 5
    if re.search(IDENT_V, low):
        return 3 if re.search(GIVEN_DATA, s) or re.search(r";", s) else 4   # K1
    if re.search(ANALYZE_V, low):  return 4
    if re.search(APPLY_V, low):    return 3
    if re.search(UNDERST_V, low):  return 2
    if re.search(REMEMBER_V, low): return 1
    return 2
