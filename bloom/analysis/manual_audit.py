# -*- coding: utf-8 -*-
"""
Manual audit of Bloom labels (annotator M, protocol-guided human annotation).
Protocol: each of the 480 generated items was read and classified against the
cognitive-process dimension of the revised taxonomy (Anderson & Krathwohl):
 1 Remember  : Recognizing, Recalling
 2 Understand: Interpreting, Exemplifying, Classifying, Summarizing, Inferring, Comparing, Explaining
 3 Apply     : Executing, Implementing (carrying out a known procedure on a given input)
 4 Analyze   : Differentiating, Organizing, Attributing
 5 Evaluate  : Checking, Critiquing (judgement against criteria)
 6 Create    : Generating, Planning, Producing (producing a new artefact)

Conventions for boundary cases (fixed BEFORE annotation):
 K1 "determine which of the listed ..." checked against a definition -> Apply (Executing)
 K2 "compare" at the level of definitions -> Understand; comparison on several criteria / trade-off -> Analyze
 K3 "prove <standard textbook result> by <named method>" -> Apply
 K4 reproducing a canonical artefact (RSA from given p, q; bijection (0,1)->(0,5); merge; fast
    exponentiation) -> Apply; producing an artefact under non-trivial constraints -> Create
 K5 "check whether the given solution/statement is correct" -> Evaluate
 K6 a task whose construction is already given in the wording -> Apply
"""
# id -> (audited level, short justification)
DEVIATIONS = {
 # --- declared Remember ---
 "GEN019": (3, "K1: residue classes must be generated, not recalled -> Executing"),
 "GEN062": (2, "second half of the item ('why are they needed') requires an explanation"),
 "GEN073": (3, "listing all 8 subsets = carrying out a procedure"),
 # --- declared Understand ---
 "GEN092": (4, "combinatorial proof via a bijection = Organizing/Attributing"),
 # --- declared Apply ---
 "GEN225": (4, "setting up a recurrence = decomposing the problem into subcases"),
 # --- declared Analyze ---
 "GEN245": (3, "K1: finding prime implicants = Quine-McCluskey algorithm"),
 "GEN258": (3, "K1: checking anagrams against the definition"),
 "GEN259": (3, "K1: checking three relations against the definition"),
 "GEN268": (3, "K1: checking injectivity/surjectivity against the definition"),
 "GEN284": (3, "K1: checking the partial-order axioms"),
 "GEN296": (2, "recognising a named rule of inference = Classifying"),
 "GEN298": (3, "K1: constructing truth tables"),
 "GEN305": (3, "K1: matching against the definition of a linear homogeneous recurrence with constant coefficients"),
 "GEN309": (3, "K1: checking transitivity"),
 "GEN313": (3, "K1: applying known cardinality results"),
 "GEN320": (3, "K1: checking the tree criterion |E|=|V|-1 and connectedness"),
 # --- declared Create ---
 "GEN402": (3, "K6: the circuit layout (two half adders + OR) is given in the item"),
 "GEN418": (3, "K1: enumerating rotation classes = exhaustive check of 16 strings"),
 "GEN423": (3, "K4: standard regex/DFA exercise"),
 "GEN426": (3, "K4: canonical linear bijection x->5x"),
 "GEN437": (3, "K4: executing the RSA procedure on given p, q, m"),
 "GEN440": (3, "K4: the algorithm is named in the item ('repeated squaring')"),
 "GEN448": (3, "K4: writing the definition of surjectivity with quantifiers = reproduction"),
 "GEN452": (3, "K3: classic textbook proof by induction"),
 "GEN454": (3, "K3: short canonical argument (a/2)"),
 "GEN472": (3, "K4: canonical bijection N<->Z"),
 "GEN477": (3, "K4: executing the algorithm for rebuilding a tree from its traversals"),
 "GEN480": (3, "K4: routine tree traversal; cf. GEN170, labelled Apply"),
}
# Agrees with the declared label, but with low confidence (used for the sensitivity analysis):
# (id, level under the strict reading)
BORDERLINE = {
 "GEN050": 3, "GEN057": 3, "GEN077": 1,
 "GEN082": 3, "GEN084": 4, "GEN103": 4, "GEN107": 4, "GEN112": 4, "GEN154": 3,
 "GEN168": 4, "GEN170": 6, "GEN182": 6, "GEN240": 3,
 "GEN246": 3, "GEN250": 2, "GEN253": 5, "GEN274": 2, "GEN286": 5,
 "GEN349": 3, "GEN352": 3, "GEN353": 2, "GEN358": 1, "GEN359": 3, "GEN394": 1,
 "GEN409": 3, "GEN416": 3, "GEN441": 3, "GEN446": 3, "GEN466": 3, "GEN471": 3,
}
# Original 500: six templates. Surface form matches the level by construction,
# but the item contains no subject content -> actual cognitive demand cannot be measured.
TEMPLATE_AUDIT = {
 "T1": (1, "actionable: verb and object are given -> Remember"),
 "T2": (2, "actionable: 'explain the main idea' -> Understand"),
 "T3": (3, "NOT ACTIONABLE: 'solve a problem' - the problem itself is not given"),
 "T4": (4, "NOT ACTIONABLE: 'an example' - the example is not given"),
 "T5": (5, "NOT ACTIONABLE: 'a proposed solution' - the solution is not given"),
 "T6": (6, "actionable: generate one's own example -> Create"),
}
UNDERSPECIFIED_TEMPLATES = {"T3", "T4", "T5"}
