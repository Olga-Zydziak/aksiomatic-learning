🧠 Axiomatic Learning Engine
Proof-Carrying Decision System for AML / Fraud / High-Trust AI

This project implements a formal, deterministic, fully auditable decision engine designed for high-stakes financial domains such as:

Anti-Money Laundering (AML)

Fraud detection

KYC / Compliance policy enforcement

Credit decisioning

High-trust regulated AI (banking, fintech, insurance)

Instead of machine-learning-based black box predictions, the system uses:

✔ Formal Logic (Z3 SMT Solver)
✔ Natural-Language-style Rules (DSL)
✔ Axioms instead of ML models
✔ Deterministic, reproducible decisions
✔ Proof bundles for every decision
✔ Human-readable explanations

This is a fully-working Proof-of-Concept suitable for demonstration to:

recruiters,

senior engineers,

solution architects,

banking/fintech decision makers.

🎯 Project Purpose

Modern financial institutions must meet strict requirements:

zero hallucinations (no LLM guessing in critical workflows),

auditability (why was a transaction flagged?),

reproducibility (same input → same output, guaranteed),

regulatory transparency (KNF, FCA, ESMA, internal audit),

explainability (analysts must understand each decision).

This project solves that by combining:

Axiomatic Learning

Rules are defined as axioms in a structured natural-language DSL.
The system converts these rules into formal logic (SMT formulas) and guarantees:

absence of contradictions,

conflict detection (UNSAT cores),

deterministic decisions,

verifiable proofs of every decision.

🧩 System Architecture
+----------------------+     +--------------------------+
|   fraud_rules.yaml   | →→  |  Rule Loader (YAML/JSON) |
+----------------------+     +--------------------------+
            ↓                          ↓
  (Natural language rules)   (Parsed + validated rules)
            ↓                          ↓
+-------------------------+    +-------------------------+
| NL Rule Parser (DSL)    |    |  AxiomKernel (Z3 SMT)   |
| - "If amount > 10000…"  |    | - conflict detection    |
| - strict grammar        | →→ | - UNSAT cores           |
+-------------------------+    | - proof bundle          |
                               +-----------+-------------+
                                           ↓
                               +--------------------------+
                               | Explanation Engine       |
                               | (human-friendly reasons) |
                               +--------------------------+
                                           ↓
                               +--------------------------+
                               | Decision + Audit Log      |
                               | (JSONL proof-carrying)    |
                               +--------------------------+

✨ Key Features
✔ Natural-Language Rule DSL

Example:

If amount > 10000 and tx_count_24h > 5 then is_suspicious = true

✔ Formal Decision Engine (Z3 SMT)

Every rule becomes a logical constraint (implication).

Kernel evaluates rules deterministically.

Detects contradictory rule sets (UNSAT core).

Provides full solving model.

✔ Proof-Carrying Decisions

Each decision contains:

rule_version,

SAT/UNSAT/ERROR status,

satisfied and violated axioms,

active/inactive rules (vacuous truth),

UNSAT core (if conflict),

full solver model (values of variables),

full audit-ready JSON bundle.

✔ Explainability Layer

Human-readable narrative:

Why was the transaction FLAGGED?

Which rule(s) fired?

Which rules did not apply?

Were there contradictions?

Were there technical errors?

✔ RuleSets from YAML / JSON

Rules are not baked into code:

Analysts can edit them,

YAML is versioned in Git,

CI can validate rule syntax,

Multiple versions (v1, v2…) can be tested.

✔ Audit Trail

All decisions are stored as JSONL records:

timestamp,

decision_id (UUID),

facts,

proof bundle,

ruleset version.

📦 Repository Structure
project/
│
├── axiomatic_kernel.py         # Core SMT-based decision engine
├── nl_rule_parser.py           # Natural-language rule DSL parser
├── explanation_engine.py       # Human-readable explanations
├── rules_io.py                 # Ruleset YAML/JSON loader + kernel integration
│
├── rules/
│   └── fraud_rules_v1.yaml     # Example ruleset (editable)
│
├── logs/
│   └── ...                     # JSONL audit trail
│
├── notebooks/
│   └── demo_rules_from_yaml.ipynb  # Full project demo notebook
│
└── README.md

▶️ How to Run the Project
1. Clone the repository
git clone <your-repo-url>
cd project

2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install z3-solver pyyaml


or, if you have requirements.txt, simply:

pip install -r requirements.txt

4. Launch the demo notebook
jupyter notebook


Open:

notebooks/demo_rules_from_yaml.ipynb


Run all cells to see:

loading rules from YAML,

applying them to the kernel,

evaluating example cases,

detecting conflicts (UNSAT),

generating explanations,

generating audit logs.

📄 Example Ruleset (fraud_rules_v1.yaml)
ruleset_id: "fraud_rules_v1"
version: "1.0.0"
description: "Basic fraud/AML rules"

rules:
  - id: "fraud.high_amount"
    text: "If amount > 10000 then is_suspicious = true"
    description: "High transaction amount"
    enabled: true
    severity: "HIGH"
    tags: ["fraud", "amount"]

  - id: "fraud.velocity"
    text: "If tx_count_24h > 5 then is_suspicious = true"
    description: "High transaction velocity"
    enabled: true
    severity: "MEDIUM"
    tags: ["fraud", "velocity"]

  - id: "kyc.pep_block"
    text: "If is_pep == true then decision = \"BLOCK\""
    description: "Block PEP customers"
    enabled: false            # disabled → rule ignored
    severity: "CRITICAL"
    tags: ["kyc", "pep"]

🧪 Example of Using the Kernel Programmatically
case = {
    "amount": 15000,
    "tx_count_24h": 7,
    "is_pep": False,
}

bundle = kernel.evaluate(case)
explanation = explainer.explain(bundle)

print(json.dumps(bundle, indent=2))
print(explanation.to_text("en"))

🌟 Why This Project Matters (For Recruiters)

This project showcases skills that are rare and highly valued:

✔ Building formal logic engines
✔ AI that is deterministic and auditable
✔ Hybrid AI / rule-based / SMT reasoning
✔ Explainability in regulated environments
✔ Parsing DSLs and designing interpreters
✔ Working with Z3 (SMT solving)
✔ Architecture for high-trust financial systems
✔ Clean, production-ready engineering
✔ Notebook demos and solid documentation

This is far beyond “toy ML projects” and demonstrates real engineering capabilities suitable for:

financial institutions,

compliance platforms,

fintech R&D,

model governance teams,

high-stakes AI design positions.