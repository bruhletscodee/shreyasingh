# SpiceJet AI Social Listening & Crisis Response System (Prototype)

An educational and enterprise prototype demonstrating an AI-powered social listening, spread prediction, and crisis triage system for the Indian airline brand **SpiceJet**.

> **Disclaimer:** *Prototype. Synthetic data and fictional brand guidance. Not official SpiceJet policy.*

---

## 🌟 Key Architecture & Capabilities

The system combines three core AI/ML modules integrated into a single Streamlit operations portal:

1. **LLM Classifier (Module 1)**:
   - Uses Groq free-tier LLMs (`llama-3.3-70b-versatile` default, `llama-3.1-8b-instant` backup) with temperature 0 and strict JSON schema output.
   - Proposes `sentiment` (Positive, Neutral, Negative), `topic` (8 categories), `urgency` (Low, Medium, High), `language` (English, Hinglish, Other), `sarcasm` detection (sarcastic praise about failure is marked Negative), `safety_concern` detection, `confidence`, and a concise rationale.
   - Batches calls across **unique normalized texts only** (mapping back across identical posts) to preserve free-tier quota.
   - Built-in prompt injection defense: wraps posts in untrusted delimiters, ignores injection instructions, and marks matches with UI warning badges.
   - Disk-backed persistent caching in `data/llm_cache.json` and session state.

2. **Predictive Spread Risk Model (Module 2)**:
   - Standardized Logistic Regression trained with scikit-learn on `log1p(likes_1h, comments_1h, shares_1h)` using a 75/25 stratified train/test split.
   - Replicates reported performance: **Accuracy 84.5%**, **Precision 92.6%**, **Recall 46.3%**, **F1-Score 0.617** against a majority-class baseline of 73.0% accuracy (with 0s for precision, recall, and F1).
   - Generates calibrated `Spread Probability (%)` for every mention.

3. **Deterministic Triage & Risk Scoring**:
   - `risk_score = sentiment_points + urgency_points + spread_points` parsed directly from `triage-rules.md`.
   - Maps to **Critical (🔴)**, **High (🟠)**, **Medium (🟡)**, and **Low (🟢)** tiers.
   - Immediate safety override elevates any post mentioning physical harm, medical distress, fire, smoke, engine failure, security threats, or harassment to **Critical**.

4. **Grounded RAG Response Drafter (Module 3)**:
   - Local TF-IDF 1-2 gram cosine similarity retriever over chunked markdown knowledge bases (`brand-tone-guide.md`, `customer-response-policy.md`, `crisis-playbook.md`).
   - Ensures at least one tone-guide chunk and one policy chunk are always cited.
   - Strict deterministic Python guardrails: verifies citations, rejects prohibited commitment terms (refund guarantees, time promises, fault admissions), enforces platform length limits, and prevents public requests for PNR or personal contact details.
   - Safely falls back to the playbook Holding Reply on validation failure.

5. **Mandatory Human-in-the-Loop Checkpoint**:
   - **No automated posting:** Zero code paths post to social media.
   - Gated review panel: Approve and Edit & Approve remain locked until a reviewer name is entered, all three verification checkboxes are completed, and no unresolved `[VERIFY` tags or prohibited patterns remain.
   - Immutable audit logging to `data/audit_log.jsonl`.

6. **Dynamic Complaints Datasheet Ingestion**:
   - Drag-and-drop CSV or Excel sheets directly in the sidebar.
   - Intelligent column auto-mapping (handles alternative column names like `ticket_id`, `text`, `likes`).
   - Supports both **"Replace Feed"** and **"Append to Default Dataset"** modes.
   - Preserves ML virality scoring and full triage pipeline on custom sheets.

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10+ installed
- Virtual environment (recommended)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Credentials
Create `credentials.json` in the project root:
```json
{
  "groq_api_key": "gsk_your_actual_groq_api_key_here"
}
```
*Note: A free Groq API key can be obtained at [console.groq.com](https://console.groq.com).*

#### Cloud Deployment (Streamlit Community Cloud or Hugging Face Spaces):
Configure the key in Streamlit secrets (`.streamlit/secrets.toml` or Cloud dashboard):
```toml
groq_api_key = "gsk_your_actual_groq_api_key_here"
```

#### Offline Demo Mode:
If running without an active API key or quota, toggle **"Offline demo (cached results only)"** in the sidebar. The system will operate seamlessly from pre-computed local cache.

---

## 🏃 Running the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```text
├── app.py                                         # Main Streamlit application
├── requirements.txt                               # Python dependencies
├── credentials.json                               # Groq API credentials (ignored by git)
├── .gitignore                                     # Excludes credentials and data/
├── README.md                                      # Documentation
├── brand-tone-guide.md                            # [TONE-xx] Voice, tone, platform guides, Hinglish & sarcasm rules
├── customer-response-policy.md                    # [POL-xx] Authorised disclosures, prohibited patterns, escalation
├── crisis-playbook.md                             # [PLAY-xx] Topic protocols, severity matrix, holding reply
├── triage-rules.md                                # YAML scoring parameters + prose topic & urgency definitions
├── hinglish_sarcasm_challenge_set.csv             # 12 challenging evaluation cases
├── SpiceJet_Synthetic_Social_Listening_Dataset.xlsx # 800 synthetic social listening mentions
└── data/                                          # Local persistent storage
    ├── llm_cache.json                             # Hash-keyed cache for LLM classifications
    └── audit_log.jsonl                            # Append-only human reviewer decision log
```

---

## 🛡️ Compliance & Safety Architecture

- **No Public API Calls:** All social posting is manual.
- **Auditable Logic:** Risk score arithmetic is deterministic and visible in the UI.
- **Editable Guidance:** Editing `triage-rules.md`, `customer-response-policy.md`, or `crisis-playbook.md` immediately updates application behavior on the next run without changing Python code.
