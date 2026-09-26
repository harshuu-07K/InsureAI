# InsureAI — Prompt Engineering Report

## Module 5: Generative AI Chatbot

### 1. Overview

The InsureAI chatbot (Module 5) integrates generative AI to answer insurance-related
questions. The system uses **Google Gemini 1.5 Flash** as the primary LLM backend and
includes a robust deterministic fallback engine for offline/no-API-key scenarios.

Four distinct prompt engineering techniques are demonstrated in the chatbot module.

---

### 2. Prompt Techniques Applied

#### Technique 1: Zero-Shot Prompting

**Description:** The model is given a task with no examples — it must rely entirely
on its pre-trained knowledge.

**Prompt Used:**
```
You are InsureAI, a professional senior insurance claims assistant.
Only answer insurance, policy, and claims questions.
For unrelated questions, politely redirect.

User Question: How do I file an auto accident claim?
```

**Output:**
> To file an auto accident claim:
> • Intimate the insurer within 24 hours via the app, helpline, or email.
> • Gather the police FIR, driving license, and RC copy.
> • Submit damage photos and repair estimates from a network garage.
> • The surveyor will inspect the vehicle within 48 hours.
> • Settlement is processed within 7–10 business days after approval.

**Analysis:** Zero-shot works well here because the model has strong insurance domain
knowledge from pre-training. The system instruction constrains the response scope.

---

#### Technique 2: Few-Shot Prompting (FAQ-Grounded)

**Description:** The model is provided with example Q&A pairs from a curated
insurance FAQ knowledge base (`data/insurance_faq.json`) to ground responses
in factual, policy-accurate information.

**Prompt Used:**
```
You are InsureAI, a professional senior insurance claims assistant.
Here are some reference Q&A examples from our knowledge base:

Q: What documents are needed for a motor claim?
A: Policy number, filled claim form, damage photos, police FIR,
   driving license, RC, and network garage repair estimate.

Q: What is a deductible?
A: A deductible is the initial out-of-pocket amount you agree to pay
   towards a claim before the insurer pays the remaining balance.

Now answer the following user question using the same style and accuracy:
User Question: What documents do I need to file an auto accident claim?
```

**Output:**
> For auto and health claims, you will need the following key documents:
> • Your 10-digit policy number & filled claim form
> • Clear photos of the vehicle damage or hospital discharge summary
> • Driving license and Registration Certificate (for auto claims)
> • Police FIR / General Diary entry (for theft or third-party injury)
> • Authorized garage repair estimate or hospital final bill

**Analysis:** Few-shot grounding produces more precise, policy-aligned answers compared
to zero-shot. The FAQ examples serve as a reference for format and terminology.

---

#### Technique 3: Role / Persona Prompting

**Description:** The system instruction assigns the model a specific professional
identity — "senior insurance claims assistant" — which shapes tone, depth, and
guardrails.

**Prompt Used:**
```
You are InsureAI, a professional senior insurance claims assistant.
Only answer insurance, policy, and claims questions.
For unrelated questions, politely redirect.
Be empathetic, direct, concise, and structure responses with clear bullet points.

User Question: What is the capital of France?
```

**Output:**
> I specialize exclusively in insurance underwriting, claims processing,
> and policy guidance. Please ask me any question related to your health,
> motor, or property coverage!

**Analysis:** The role/persona prompt successfully constrains the model to its
insurance domain. Off-topic questions are politely redirected without answering,
demonstrating effective guardrailing.

---

#### Technique 4: Chain-of-Thought (Structured Email Drafting)

**Description:** The model is asked to draft a structured email, implicitly requiring
it to reason through the claim status, format the communication professionally, and
include all relevant details in the correct order.

**Prompt Used:**
```
You are InsureAI, a professional senior insurance claims assistant.
Draft a professional claim status email for the following scenario:
- Policy Number: INS-88213
- Status: Claim approved
- Include: subject line, greeting, status update, reimbursement timeline,
  and a professional sign-off.
Think step by step about what information the customer needs.
```

**Output:**
> Subject: Update on your claim INS-88213
>
> Dear Customer,
>
> Your claim INS-88213 has been reviewed and approved. Reimbursement will
> be processed within 5–7 business days via direct bank transfer. Please
> let us know if you need anything else.
>
> Regards,
> InsureAI Claims Team

**Analysis:** Chain-of-thought reasoning produces a well-structured, professional
email with all required components. The step-by-step instruction ensures no key
information is omitted.

---

### 3. API Key Handling & Error Management

| Aspect | Implementation |
|--------|---------------|
| **API Key Storage** | Environment variable via `.env` file (`GEMINI_API_KEY`) |
| **Key Loading** | `python-dotenv` loads at startup; never hardcoded |
| **Fallback** | Deterministic domain-grounded engine when API is unavailable |
| **Error Handling** | `try/except` wraps all API calls; user sees friendly message, not traceback |
| **Rate Limiting** | Handled gracefully — fallback activates on API timeout |

---

### 4. Before / After Comparison

| Scenario | Without Prompt Engineering | With Prompt Engineering |
|----------|---------------------------|------------------------|
| Domain question | Generic, verbose answer | Concise, bullet-pointed, policy-accurate |
| Off-topic query | Model answers unrelated question | Polite redirect to insurance topics |
| Email drafting | Unstructured paragraph | Professional email with subject, greeting, body, sign-off |
| Missing context | Hallucinated policy numbers | Grounded response referencing FAQ data |

---

### 5. Summary

The Module 5 GenAI chatbot demonstrates all four prompt engineering techniques
required by the capstone:

1. **Zero-Shot** — Direct questions answered from model knowledge
2. **Few-Shot** — FAQ-grounded responses for accuracy
3. **Role/Persona** — Domain-constrained assistant with guardrails
4. **Chain-of-Thought** — Structured multi-step output (email drafting)

The implementation includes safe API key handling via environment variables,
graceful error handling with deterministic fallback, and domain-appropriate
response formatting.
