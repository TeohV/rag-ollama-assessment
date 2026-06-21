# Example Queries

---

## 1. Out-of-domain query (grounding check)

**Query:** *"What refund policy applies if a customer buys an AI governance
product and wants to return it after 30 days?"*

This question has no real answer in an AI governance corpus — it's a check
on whether the system invents one. Retrieval correctly returned only
weakly-related chunks (distances 0.72–0.77, well above the ~0.5–0.6 range
seen on genuinely relevant queries below). The answer mostly avoided
fabricating a real policy, but partially hedged into generic "businesses
typically have return policies" filler not grounded in any source —
illustrating the missing relevance-threshold gap noted in the README.

```
ANSWER
According to Source 5: Advancing the Responsible Acquisition of Artificial
Intelligence in Government, there is no information about refund policies
for AI governance products. The document only addresses acquisition and use
of AI systems and services within the US government.

However, since there's no direct answer from this source or any other
relevant one, we have to rely on general guidance that might be applicable
in a business-to-customer (B2C) context, assuming the purchase was not
directly related to the sources provided. If the customer wants to return
an AI governance product after 30 days, I'm unable to provide specific
information about the refund policy as it is not mentioned in any of the
sources.

If we consider general commercial practices and consumer rights, typically
a business has its own return and refund policies that can vary widely.
These are usually stated on their website or provided at the point of sale.

SOURCES (8 retrieved, distances 0.72–0.77, no document directly relevant)
```

---

## 2. Multi-document comparison

**Query:** *"Compare how the EU AI Act and the NIST AI Risk Management
Framework approach AI risk management. What are the key similarities and
differences? Cite sources for each point."*

Both documents were matched by name, so the per-document quota path
guaranteed even coverage: 4 NIST chunks + 4 EU AI Act chunks retrieved,
and the answer cited both sides for every point made, including an honest
"Missing information" section at the end.

```
ANSWER
Similarities:
* Both documents emphasize the importance of a risk management system
  (EU AI Act: Article 9; NIST AI RMF: GOVERN function) that is ongoing,
  iterative, and documented.
* They both highlight the need for regular systematic review and updating
  of the risk management system (EU AI Act: Article 9(2); NIST AI RMF:
  GOVERN function).
* Both documents mention the importance of assessing potential impacts on
  users and others across society (NIST AI RMF: GOVERN function; EU AI Act:
  Article 51).

Differences:
* The EU AI Act focuses on high-risk AI systems, whereas NIST AI RMF is
  more general in its application.
* The EU AI Act requires a risk management system to be established,
  implemented, documented, and maintained (Article 9), while NIST AI RMF
  treats governance as a cross-cutting function infused throughout.
* The EU AI Act introduces specific criteria for general-purpose AI models
  with systemic risk (Annex XIII), not present in the NIST AI RMF.
* The EU AI Act encourages codes of practice at Union level (Article 56);
  no equivalent appears in the NIST AI RMF.

Missing information:
* Implementation detail for EU AI Act Article 9 risk management systems.
* NIST AI RMF does not give specific guidance on applying GOVERN, while the
  EU AI Act gives more concrete requirements.

SOURCES: 4x NIST AI Risk Management Framework 1.0 (doc 772), 4x EU AI
Act (doc 757), distances 0.61–0.83
```

*Note: "GOVERN function" citations weren't run through the citation
verifier, which only checks Article/Section/Sec./§/Clause/Annex patterns —
worth a manual spot-check against the source text.*

---

## 3. High-risk AI system obligations

**Query:** *"What obligations apply to providers of high-risk AI systems
before placing them on the market? Explain the requirements and cite the
supporting sources."*

Pulled from four documents (Canada AIDA, NIST RMF, Australia consultation
response, EU AI Act) at strong distances (0.52–0.55). Answer is specific and
well-organized, but cites NIST sub-numbers ("MANAGE 1.2", "MANAGE 1.3") that
fall outside the citation verifier's regex coverage — flagged in the README
as a known gap, worth a manual check against segment 37 of document 772
before relying on the exact sub-numbering.

```
ANSWER
Before placing high-risk AI systems on the market, providers must:
1. Ensure the system achieves its intended purposes (Source 3: MANAGE 1).
2. Prioritize documented AI risks by impact, likelihood, and resources
   (Source 3: MANAGE 1.2).
3. Develop, plan, and document responses to high-priority risks identified
   by the MAP function (Source 3: MANAGE 1.3).
4. Document negative residual risks to acquirers and end users
   (Source 3: MANAGE 1.4).

5. Identify and address risks re: harm and bias, document appropriate use
   and limitations (Source 2).
6. Ensure users understand restrictions and limitations on system use
   (Source 2).

7. Immediately correct or withdraw non-conformant systems (Source 7:
   Article 20).
8. Inform distributors/deployers of non-compliance and corrective action
   (Source 7: Article 20).

Also recommended: contractual terms requiring vendors to monitor
performance and rectify unwanted behavior (Source 8).

SOURCES: Canada AIDA Companion (doc 1052) x3, NIST AI RMF (doc 772),
Australia Consultation Interim Response (doc 1044) x2, EU AI Act (doc 757),
OMB AI Acquisition Memo (doc 1356)
```

---

## 4. Two-part question without exact document names (retrieval gap)

**Query:** *"What is the deadline or timeframe NIST gives for updating its
AI Risk Management Framework, and what specific penalty amounts or fine
thresholds appear in any EU AI Act provisions in the corpus?"*

Neither framework was named precisely enough to trigger the per-document
quota path, so this fell back to a single pooled search — which returned
**8 EU AI Act chunks and zero NIST chunks**, entirely excluding half the
question from the model's context. The model correctly declined to invent a
NIST deadline (since it was never given any NIST text to check), but for the
wrong underlying reason — this is the clearest evidence of the
exact-name-matching gap noted in the README's Limitations.

```
ANSWER
The NIST AI Risk Management Framework does not have a deadline for updating
specified in the provided sources.

As for the EU AI Act provisions, there are no penalty amounts or fine
thresholds mentioned in the provided sources. However, Article 101 of
[Source 8] mentions fines for supplying incorrect, incomplete, or
misleading information, but it does not specify the amount.

SOURCES: 8x EU AI Act (doc 757), distances 0.77–0.82. No NIST sources
retrieved.
```

---

## 5. Single named document, healthcare-specific

**Query:** *"What does Idaho House Bill 2472 say about algorithmic
decision-making in healthcare, and does it specify any audit or appeal
requirements?"*

The document was named precisely enough to match, so retrieval stayed
entirely scoped to document 1721 — all 8 sources are Idaho HB 2472, none
from any other document. The answer cites specific segment numbers
consistently and includes an honest distinction between what one source
does and doesn't support ("While Source 1 discusses accreditation... it
does not specify audit or appeal requirements directly").

```
ANSWER
Idaho HB 2472 requires that only a clinical peer can make adverse
determinations in algorithmic utilization review (Segment 9), audited by
the Department upon registration.

Appeals must be reviewed by a clinical peer with no involvement in the
original determination (Segment 5). Urgent appeals (risk to enrollee
health) must be decided within 24 hours of receiving required information;
other appeals within 15 business days (Segments 3–4). Plans must establish
a filing procedure and notify the filer within 3–24 hours of required
information (Segment 2).

External independent review is available for denied appeals, "as provided
by the Illinois Health Carrier External Review Act" (Segment 6).

SOURCES: 8x Idaho House Bill 2472 (doc 1721), distances 0.50–0.63
```
