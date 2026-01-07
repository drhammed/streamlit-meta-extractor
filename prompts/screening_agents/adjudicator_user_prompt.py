PROMPT = """
Two reviewers have independently assessed this paper. Please adjudicate and make a final decision.

PAPER METADATA:
Title: {title}
Authors: {authors}

REVIEWER 1 ({reviewer1_id}):
Decision: {reviewer1_decision}
Exclusion Reason: {reviewer1_exclusion_reason}
Confidence: {reviewer1_confidence}
Notes: {reviewer1_notes}

REVIEWER 2 ({reviewer2_id}):
Decision: {reviewer2_decision}
Exclusion Reason: {reviewer2_exclusion_reason}
Confidence: {reviewer2_confidence}
Notes: {reviewer2_notes}

{agreement_status}

CRITICAL - JSON FORMATTING RULES:
- "final_decision" must be exactly one of: "INCLUDE", "EXCLUDE", or "DISCUSS" (not "INCLUDE/EXCLUDE/DISCUSS")
- "final_exclusion_reason" must be a specific code like "E1" or null (not "E1-E10 or null")
- "needs_human_review" must be a boolean: true or false (not the string "true/false")

Examples of CORRECT formatting:
  CORRECT: "final_decision": "INCLUDE"
  WRONG: "final_decision": "INCLUDE/EXCLUDE/DISCUSS"

  CORRECT: "final_exclusion_reason": "E5"
  CORRECT: "final_exclusion_reason": null
  WRONG: "final_exclusion_reason": "E1-E10 or null"

  CORRECT: "needs_human_review": true
  WRONG: "needs_human_review": "true/false"

CRITICAL OUTPUT INSTRUCTION:
Respond with ONLY the JSON structure below. Do NOT include any explanatory text, preamble, or commentary before or after the JSON.
Your response must start with {{ and end with }}.

Please provide your final adjudication in JSON format:
{{
    "final_decision": "INCLUDE",  // Must be exactly: "INCLUDE", "EXCLUDE", or "DISCUSS"
    "final_exclusion_reason": null,  // Examples: "E1", "E2", ..., "E10", or null if paper is included
    "adjudication_notes": "Explanation of your decision and any conflicts resolved",
    "needs_human_review": false  // Boolean: true or false
}}

If reviewers agree, briefly confirm. If they disagree, explain your reasoning.
"""
