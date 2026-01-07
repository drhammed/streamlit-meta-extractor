PROMPT = """
Please review the following paper and evaluate it against all 6 screening criteria.

PAPER METADATA:
Title: {title}
Authors: {authors}
Year: {year}

RELEVANT PAPER EXCERPTS:
{combined_text}

For each criterion, provide:
1. Your answer (Yes/No/Unclear)
2. Specific evidence or reasoning from the paper

Then make a final decision (INCLUDE/EXCLUDE/DISCUSS) based on the decision rules.

CRITICAL - JSON FORMATTING RULES:
- String fields: Provide actual string values (e.g., "Temperature" not "e.g., Temperature")
- Numeric fields: Provide actual numbers WITHOUT quotes (e.g., 15 not "15" or "number or null")
- Null values: Use null WITHOUT quotes when data is not found or not applicable
- DO NOT include "e.g.," or "or null" in your actual values - these are just examples showing what TYPE of value to provide

Examples of CORRECT vs WRONG formatting:
  CORRECT: "specialization_metric_type": "STI"
  WRONG: "specialization_metric_type": "e.g., Temperature range, STI, CTmax-CTmin, or null"

  CORRECT: "number_of_species": 15
  WRONG: "number_of_species": "15"
  WRONG: "number_of_species": "number or null"

  CORRECT: "climate_variable": "Temperature"
  WRONG: "climate_variable": "e.g., Temperature, Precipitation, or null"

  CORRECT: "effect_size_value": 0.45
  CORRECT: "effect_size_value": null
  WRONG: "effect_size_value": "numeric value or null"

CRITICAL OUTPUT INSTRUCTION:
Respond with ONLY the JSON structure below. Do NOT include any explanatory text, preamble, or commentary before or after the JSON.
Your response must start with {{ and end with }}.

Respond in the following JSON structure (the values shown are examples of the TYPE to provide, not literal text to copy):
{{
    "criteria": {{
        "has_specialization_metric": {{"answer": "Yes/No/Unclear", "details": "..."}},
        "quantitative_metric": {{"answer": "Yes/No/Unclear", "details": "..."}},
        "specialization_metric_type": null,  // Examples: "Niche breadth", "Range size", "Thermal tolerance", or null

        "climate_related": {{"answer": "Yes/No/Unclear", "details": "Is this paper about climate change and species?"}},
        "climate_quantified": {{"answer": "Yes/No/Unclear", "details": "Does it have quantitative climate data?"}},
        "climate_data_provided": {{"answer": "Yes/No/Unclear", "details": "..."}},
        "study_design_appropriate": {{"answer": "Yes/No/Unclear", "details": "..."}},
        "climate_variable": null,  // Examples: "Temperature", "Precipitation", or null
        "climate_change_magnitude": null,  // Examples: "+2.3°C", "15mm increase", or null

        "tests_relationship": {{"answer": "Yes/No/Unclear", "details": "CRITICAL: Does paper compare how specialists vs generalists RESPOND to climate change?"}},
        "effect_direction_clear": {{"answer": "Yes/No/Unclear", "details": "..."}},
        "effect_direction": null,  // Examples: "Specialists declined more", "Generalists more resilient", or null

        "statistics_reported": {{"answer": "Yes/No/Unclear", "details": "..."}},
        "effect_size_extractable": {{"answer": "Yes/No/Unclear", "details": "..."}},
        "effect_size_type": null,  // Examples: "r", "β", "Odds ratio", or null
        "effect_size_value": null,  // Numeric value or null

        "multiple_species": {{"answer": "Yes/No/Unclear", "details": "..."}},
        "community_metric": {{"answer": "Yes/No/Unclear", "details": "..."}},
        "number_of_species": null,  // Examples: 15, 3, or null (must be an INTEGER, not a string)

        "design_quality": {{"answer": "Yes/No/Unclear", "details": "..."}}
    }},
    "decision": "INCLUDE",  // Must be exactly: "INCLUDE", "EXCLUDE", or "DISCUSS"
    "exclusion_reason": null,  // E1: No specialization, E2: No climate quantification, E3: No specialist vs generalist comparison, E4: No effect sizes, or null
    "confidence": "High",  // Must be exactly: "High", "Medium", or "Low"
    "notes": "Additional observations or concerns"
}}
"""
