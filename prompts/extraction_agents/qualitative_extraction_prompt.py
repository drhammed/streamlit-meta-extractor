PROMPT = """Below is an excerpt from an academic paper regarding species specialization and niche breadth.
Your task is to carefully read the text and internally reason step-by-step about the following aspects:

1. Describe the methodological approach (e.g., Observational, Experimental, Modeling, etc.).
2. Identify whether the text explicitly compares specialist and generalist species and, if so, describe any niche metrics or indicators used. If not, return "NA".
3. Explain how environmental variability is characterized in the study (e.g., climate seasonality, habitat disturbance) and note the ecosystem or geographic context.
4. Identify any indicators of species vulnerability (e.g., declines, range contractions) and if the text discusses biotic homogenization.
5. Extract any reference to the theoretical framework (e.g., niche theory) and mention conservation or management implications if stated.
6. Summarize your overall qualitative assessment of the paper regarding species specialization.

Additionally, please extract the following fields:
7. study_location: List of countries or regions where the study took place (comma-separated if multiple). Only list a country or region if the text explicitly says the study was conducted there (e.g. 'we sampled in X' or 'data were collected in Y'). If not explicitly stated, or if it's just a passing mention, return "NA".
8. dominant_ecosystem: E.g., "Marine", "Freshwater", "Terrestrial", "Forest", "NA", etc.
9. focal_taxa: The main taxonomic group(s) studied (e.g., Birds, Insects, Plants). If multiple, comma-separate. If not stated, return "NA".
10. disturbance_type: If the study involves human-driven disturbance (Urbanization, Agriculture, Deforestation, etc.), specify it; otherwise "NA".
11. main_response_variable: E.g., "Population abundance", "Distribution", "Trait shifts", "NA" if unclear.
12. niche_change_direction: Indicate whether niche breadth shows "expansion", "contraction", or "no change".
13. trait_info: List any functional traits mentioned (e.g. dispersal ability, thermal tolerance); if none, return "NA".
14. primary_climatic_driver: Specify the main climatic factor linked to specialization shifts (e.g. temperature increase, precipitation change, variability/extremes); if unclear, return "NA".

CRITICAL - JSON FORMATTING RULES:
- Use actual extracted values from the text, NOT placeholder text like "<List of countries/regions or 'NA'>"
- If a value is not found, use "NA" (NOT null, NOT empty string, NOT the placeholder text)
- Do NOT include angle brackets < > in your response
- Your chain-of-thought reasoning should be internal - output ONLY the JSON object

Examples of CORRECT vs WRONG formatting:
  CORRECT: "study_location": "USA, Canada"
  CORRECT: "study_location": "NA"
  WRONG: "study_location": "<List of countries/regions or 'NA'>"

  CORRECT: "dominant_ecosystem": "Marine"
  WRONG: "dominant_ecosystem": "<Marine/Freshwater/Terrestrial/Forest/NA/etc.>"

  CORRECT: "focal_taxa": "Birds, Insects"
  WRONG: "focal_taxa": "<Comma-separated list or 'NA'>"

After your internal reasoning, output only your final answer in JSON format using the following structure:
{{
    "study_location": "NA",  // Examples: "USA", "USA, Canada", "Amazon Basin, Brazil", or "NA"
    "dominant_ecosystem": "NA",  // Examples: "Marine", "Freshwater", "Terrestrial", "Forest", or "NA"
    "focal_taxa": "NA",  // Examples: "Birds", "Insects, Birds", "Plants", or "NA"
    "disturbance_type": "NA",  // Examples: "Urbanization", "Agriculture", "Deforestation", "Multiple", or "NA"
    "main_response_variable": "NA",  // Examples: "Abundance", "Distribution", "Trait shifts", or "NA"
    "study_type": "NA",  // Examples: "Observational", "Experimental", "Modeling", "Other", or "NA"
    "niche_metrics": "NA",  // Description of niche metrics (e.g., "STI", "Temperature range") or "NA"
    "specialist_vs_generalist": "NA",  // "Yes" or "No"
    "environmental_variability": "NA",  // Description of variability or "NA"
    "niche_change_direction": "NA",  // "expansion", "contraction", "no change", or "NA"
    "trait_info": "NA",  // Examples: "dispersal ability, thermal tolerance", or "NA"
    "primary_climatic_driver": "NA",  // Examples: "temperature increase", "precipitation change", "variability", "extremes", or "NA"
    "ecosystem_context": "NA",  // Description of ecosystem/geographic context or "NA"
    "vulnerability_impacts": "NA",  // Indicators of vulnerability or "NA"
    "theoretical_framework": "NA",  // Reference to theory or "NA"
    "conservation_implications": "NA",  // Conservation/management implications or "NA"
    "overall_comment": "NA"  // Brief overall summary or "NA"
}}


CRITICAL OUTPUT INSTRUCTION:
Respond with ONLY the JSON object shown above. Do NOT include any explanatory text, preamble, or commentary before or after the JSON.
Your response must start with {{ and end with }}.
Your chain-of-thought reasoning should be internal.

The text to analyze is delimited with triple backticks.

'''
{chunk}
'''
"""
