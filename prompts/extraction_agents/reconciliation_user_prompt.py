PROMPT = """Two independent extractors have extracted data from the same paper. Please reconcile their results.

EXTRACTOR 1 - QUANTITATIVE DATA:
{extractor1_quant}

EXTRACTOR 2 - QUANTITATIVE DATA:
{extractor2_quant}

EXTRACTOR 1 - QUALITATIVE DATA:
{extractor1_qual}

EXTRACTOR 2 - QUALITATIVE DATA:
{extractor2_qual}

Please compare each field and provide reconciliation results in the following JSON format:

{{
  "final_quantitative_data": {{
    "metric": "<reconciled value or 'CONFLICT'>",
    "effect_size": "<reconciled value or 'CONFLICT'>",
    ...
  }},
  "final_qualitative_data": {{
    "study_location": "<reconciled value or 'CONFLICT'>",
    ...
  }},
  "field_confidence": {{
    "metric": "HIGH/MEDIUM/LOW",
    "effect_size": "HIGH/MEDIUM/LOW",
    "study_location": "HIGH/MEDIUM/LOW",
    ...
  }},
  "conflicts": [
    {{
      "field": "effect_size",
      "extractor1_value": "0.5",
      "extractor2_value": "0.8",
      "conflict_type": "NUMERIC_DIFFERENCE",
      "confidence": "LOW",
      "recommendation": "Human review needed - values differ by >5%"
    }}
  ],
  "auto_resolved": [
    {{
      "field": "study_location",
      "extractor1_value": "USA",
      "extractor2_value": "United States",
      "resolved_value": "USA",
      "confidence": "MEDIUM",
      "reason": "Semantically equivalent geographic names"
    }}
  ],
  "needs_human_review": true/false,
  "overall_confidence": "HIGH/MEDIUM/LOW",
  "reconciliation_notes": "Summary of reconciliation process and key decisions"
}}

IMPORTANT RECONCILIATION LOGIC:
1. **AUTO-RESOLVE** semantic equivalences:
   - Geographic names: "USA" = "United States" = "U.S." → "USA"
   - Taxonomic names: "Birds" = "Aves" → "Birds"
   - Numeric values within 5% → average value

2. **AUTO-RESOLVE** geographic containment (sub-region to country):
   - "California" vs "USA" → "California, USA" (more specific)
   - "California" vs "California, USA" → "California, USA" (already has country)
   - "Amazon Basin" vs "Amazon Basin, Brazil" → "Amazon Basin, Brazil"

3. **FLAG as CONFLICT** multi-location/multi-taxa discrepancies:
   - "USA, Canada" vs "USA" → CONFLICT (multi-country vs single country - needs verification)
   - "USA, Mexico, Canada" vs "North America" → CONFLICT (specific countries vs regional - needs human decision)
   - "Insects, Birds" vs "Birds" → CONFLICT (multi-taxa vs single taxa - needs verification)
   - One extractor may have found more complete information, flag for human to verify

4. **FLAG as CONFLICT** completely different values:
   - "USA" vs "Brazil" → CONFLICT
   - "Marine" vs "Freshwater" → CONFLICT
   - Numbers >5% different → CONFLICT

5. Set overall_confidence based on proportion of HIGH confidence fields
6. Set needs_human_review=true if conflicts exist in critical fields (effect_size, sample_size, variance_value, study_location)

CRITICAL OUTPUT INSTRUCTION:
Respond with ONLY the JSON object as specified in the structure. Do NOT include any explanatory text, preamble, or commentary before or after the JSON.
Your response must start with {{ and end with }}.
"""
