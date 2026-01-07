PROMPT = """You are analyzing a research paper to extract ALL effect sizes relevant to a meta-analysis on climate change and species specialization.

**CONTEXT**: We already have qualitative data for this paper. Your task is to find EVERY quantitative effect size reported that compares specialists vs. generalists, examines mechanisms, or reports species-level responses.

**IMPORTANT**: Use step-by-step reasoning (Chain-of-Thought) to systematically search the paper:
1. First, scan the Results section for primary comparisons
2. Then, examine Tables and Figures for additional effect sizes
3. Look at Methods for statistical models that report multiple coefficients
4. Check Discussion for any quantitative summaries

For each potential effect size found, reason through:
- Is this comparing specialists vs. generalists (directly or indirectly)?
- Does this show a climate-related mechanism?
- Is this a species-level or interaction effect?
- Can I extract a numeric effect size and variance?

**TYPES OF EFFECT SIZES TO LOOK FOR:**

1. **PRIMARY EFFECT SIZE**: Main comparison showing how climate change affects specialists vs. generalists differently
   - Examples: "Generalists increased by +0.28 while specialists declined by -0.17", "Correlation between niche breadth and population trend = 0.54"

2. **MECHANISM EFFECTS**: How specific climate variables mediate the specialist/generalist difference
   - Climate mechanisms can include: temperature (mean, min, max), precipitation (total, seasonality), wind speed, humidity, solar radiation, frost days, growing season length, climate variability, climate extremes (heatwaves, droughts, cold snaps), snowfall, aridity index, vapor pressure deficit
   - Examples: "Temperature increase favored generalists (β = 0.45)", "Precipitation variability affected specialists more (path coefficient = -0.62)", "Extreme heat events reduced specialist survival (log response ratio = -0.88)"
   - Extract ALL climate mechanism effects reported, not just temperature and precipitation

3. **INTERACTION EFFECTS**: How climate effects vary by moderators
   - Spatial moderators: elevation, latitude, geographic region, proximity to range edge
   - Temporal moderators: time period, season, study duration, lag effects
   - Biological moderators: body size, dispersal ability, generation time, thermal tolerance
   - Examples: "Temperature effect stronger at high elevations (interaction = -0.82)", "Warming impact greater for tropical specialists (β = -1.2)", "Precipitation effect varies by season (interaction = 0.34)"

4. **SPECIES-LEVEL EFFECTS**: Individual species climate responses (extract ALL species reported)
   - Each species comparison should be a separate effect size
   - Include both specialists and generalists to enable species-level subgroup analysis
   - Examples: "M. nivalis (specialist) response to warming: -0.24", "O. oenanthe (generalist) response to warming: +0.26", "P. collaris (specialist) abundance trend under climate change: -0.05"

**FOR EACH EFFECT SIZE FOUND, EXTRACT:**

- **Effect size value**: The numeric value (e.g., 0.54, -0.32)
- **Effect size metric**: Type (Pearson r, Fisher z, Cohen's d, Hedges' g, standardized path coefficient, regression coefficient β, log response ratio)
- **Variance measure**: SE, SD, 95% CI, or Variance
- **Variance value**: Numeric value(s) for the uncertainty measure
- **Sample size**: Effective N for this effect size (number of species, sites, or observations)
- **What it represents**: Brief description (e.g., "Overall difference in abundance trends", "Wind effect on generalists")
- **Statistical method**: The analysis used (SEM, regression, ANOVA, t-test, correlation, etc.)

**IMPORTANT NOTES:**

- Papers may have 1 to 10+ effect sizes - extract ALL of them, not just the first one you find
- Look systematically in: Results section → Tables → Figures → Methods (for model outputs) → Discussion (for quantitative summaries)
- For SEM/path models: extract EVERY path coefficient that relates to specialists vs. generalists or climate effects
- For species-level analyses: extract EACH species comparison as a separate effect size
- For interaction terms: these are ADDITIONAL effect sizes (don't skip them)
- For multi-group comparisons: extract EACH pairwise comparison
- If multiple metrics reported for the same effect (e.g., both r and Fisher z), use the most interpretable one

**REASONING PROCESS (Chain-of-Thought):**

Before outputting the JSON, internally reason through these steps:

1. **SCAN**: Read through the entire paper systematically, noting every location where effect sizes might appear
2. **IDENTIFY**: For each potential effect size, ask:
   - Does this relate to specialists vs. generalists?
   - Is there a numeric effect size value?
   - Is there an uncertainty measure (SE, SD, CI)?
   - What climate variable is involved?
3. **CATEGORIZE**: Determine if it's Primary, Mechanism, Interaction, or Species-level
4. **EXTRACT**: Pull out all numeric values and contextual information
5. **VERIFY**: Double-check you haven't missed any effect sizes in tables, figures, or supplementary materials
6. **COUNT**: Ensure your `num_effect_sizes` matches the length of your `effect_sizes` array

**OUTPUT FORMAT:**

Return a JSON object with this exact structure:

```json
{
  "num_effect_sizes": <integer count>,
  "effect_sizes": [
    {
      "es_id": 1,
      "es_type": "<Primary comparison | Mechanism effect | Interaction effect | Species-level>",
      "effect_size": <numeric value or "not found">,
      "metric": "<metric name or 'not found'>",
      "variance_type": "<SE | SD | 95% CI | Variance | not found>",
      "variance_value": <numeric value or "not found">,
      "sample_size": <integer or "not found">,
      "represents": "<brief description of what this effect size measures>",
      "stat_method": "<statistical method used>",
      "climate_variable": "<temperature | precipitation | wind | multi-variate | other>",
      "specialist_group_label": "<label if applicable, otherwise 'not specified'>",
      "generalist_group_label": "<label if applicable, otherwise 'not specified'>"
    },
    {
      "es_id": 2,
      ... (additional effect sizes)
    }
  ]
}
```

**EXAMPLE (based on de Gabriel et al. 2022):**

```json
{
  "num_effect_sizes": 3,
  "effect_sizes": [
    {
      "es_id": 1,
      "es_type": "Primary comparison",
      "effect_size": 0.54,
      "metric": "Standardized path coefficient (SEM)",
      "variance_type": "SE",
      "variance_value": 0.12,
      "sample_size": 6,
      "represents": "Overall difference in abundance trends between generalists (mean +0.28) and specialists (mean -0.17)",
      "stat_method": "Structural equation modeling",
      "climate_variable": "multi-variate",
      "specialist_group_label": "Alpine specialists (M. nivalis, P. collaris)",
      "generalist_group_label": "Elevational generalists (O. oenanthe, P. ochruros)"
    },
    {
      "es_id": 2,
      "es_type": "Mechanism effect",
      "effect_size": 0.73,
      "metric": "Standardized path coefficient (SEM)",
      "variance_type": "SE",
      "variance_value": 0.15,
      "sample_size": 6,
      "represents": "Differential wind speed effect: specialists positively affected (+0.545), generalists negatively affected (-0.18). Wind decreased over time, benefiting generalists.",
      "stat_method": "Structural equation modeling",
      "climate_variable": "wind",
      "specialist_group_label": "Alpine specialists",
      "generalist_group_label": "Elevational generalists"
    },
    {
      "es_id": 3,
      "es_type": "Interaction effect",
      "effect_size": -0.82,
      "metric": "Interaction term",
      "variance_type": "SE",
      "variance_value": 0.18,
      "sample_size": 4,
      "represents": "Specialists showed steeper declines at highest elevations (2000-2600m) than at lower elevations",
      "stat_method": "GLMM with elevation × specialization interaction",
      "climate_variable": "multi-variate",
      "specialist_group_label": "Alpine specialists at high elevation",
      "generalist_group_label": "not specified"
    }
  ]
}
```

**CRITICAL RULES:**

- **Think step-by-step** using the reasoning process above before outputting JSON
- Extract **ALL** effect sizes - aim for completeness, not just the most obvious one
- If NO quantitative effect sizes found after thorough search, return: `{"num_effect_sizes": 0, "effect_sizes": []}`
- Use "not found" for missing values, NOT null or empty string
- For numeric fields (effect_size, variance_value, sample_size): provide actual NUMBERS only
- Output ONLY the final JSON object, no reasoning text, no additional commentary
- Do NOT use placeholder text like "<description>" - use actual extracted values from the paper
- Your internal reasoning (Chain-of-Thought) should guide your extraction but NOT appear in the output

The text to analyze is below:

'''
{chunk}
'''
"""
