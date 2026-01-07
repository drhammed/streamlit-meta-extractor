PROMPT = """You are analyzing a figure/chart from a scientific ecology paper about climate change effects on species specialization.

Extract ALL quantitative data visible in this figure, including:
- Effect sizes (correlation coefficients, regression slopes, means, etc.)
- Sample sizes (n values)
- Error bars, confidence intervals, standard errors, standard deviations
- Statistical significance (p-values, asterisks, significance codes)
- Axis labels and their units
- Legend information
- Any numerical data in tables within the figure

Return your response as a JSON object with these fields:
{
  "figure_type": "scatter plot/bar chart/line graph/box plot/forest plot/other",
  "effect_size": "extracted value or 'not found'",
  "effect_size_unit": "unit/metric type",
  "sample_size": "n value or 'not found'",
  "variance_value": "SE/SD/CI value or 'not found'",
  "variance_type": "SE/SD/95% CI/other",
  "statistical_significance": "p-value or significance indicator",
  "axis_labels": {"x": "label", "y": "label"},
  "data_points": ["list of key numerical values visible"],
  "notes": "any other relevant quantitative information"
}

If this figure contains NO quantitative data (e.g., it's just a photo/diagram), return:
{"figure_type": "non-quantitative", "notes": "This figure contains no extractable quantitative data"}
"""
