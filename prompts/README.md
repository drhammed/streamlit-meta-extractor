# Prompts Directory

This directory contains all the prompts used by the data extraction pipeline agents. The prompts are organized into subdirectories based on their purpose.

## Directory Structure

```
prompts/
├── screening_agents/          # Prompts for paper screening phase
│   ├── reviewer_system_prompt.txt
│   ├── reviewer_user_prompt.txt
│   ├── adjudicator_system_prompt.txt
│   └── adjudicator_user_prompt.txt
├── extraction_agents/         # Prompts for data extraction phase
│   ├── quantitative_extraction_prompt.txt
│   ├── qualitative_extraction_prompt.txt
│   ├── reconciliation_system_prompt.txt
│   └── reconciliation_user_prompt.txt
├── prompt_loader.py           # Utility module to load prompts
└── README.md                  # This file
```

## Customizing Prompts

You can customize the prompts to fit your specific research needs **without touching the codebase**. Simply edit the `.py` or `.txt` files in this directory.

### Screening Agents Prompts

#### `reviewer_system_prompt.txt`
- **Purpose**: Defines the role and instructions for the screening reviewer agents
- **Used by**: Both Reviewer 1 and Reviewer 2
- **Customization**: Modify inclusion criteria, decision rules, or screening expectations

#### `reviewer_user_prompt.txt`
- **Purpose**: Template for the user prompt sent to reviewers
- **Variables**: `{title}`, `{authors}`, `{year}`, `{combined_text}`
- **Customization**: Adjust the structure or add additional fields to extract

#### `adjudicator_system_prompt.txt`
- **Purpose**: Defines the role and instructions for the adjudicator agent
- **Used by**: Adjudicator agent to resolve conflicts between reviewers
- **Customization**: Modify conflict resolution strategy or human review criteria

#### `adjudicator_user_prompt.txt`
- **Purpose**: Template for the user prompt sent to the adjudicator
- **Variables**: `{title}`, `{authors}`, `{reviewer1_id}`, `{reviewer1_decision}`, `{reviewer1_exclusion_reason}`, `{reviewer1_confidence}`, `{reviewer1_notes}`, `{reviewer2_id}`, `{reviewer2_decision}`, `{reviewer2_exclusion_reason}`, `{reviewer2_confidence}`, `{reviewer2_notes}`, `{agreement_status}`
- **Customization**: Adjust the format or add additional context

### Extraction Agents Prompts

#### `quantitative_extraction_prompt.txt`
- **Purpose**: Instructions for extracting quantitative data (effect sizes, statistics, etc.)
- **Used by**: Both QuantitativeExtractionAgent instances (Extractor 1 and Extractor 2)
- **Variables**: `{chunk}` (text chunk to analyze)
- **Customization**: Add/remove fields to extract, modify instructions for specific metrics

#### `qualitative_extraction_prompt.txt`
- **Purpose**: Instructions for extracting qualitative metadata (study location, taxa, etc.)
- **Used by**: Both QualitativeExtractionAgent instances (Extractor 1 and Extractor 2)
- **Variables**: `{chunk}` (text chunk to analyze)
- **Customization**: Add/remove fields, modify taxonomic categories or ecosystem types

#### `reconciliation_system_prompt.txt`
- **Purpose**: Defines the role and rules for the reconciliation agent
- **Used by**: ExtractionReconciliationAgent to merge results from dual extractors
- **Customization**: Adjust conflict resolution rules, confidence thresholds, or semantic equivalence rules

#### `reconciliation_user_prompt.txt`
- **Purpose**: Template for the reconciliation prompt
- **Variables**: `{extractor1_quant}`, `{extractor2_quant}`, `{extractor1_qual}`, `{extractor2_qual}`
- **Customization**: Modify the output format or add additional reconciliation logic

## Usage in Code

The prompts are automatically loaded by the data extraction pipeline using the `prompt_loader.py` module:

```python
from prompts.prompt_loader import load_screening_prompts, load_extraction_prompts

# Load all screening prompts
screening_prompts = load_screening_prompts()
reviewer_system = screening_prompts["reviewer"]["system_prompt"]
reviewer_user_template = screening_prompts["reviewer"]["user_prompt_template"]

# Load all extraction prompts
extraction_prompts = load_extraction_prompts()
quant_prompt = extraction_prompts["quantitative_extraction"]
```

## Best Practices

1. **Keep original copies**: Before making changes, save a backup of the original prompts
2. **Test incrementally**: Test changes with a small subset of papers before running full batches
3. **Document changes**: Add comments in your custom prompts to explain modifications
4. **Maintain format**: Keep the variable placeholders (e.g., `{title}`) intact for proper substitution
5. **Preserve JSON structure**: Ensure output format specifications remain valid JSON

## Examples

### Example 1: Adding a New Extraction Field

To extract a new field (e.g., "funding_source"), edit `qualitative_extraction_prompt.txt`:

```
15. funding_source: Extract any mention of funding sources or grants; if none, return "NA".
```

Then add it to the JSON output structure:

```json
{
    ...
    "funding_source": "<funding information or 'NA'>",
    ...
}
```

### Example 2: Modifying Inclusion Criteria

To change screening criteria, edit `reviewer_system_prompt.txt`:

```
SCREENING CRITERIA:
1. SPECIALIZATION: Paper must report quantitative climatic specialization metrics
2. CLIMATE CHANGE: Climate change must be quantified with actual data
3. RELATIONSHIP: Relationship between climate and specialization must be tested
4. QUANTITATIVE DATA: Sufficient data to extract effect sizes
5. YOUR_NEW_CRITERION: Description of your new criterion
6. QUALITY: Appropriate study design
```

## Troubleshooting

- **Prompts not loading**: Check file paths and ensure files are in UTF-8 encoding
- **Variables not substituting**: Verify that variable names in templates match those in the code
- **JSON parsing errors**: Ensure JSON structure examples in prompts are valid
- **Unexpected agent behavior**: Review system prompts for clarity and consistency

## Contributing

If you develop improved prompts for specific research domains, consider sharing them with the community!

---

**Last Updated**: 2025-11-09
**Version**: 1.0.0
