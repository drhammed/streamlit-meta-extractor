# Meta-Analysis Data Extractor

A Streamlit app for automated paper screening and data extraction for ecological meta-analyses. Uses LLM-powered dual-extractor pipelines with automatic conflict resolution.

## Features

- **Paper Screening**: Multi-agent system (2 reviewers + adjudicator) evaluates papers against inclusion criteria
- **Data Extraction**: Dual independent extractors with automatic reconciliation for qualitative and quantitative data
- **Batch Processing**: Process multiple papers with progress tracking
- **Multiple LLM Providers**: Supports Anthropic (Claude), OpenAI (GPT-4o), and Ollama Cloud
- **Vision Support**: Optional figure/table extraction from PDFs

## Usage

1. Select your LLM provider and enter your API key in the sidebar
2. Upload a PDF or select from a directory
3. Choose screening, extraction, or batch processing
4. Download results as JSON or CSV

## Supported Providers

| Provider | Models | API Key Required |
|----------|--------|------------------|
| Anthropic | claude-3-5-haiku, claude-3-5-sonnet | Yes |
| OpenAI | gpt-4o | Yes |
| Ollama Cloud | llama3.2, llama3.2-vision | Yes |

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

This app is designed for Streamlit Cloud. Add your API keys in Settings > Secrets.

## Author

Built by Hammed Akande
