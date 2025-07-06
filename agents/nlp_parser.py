"""
NLP Parser for extracting scraping job parameters from natural language input.
"""
import re
from typing import Dict


def parse_nlp_job(nl_input: str) -> Dict:
    """
    Improved rule-based parser for demo purposes.
    Extracts URL, selectors, output format, and number of pages from the input string.
    """
    # Extract URL
    url_match = re.search(r'(https?://\S+)', nl_input)
    url = url_match.group(1) if url_match else None

    # Extract output format
    fmt = 'json'
    if 'csv' in nl_input.lower():
        fmt = 'csv'
    elif 'json' in nl_input.lower():
        fmt = 'json'

    # Extract number of pages
    pages = 1
    pages_match = re.search(r'(first|next)?\s*(\d+)\s*pages?', nl_input.lower())
    if pages_match:
        pages = int(pages_match.group(2))

    # Improved field/selector extraction
    # Look for patterns like 'all X and Y', 'X, Y, and Z', or 'X and Y'
    selectors = {}
    # Remove URL and output format from input for cleaner field extraction
    clean_input = re.sub(r'(https?://\S+)', '', nl_input)
    clean_input = re.sub(r'(save|as|to|in|output|csv|json|first|next|pages?)', '', clean_input, flags=re.IGNORECASE)
    # Find fields after 'all', 'scrape', or at the start
    field_match = re.search(r'(all|scrape)?\s*([\w\s,]+?)(?: from| at| on|$)', clean_input.lower())
    if field_match:
        fields = field_match.group(2)
        for field in re.split(r',| and ', fields):
            field = field.strip()
            if field and field not in ['products', 'product', 'scrape', 'save', 'as', 'to', 'from', 'the', 'first', 'next', 'output']:
                selectors[field.replace(' ', '_')] = f'.{field.replace(" ", "-")}'

    # Output path
    output_path = f'output/nlpjob_output.{fmt}'

    return {
        'url': url or '',
        'selectors': selectors or {'data': 'body'},
        'output_path': output_path,
        'format': fmt,
        'pages': pages
    } 