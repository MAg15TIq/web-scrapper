"""
NLPJobs command group: submit natural language scraping jobs
"""
import json
import os
import re
from colorama import Fore, Style
from cli import workflow
from agents.nlp_parser import parse_nlp_job  # Import the new modular parser

TEMPLATE_PATH = "config/workflow_templates/basic_scraping.json"
TEMP_JOB_PATH = "temp/nlpjob_temp.json"

def handle(args):
    if not args or args[0] in ('help', '-h', '--help'):
        help()
        return
    nl_input = ' '.join(args)
    params = parse_nlp_job(nl_input)  # Use the new parser
    if not params['url']:
        print(Fore.RED + 'Could not find a URL in your request.' + Style.RESET_ALL)
        return
    # Load template
    with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        template = json.load(f)
    # Fill template
    template['parameters']['url'] = params['url']
    template['parameters']['selectors'] = params['selectors']
    template['parameters']['output_path'] = params['output_path']
    # Save temp job file
    os.makedirs(os.path.dirname(TEMP_JOB_PATH), exist_ok=True)
    with open(TEMP_JOB_PATH, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2)
    print(Fore.YELLOW + f"Submitting NLP job for URL: {params['url']} (output: {params['output_path']})" + Style.RESET_ALL)
    # Run workflow
    workflow.handle(['run', TEMP_JOB_PATH])

def help():
    print("""
Usage: nlpjobs <natural language request>
Example:
  nlpjobs "Scrape all product names and prices from https://example.com/shop, save as CSV, first 5 pages."
This will extract the URL, fields, output format, and number of pages, and submit a scraping job.
""") 