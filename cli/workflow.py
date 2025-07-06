"""Workflow command group: workflow management"""
from tabulate import tabulate
from colorama import Fore, Style
import os
import json
from agents.workflow_orchestrator import WorkflowOrchestrator
import asyncio

TEMPLATE_DIR = "config/workflow_templates"

# Track running workflows in memory for CLI
_running_workflows = {}


def handle(args):
    if not args or args[0] == "list":
        # List available workflow templates
        templates = [f for f in os.listdir(TEMPLATE_DIR) if f.endswith('.json')]
        table = [[t] for t in templates]
        print(tabulate(table, headers=["Workflow Template"]))
    elif args[0] == "run":
        if len(args) < 2:
            print(Fore.RED + "Usage: workflow run <template>" + Style.RESET_ALL)
            return
        template_file = os.path.join(TEMPLATE_DIR, args[1])
        if not os.path.exists(template_file):
            print(Fore.RED + f"Template '{args[1]}' not found." + Style.RESET_ALL)
            return
        with open(template_file, "r", encoding="utf-8") as f:
            template = json.load(f)
        orchestrator = WorkflowOrchestrator()
        # Simulate user input from template
        user_input = template.get("description", "Run workflow")
        print(Fore.YELLOW + f"Running workflow: {args[1]}..." + Style.RESET_ALL)
        result = asyncio.run(orchestrator.execute_workflow(user_input))
        workflow_id = result.get("workflow_id", "unknown")
        _running_workflows[workflow_id] = result
        print(Fore.GREEN + f"Workflow '{args[1]}' completed. Workflow ID: {workflow_id}" + Style.RESET_ALL)
        print(json.dumps(result, indent=2))
    elif args[0] == "status":
        if len(args) < 2:
            print(Fore.RED + "Usage: workflow status <workflow_id>" + Style.RESET_ALL)
            return
        workflow_id = args[1]
        result = _running_workflows.get(workflow_id)
        if not result:
            print(Fore.RED + f"Workflow '{workflow_id}' not found or not run in this session." + Style.RESET_ALL)
            return
        print(Fore.GREEN + f"Status for workflow {workflow_id}:" + Style.RESET_ALL)
        print(json.dumps(result, indent=2))
    else:
        print(Fore.RED + f"Unknown subcommand: {' '.join(args)}" + Style.RESET_ALL)
        help()

def help():
    print("""
Usage: workflow <subcommand>
Subcommands:
  list                        List available workflow templates
  run <template>              Run a workflow from a template
  status <workflow_id>        Show status/result of a workflow (in this session)
""") 