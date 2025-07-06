"""Agents command group: manage agents (list, status, health, start, stop, details)"""
import importlib
import pkgutil
from tabulate import tabulate
from colorama import Fore, Style
from agents.coordinator import CoordinatorAgent
from agents.base import Agent

# Dynamically discover all agent classes in the agents/ directory
import agents

def discover_agent_classes():
    agent_classes = []
    for _, module_name, ispkg in pkgutil.iter_modules(agents.__path__):
        if ispkg or module_name in ("base", "coordinator", "__init__"):  # skip base, coordinator, init
            continue
        try:
            module = importlib.import_module(f"agents.{module_name}")
            for attr in dir(module):
                obj = getattr(module, attr)
                if isinstance(obj, type) and issubclass(obj, Agent) and obj is not Agent:
                    agent_classes.append((module_name, obj))
        except Exception:
            continue
    return agent_classes

# Instantiate one of each agent type and register with coordinator
coordinator = CoordinatorAgent()
def bootstrap_agents():
    agent_classes = discover_agent_classes()
    for module_name, cls in agent_classes:
        try:
            agent = cls()
            coordinator.register_agent(agent)
        except Exception:
            continue

# Only bootstrap once
_bootstrapped = False
def ensure_bootstrapped():
    global _bootstrapped
    if not _bootstrapped:
        bootstrap_agents()
        _bootstrapped = True

def handle(args):
    ensure_bootstrapped()
    if not args or args[0] == "list":
        # List all agents
        table = []
        for agent_id, info in coordinator.agents.items():
            color = Fore.GREEN if info["status"] == "idle" else Fore.YELLOW if info["status"] == "busy" else Fore.RED
            table.append([
                color + agent_id + Style.RESET_ALL,
                info["type"],
                info["status"],
                info["tasks"],
                info.get("last_seen", "")
            ])
        print(tabulate(table, headers=["Agent ID", "Type", "Status", "Tasks", "Last Seen"]))
    elif args[0] == "status":
        # Show status for each agent
        table = []
        for agent_id, info in coordinator.agents.items():
            color = Fore.GREEN if info["status"] == "idle" else Fore.YELLOW if info["status"] == "busy" else Fore.RED
            table.append([
                color + agent_id + Style.RESET_ALL,
                info["type"],
                info["status"],
                info["tasks"]
            ])
        print(tabulate(table, headers=["Agent ID", "Type", "Status", "Tasks"]))
    elif args[0] == "details":
        if len(args) < 2:
            print(Fore.RED + "Usage: agents details <agent_id>" + Style.RESET_ALL)
            return
        agent_id = args[1]
        info = coordinator.get_agent_status(agent_id)
        if not info:
            print(Fore.RED + f"Agent '{agent_id}' not found." + Style.RESET_ALL)
            return
        for k, v in info.items():
            print(f"{Fore.CYAN}{k}{Style.RESET_ALL}: {v}")
    else:
        print(Fore.RED + f"Unknown subcommand: {' '.join(args)}" + Style.RESET_ALL)
        help()

def help():
    print("""
Usage: agents <subcommand>
Subcommands:
  list                List all agents
  status              Show agent status
  details <agent_id>  Show agent details
""") 