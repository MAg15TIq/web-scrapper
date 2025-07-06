"""Monitoring command group: system health, logs, metrics"""
from tabulate import tabulate
from colorama import Fore, Style
from agents.monitoring import MonitoringAgent

# Use a singleton MonitoringAgent for CLI
_monitoring_agent = None
def get_monitoring_agent():
    global _monitoring_agent
    if _monitoring_agent is None:
        _monitoring_agent = MonitoringAgent()
    return _monitoring_agent

def handle(args):
    agent = get_monitoring_agent()
    if not args or args[0] == "system_health":
        # Show system health metrics
        import asyncio
        metrics = asyncio.run(agent._get_system_metrics())
        table = [[k, v] for k, v in metrics.items()]
        print(tabulate(table, headers=["Metric", "Value"]))
    elif args[0] == "logs":
        # Show recent alerts
        alerts = agent.get_alert_history(limit=20)
        table = []
        for a in alerts:
            color = (
                Fore.RED if a["level"] in ("error", "critical")
                else Fore.YELLOW if a["level"] == "warning"
                else Fore.GREEN
            )
            table.append([
                color + a["title"] + Style.RESET_ALL,
                a["level"],
                a["message"],
                a["timestamp"]
            ])
        print(tabulate(table, headers=["Title", "Level", "Message", "Timestamp"]))
    elif args[0] == "metrics":
        if len(args) < 2:
            print(Fore.RED + "Usage: monitoring metrics <type>" + Style.RESET_ALL)
            return
        metric_type = args[1]
        metrics = agent.get_metrics(metric_type)
        if not metrics:
            print(Fore.YELLOW + f"No metrics found for type '{metric_type}'." + Style.RESET_ALL)
            return
        table = []
        for m in metrics[-20:]:
            table.append([
                m["timestamp"],
                m["value"],
                m.get("tags", {})
            ])
        print(tabulate(table, headers=["Timestamp", "Value", "Tags"]))
    else:
        print(Fore.RED + f"Unknown subcommand: {' '.join(args)}" + Style.RESET_ALL)
        help()

def help():
    print("""
Usage: monitoring <subcommand>
Subcommands:
  system_health         Show system health metrics
  logs                 Show recent system alerts
  metrics <type>       Show metrics of a specific type
""") 