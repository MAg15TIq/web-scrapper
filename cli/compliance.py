"""Compliance command group: GDPR, audit logs"""
from tabulate import tabulate
from colorama import Fore, Style
from agents.gdpr_compliance import GDPRComplianceManager
import time

manager = GDPRComplianceManager()

def handle(args):
    if not args or args[0] == "gdpr":
        # Show GDPR compliance report for the last 30 days
        end = time.time()
        start = end - 30 * 24 * 3600
        report = manager.generate_compliance_report(start, end)
        table = [[k, v] for k, v in report.items()]
        print(tabulate(table, headers=["Metric", "Value"]))
    elif args[0] == "audit_logs":
        # Show audit logs (stub: show compliance records for now)
        if len(args) < 2:
            print(Fore.YELLOW + "Usage: compliance audit_logs <data_subject_id>" + Style.RESET_ALL)
            return
        records = manager._get_processing_records_for_subject(args[1])
        if not records:
            print(Fore.RED + f"No audit logs for subject {args[1]}" + Style.RESET_ALL)
            return
        table = [[r.get('timestamp', '-'), r.get('action', '-'), r.get('details', '-')]
                 for r in records]
        print(tabulate(table, headers=["Timestamp", "Action", "Details"]))
    else:
        print(Fore.RED + f"Unknown subcommand: {' '.join(args)}" + Style.RESET_ALL)
        help()

def help():
    print("""
Usage: compliance <subcommand>
Subcommands:
  gdpr                        Show GDPR compliance report (last 30 days)
  audit_logs <data_subject_id>  Show audit logs for a data subject
""") 