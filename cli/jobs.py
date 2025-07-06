"""Jobs command group: manage jobs (queue, running, completed, failed, details, cancel)"""
from tabulate import tabulate
from colorama import Fore, Style
from agents.coordinator import CoordinatorAgent
from models.task import TaskStatus

# Use the same coordinator as agents.py
from cli.agents import coordinator, ensure_bootstrapped

def handle(args):
    ensure_bootstrapped()
    if not args or args[0] == "queue":
        # Show pending/assigned jobs
        table = []
        for task in coordinator.all_tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED):
                color = Fore.YELLOW if task.status == TaskStatus.ASSIGNED else Fore.CYAN
                table.append([
                    color + task.id + Style.RESET_ALL,
                    task.type,
                    task.status,
                    task.assigned_to or "-",
                    task.created_at,
                    task.updated_at
                ])
        print(tabulate(table, headers=["Job ID", "Type", "Status", "Assigned To", "Created", "Updated"]))
    elif args[0] == "running":
        table = []
        for task in coordinator.all_tasks.values():
            if task.status == TaskStatus.RUNNING:
                table.append([
                    Fore.YELLOW + task.id + Style.RESET_ALL,
                    task.type,
                    task.status,
                    task.assigned_to or "-",
                    task.created_at,
                    task.updated_at
                ])
        print(tabulate(table, headers=["Job ID", "Type", "Status", "Assigned To", "Created", "Updated"]))
    elif args[0] == "completed":
        table = []
        for task in coordinator.all_tasks.values():
            if task.status == TaskStatus.COMPLETED:
                table.append([
                    Fore.GREEN + task.id + Style.RESET_ALL,
                    task.type,
                    task.status,
                    task.assigned_to or "-",
                    task.created_at,
                    task.updated_at
                ])
        print(tabulate(table, headers=["Job ID", "Type", "Status", "Assigned To", "Created", "Updated"]))
    elif args[0] == "failed":
        table = []
        for task in coordinator.all_tasks.values():
            if task.status == TaskStatus.FAILED:
                table.append([
                    Fore.RED + task.id + Style.RESET_ALL,
                    task.type,
                    task.status,
                    task.assigned_to or "-",
                    task.created_at,
                    task.updated_at
                ])
        print(tabulate(table, headers=["Job ID", "Type", "Status", "Assigned To", "Created", "Updated"]))
    elif args[0] == "details":
        if len(args) < 2:
            print(Fore.RED + "Usage: jobs details <job_id>" + Style.RESET_ALL)
            return
        job_id = args[1]
        info = coordinator.get_task_status(job_id)
        if not info:
            print(Fore.RED + f"Job '{job_id}' not found." + Style.RESET_ALL)
            return
        for k, v in info.items():
            print(f"{Fore.CYAN}{k}{Style.RESET_ALL}: {v}")
    elif args[0] == "cancel":
        if len(args) < 2:
            print(Fore.RED + "Usage: jobs cancel <job_id>" + Style.RESET_ALL)
            return
        job_id = args[1]
        task = coordinator.all_tasks.get(job_id)
        if not task:
            print(Fore.RED + f"Job '{job_id}' not found." + Style.RESET_ALL)
            return
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED):
            print(Fore.YELLOW + f"Job '{job_id}' is already {task.status}." + Style.RESET_ALL)
            return
        task.status = TaskStatus.CANCELED
        print(Fore.GREEN + f"Job '{job_id}' canceled." + Style.RESET_ALL)
    else:
        print(Fore.RED + f"Unknown subcommand: {' '.join(args)}" + Style.RESET_ALL)
        help()

def help():
    print("""
Usage: jobs <subcommand>
Subcommands:
  queue                Show pending/assigned jobs
  running              Show running jobs
  completed            Show completed jobs
  failed               Show failed jobs
  details <job_id>     Show job details
  cancel <job_id>      Cancel a job
""") 