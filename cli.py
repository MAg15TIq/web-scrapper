from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter

# Import command modules
from cli import agents, jobs, monitoring, plugins, config, users, data, workflow, compliance
# Add import for nlpjobs
from cli import nlpjobs

COMMAND_MODULES = {
    'agents': agents,
    'jobs': jobs,
    'monitoring': monitoring,
    'plugins': plugins,
    'config': config,
    'users': users,
    'data': data,
    'workflow': workflow,
    'compliance': compliance,
    'nlpjobs': nlpjobs,  # Register the new NLP jobs command
}

COMMANDS = list(COMMAND_MODULES.keys()) + ['help', 'exit', 'quit']

# Simple completer for top-level commands
command_completer = WordCompleter(COMMANDS, ignore_case=True)

def print_help():
    print("""
Available commands:
  agents      Manage agents (list, status, health, start, stop, details)
  jobs        Manage jobs (queue, running, completed, failed, details, cancel)
  monitoring  System health, logs, metrics
  plugins     Manage plugins (list, enable, disable, install)
  config      View/edit/backup/restore configuration
  users       User authentication and management
  data        Data export/import/clean/validate
  workflow    Workflow management
  compliance  Compliance and audit logs
  nlpjobs     Submit natural language scraping jobs
  help        Show this help message
  exit/quit   Exit the CLI
""")

def main():
    session = PromptSession('web-scrapper> ', completer=command_completer)
    print("Welcome to the Web Scrapper CLI! Type 'help' to see available commands.")
    while True:
        try:
            user_input = session.prompt()
            if not user_input.strip():
                continue
            cmd, *args = user_input.strip().split()
            if cmd in ('exit', 'quit'):
                print("Exiting CLI. Goodbye!")
                break
            elif cmd == 'help':
                print_help()
            elif cmd in COMMAND_MODULES:
                # If user types '<command> help', show that command's help
                if args and args[0] == 'help':
                    COMMAND_MODULES[cmd].help()
                else:
                    COMMAND_MODULES[cmd].handle(args)
            else:
                print(f"Unknown command: {cmd}. Type 'help' for a list of commands.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting CLI. Goodbye!")
            break

if __name__ == '__main__':
    main() 