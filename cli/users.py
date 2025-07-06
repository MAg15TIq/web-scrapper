"""Users command group: user authentication and management"""
from tabulate import tabulate
from colorama import Fore, Style
from auth.unified_auth import get_unified_auth_manager
import getpass

manager = get_unified_auth_manager()

# Track current session in memory for CLI
_current_session = None
_current_user = None

def handle(args):
    global _current_session, _current_user
    if not args or args[0] == "list":
        # List all users
        users = list(manager._users.values())
        table = []
        for u in users:
            color = Fore.GREEN if u.is_active else Fore.RED
            table.append([
                color + u.username + Style.RESET_ALL,
                u.email or "-",
                u.role,
                "Yes" if u.is_admin else "No",
                u.created_at,
                u.last_login or "-"
            ])
        print(tabulate(table, headers=["Username", "Email", "Role", "Admin", "Created", "Last Login"]))
    elif args[0] == "login":
        username = input("Username: ")
        password = getpass.getpass("Password: ")
        user = manager.authenticate_user(username, password)
        if not user:
            print(Fore.RED + "Login failed." + Style.RESET_ALL)
            return
        session = manager.create_session(user, client_info={"client": "cli"})
        _current_session = session
        _current_user = user
        print(Fore.GREEN + f"Login successful. Welcome, {user.username}!" + Style.RESET_ALL)
    elif args[0] == "logout":
        if not _current_session:
            print(Fore.YELLOW + "No active session." + Style.RESET_ALL)
            return
        manager._revoke_session(_current_session.id)
        print(Fore.GREEN + "Logged out successfully." + Style.RESET_ALL)
        _current_session = None
        _current_user = None
    else:
        print(Fore.RED + f"Unknown subcommand: {' '.join(args)}" + Style.RESET_ALL)
        help()

def help():
    print("""
Usage: users <subcommand>
Subcommands:
  list           List all users
  login          Log in a user
  logout         Log out a user
""") 