"""MediTrust CLI main entry point."""
import argparse
import sys

from app.cli.admin import main as admin_main


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MediTrust CLI - Administrative tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Admin subcommand
    admin_parser = subparsers.add_parser(
        "admin",
        help="Admin account management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    admin_subparsers = admin_parser.add_subparsers(dest="admin_command", help="Admin commands")

    # Create admin subcommand
    create_parser = admin_subparsers.add_parser(
        "create",
        help="Create a new admin account",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    create_parser.add_argument("--name", "-n", help="Admin's full name")
    create_parser.add_argument("--email", "-e", help="Admin's email address")
    create_parser.add_argument("--password", "-p", help="Admin's password")
    create_parser.add_argument("--no-confirm", action="store_true", help="Skip password confirmation")

    args = parser.parse_args()

    if args.command == "admin" and args.admin_command == "create":
        # Forward to admin create main
        sys.argv = ["admin-create"] + sys.argv[3:]
        return admin_main()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())