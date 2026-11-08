"""CLI command to create an admin account."""
import argparse
import asyncio
import getpass
import sys
from typing import Optional

from app.core.config import settings
from app.core.security.password import hash_password
from app.database.connection import db_manager
from app.schemas.auth import UserCreate
from app.services.auth import AuthService


async def create_admin(
    name: str,
    email: str,
    password: str,
) -> None:
    """
    Create an admin account using existing AuthService.

    Args:
        name: Admin's full name.
        email: Admin's email address.
        password: Admin's password (will be hashed).

    Raises:
        ValueError: If email already exists or validation fails.
        Exception: For database connection or other errors.
    """
    await db_manager.connect()

    try:
        db = db_manager.db
        if db is None:
            raise RuntimeError("Database not initialized after connection")

        auth_service = AuthService(db)

        user_data = UserCreate(email=email, password=password)

        user = await auth_service.create_user(user_data, role="admin")

        print(f"\n[OK] Admin account created successfully!")
        print(f"  Name:  {name}")
        print(f"  Email: {user.email}")
        print(f"  Role:  {user.role}")
        print(f"  ID:    {user.id}")
        print(f"  Created at: {user.created_at}")

    finally:
        await db_manager.disconnect()


async def reset_admin_password(
    email: str,
    new_password: str,
) -> None:
    """
    Reset an existing admin account's password.

    Args:
        email: Admin's email address.
        new_password: New password (will be hashed).

    Raises:
        ValueError: If account not found, not an admin, or validation fails.
        Exception: For database connection or other errors.
    """
    await db_manager.connect()

    try:
        db = db_manager.db
        if db is None:
            raise RuntimeError("Database not initialized after connection")

        # Find the admin account
        user_doc = await db.users.find_one({"email": email.lower(), "role": "admin"})
        if not user_doc:
            # Check if email exists but is not admin
            existing = await db.users.find_one({"email": email.lower()})
            if existing:
                raise ValueError("Account exists but is not an admin account")
            raise ValueError("Admin account not found")

        # Hash new password
        hashed_password = hash_password(new_password)

        # Update only the hashed_password field
        result = await db.users.update_one(
            {"_id": user_doc["_id"]},
            {"$set": {"hashed_password": hashed_password}}
        )

        if result.modified_count == 0:
            raise RuntimeError("Failed to update password")

        print(f"\n[OK] Admin password reset successfully!")
        print(f"  Email: {email}")

    finally:
        await db_manager.disconnect()


def get_input(prompt: str, required: bool = True) -> Optional[str]:
    """Get user input with validation."""
    while True:
        value = input(prompt).strip()
        if value or not required:
            return value if value else None
        print("  This field is required. Please try again.")


def get_password(prompt: str = "Password: ", confirm: bool = True) -> str:
    """Get password securely with optional confirmation."""
    while True:
        password = getpass.getpass(prompt)
        if not password:
            print("  Password cannot be empty. Please try again.")
            continue

        if confirm:
            confirm_password = getpass.getpass("Confirm password: ")
            if password != confirm_password:
                print("  Passwords do not match. Please try again.")
                continue

        return password


def validate_email(email: str) -> bool:
    """Basic email validation."""
    return "@" in email and "." in email.split("@")[-1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Admin account management for MediTrust",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Create subcommand
    create_parser = subparsers.add_parser(
        "create",
        help="Create a new admin account",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.cli.admin create
  python -m app.cli.admin create --email admin@example.com --name "Admin User"
  python -m app.cli.admin create --email admin@example.com --name "Admin User" --password "securePass123"

Note: If --password is not provided, you will be prompted to enter it securely.
        """,
    )
    create_parser.add_argument(
        "--name",
        "-n",
        help="Admin's full name",
        type=str,
    )
    create_parser.add_argument(
        "--email",
        "-e",
        help="Admin's email address",
        type=str,
    )
    create_parser.add_argument(
        "--password",
        "-p",
        help="Admin's password (min 8 characters). If not provided, will prompt securely.",
        type=str,
    )
    create_parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip password confirmation prompt",
    )

    # Reset-password subcommand
    reset_parser = subparsers.add_parser(
        "reset-password",
        help="Reset an existing admin account's password",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.cli.admin reset-password --email admin@example.com
  python -m app.cli.admin reset-password --email admin@example.com --password "newSecurePass123"

Note: If --password is not provided, you will be prompted to enter it securely.
        """,
    )
    reset_parser.add_argument(
        "--email",
        "-e",
        help="Admin's email address (required)",
        type=str,
        required=True,
    )
    reset_parser.add_argument(
        "--password",
        "-p",
        help="New password (min 8 characters). If not provided, will prompt securely.",
        type=str,
    )
    reset_parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip password confirmation prompt",
    )

    return parser.parse_args()


async def main() -> int:
    """Main CLI entry point."""
    args = parse_args()

    if args.command == "create":
        return await handle_create(args)
    elif args.command == "reset-password":
        return await handle_reset_password(args)
    else:
        print(f"[ERROR] Unknown command: {args.command}")
        return 1


async def handle_create(args: argparse.Namespace) -> int:
    """Handle the create admin command."""
    print("\n============================================================")
    print("       MediTrust Admin Account Creation CLI")
    print("============================================================\n")

    # Get name
    name = args.name
    if not name:
        name = get_input("Admin name: ")
    print(f"  Name: {name}")

    # Get email
    email = args.email
    if not email:
        while True:
            email = get_input("Admin email: ")
            if validate_email(email):
                break
            print("  Invalid email format. Please try again.")
    print(f"  Email: {email}")

    # Get password
    password = args.password
    if not password:
        password = get_password(confirm=not args.no_confirm)
    else:
        if len(password) < 8:
            print("[ERROR] Error: Password must be at least 8 characters long.")
            return 1
        if args.no_confirm is False:
            # If password provided via CLI, still confirm unless --no-confirm
            confirm = getpass.getpass("Confirm password: ")
            if password != confirm:
                print("[ERROR] Error: Passwords do not match.")
                return 1

    print("\nCreating admin account...")

    try:
        await create_admin(name=name, email=email, password=password)
        return 0
    except ValueError as exc:
        print(f"\n[ERROR] Error: {exc}")
        return 1
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error: {exc}")
        return 1


async def handle_reset_password(args: argparse.Namespace) -> int:
    """Handle the reset-password command."""
    print("\n============================================================")
    print("       MediTrust Admin Password Reset CLI")
    print("============================================================\n")

    email = args.email
    if not validate_email(email):
        print("[ERROR] Error: Invalid email format.")
        return 1
    print(f"  Email: {email}")

    # Get new password
    password = args.password
    if not password:
        password = get_password(confirm=not args.no_confirm)
    else:
        if len(password) < 8:
            print("[ERROR] Error: Password must be at least 8 characters long.")
            return 1
        if args.no_confirm is False:
            confirm = getpass.getpass("Confirm password: ")
            if password != confirm:
                print("[ERROR] Error: Passwords do not match.")
                return 1

    print("\nResetting admin password...")

    try:
        await reset_admin_password(email=email, new_password=password)
        return 0
    except ValueError as exc:
        print(f"\n[ERROR] Error: {exc}")
        return 1
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))