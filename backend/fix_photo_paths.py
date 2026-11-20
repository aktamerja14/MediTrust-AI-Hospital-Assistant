"""Script to fix doctor photo_path from .jpg to .svg in MongoDB."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.database.connection import db_manager


async def fix_photo_paths():
    """Fix photo_path from .jpg to .svg for doctors where SVG file exists."""
    await db_manager.connect()
    db = db_manager.db

    # Get all doctors
    doctors = await db["doctors"].find().to_list(length=100)

    print("Current doctor records:")
    for doc in doctors:
        print(f"  {doc['doctor_id']}: {doc['photo_path']}")

    # Check which records need updating
    photo_dir = Path(__file__).parent / "doctor_photos"
    updates = []

    for doc in doctors:
        current_path = doc["photo_path"]
        if current_path.endswith(".jpg"):
            # Check if corresponding .svg exists
            svg_name = Path(current_path).stem + ".svg"
            svg_path = photo_dir / svg_name
            if svg_path.exists():
                new_path = f"doctor_photos/{svg_name}"
                updates.append({
                    "doctor_id": doc["doctor_id"],
                    "old_path": current_path,
                    "new_path": new_path
                })
            else:
                print(f"WARNING: No SVG found for {doc['doctor_id']}: {svg_path}")

    print("\nProposed updates:")
    for u in updates:
        print(f"  {u['doctor_id']}: {u['old_path']} -> {u['new_path']}")

    if not updates:
        print("\nNo updates needed.")
        await db_manager.disconnect()
        return

    # Ask for confirmation (but we'll auto-confirm for this script)
    print("\nApplying updates...")
    for u in updates:
        result = await db["doctors"].update_one(
            {"doctor_id": u["doctor_id"]},
            {"$set": {"photo_path": u["new_path"]}}
        )
        if result.modified_count == 1:
            print(f"  Updated: {u['doctor_id']}")
        else:
            print(f"  No change (already correct): {u['doctor_id']}")

    # Verify
    print("\nVerification after update:")
    doctors = await db["doctors"].find().to_list(length=100)
    for doc in doctors:
        print(f"  {doc['doctor_id']}: {doc['photo_path']}")

    await db_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(fix_photo_paths())