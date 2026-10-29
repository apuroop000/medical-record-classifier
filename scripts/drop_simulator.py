import argparse
import os
import shutil
import time

DEFAULT_DROPZONE = os.path.join("data", "incoming_dropzone")


def simulate(source_dir: str, dropzone_dir: str, interval: int):
    if not os.path.isdir(source_dir):
        print(f"❌ Source folder not found: {source_dir}")
        print("   Create it and put a few sample PDF/JPG/DOCX files inside first.")
        return

    os.makedirs(dropzone_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f)) and not f.startswith(".")
    )

    if not files:
        print(f"❌ No files found in {source_dir}")
        return

    print(f"🎬 Simulating {len(files)} incoming document(s), one every {interval}s.")
    print(f"   Source: {os.path.abspath(source_dir)}")
    print(f"   Target dropzone: {os.path.abspath(dropzone_dir)}\n")

    for i, fname in enumerate(files, 1):
        src_path = os.path.join(source_dir, fname)
        dst_path = os.path.join(dropzone_dir, fname)
        shutil.copy2(src_path, dst_path)
        print(f"[{i}/{len(files)}] 📨 Simulated arrival: {fname}  (dropped at {time.strftime('%H:%M:%S')})")

        if i < len(files):
            time.sleep(interval)

    print("\n✅ Simulation complete. Check audit_log.xlsx and the dashboard's Audit Log tab.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate documents arriving from a scanner/email.")
    parser.add_argument("--source", default="sample_files", help="Folder of sample files to simulate arriving")
    parser.add_argument("--dropzone", default=DEFAULT_DROPZONE, help="Target dropzone folder")
    parser.add_argument("--interval", type=int, default=10, help="Seconds between simulated arrivals")
    args = parser.parse_args()

    simulate(args.source, args.dropzone, args.interval)
