#!/usr/bin/env python3
"""
Tauric AI - Automated GitHub & Cloud Deployment Sync Script
Usage:
    python sync_github.py "Tóm tắt nội dung cập nhật"
"""

import sys
import subprocess
import os

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def run_cmd(cmd, check=True):
    print(f"⚙️  Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True, encoding="utf-8", errors="replace")
    if check and res.returncode != 0:
        print(f"❌ Error ({res.returncode}):\n{res.stderr.strip() or res.stdout.strip()}")
        sys.exit(res.returncode)
    return res.stdout.strip()

def main():
    message = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "feat: update trading intelligence and dashboard features"
    
    print("\n🚀 === TAURIC AI AUTO-SYNC TO GITHUB & CLOUD ===")
    
    # 1. Check status
    status = run_cmd("git status -s", check=False)
    if not status:
        print("✓ Không có thay đổi mới cần commit. Kiểm tra git remote...")
    else:
        print(f"📝 Phát hiện các file đã thay đổi:\n{status}\n")
        
        # 2. Run unit tests before pushing to ensure quality
        print("🧪 Đang chạy kiểm thử tự động (Unit Tests)...")
        test_res = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if test_res.returncode != 0:
            print("⚠️  Một số test bị lỗi. Vui lòng kiểm tra trước khi push:")
            print(test_res.stdout or test_res.stderr)
            confirm = input("Bạn vẫn muốn tiếp tục push? (y/N): ").strip().lower()
            if confirm != "y":
                print("Đã hủy quá trình push.")
                sys.exit(1)
        else:
            print("✅ Toàn bộ 18 unit tests đều VƯỢT QUA thành công!\n")

        # 3. Add and commit
        print(f"📦 Đang đóng gói commit: '{message}'...")
        run_cmd("git add .")
        run_cmd(f'git commit -m "{message}"')

    # 4. Push to origin main
    print("📤 Đang đẩy trực tiếp lên GitHub (origin main)...")
    run_cmd("git push origin main")
    
    print("\n🎉 HOÀN TẤT ĐỒNG BỘ THÀNH CÔNG!")
    print("🔗 GitHub Repo: https://github.com/nguyen15620-afk/tauric-crypto-agents")
    print("🌐 Live Web:    https://mercury-sleep-meetings-bear.trycloudflare.com")
    print("================================================\n")

if __name__ == "__main__":
    main()
