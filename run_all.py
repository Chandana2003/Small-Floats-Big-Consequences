#!/usr/bin/env python3
import sys
import subprocess

def main():
    # use the current interpreter instead of hardcoding "python"
    py = sys.executable
    subprocess.check_call([py, '-m', 'src.main'])
    subprocess.check_call([py, '-m', 'src.make_plots'])
    print("Done. See results/ (CSVs) and plots/ (PNGs).")

if __name__ == '__main__':
    main()
