import csv, os

def write_row(path, header, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new: w.writerow(header)
        w.writerow(row)
