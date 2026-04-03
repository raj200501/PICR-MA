def to_csv_gz(df_rows, path):
    # stdlib fallback writer; accepts list of dicts
    import csv
    if not df_rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(df_rows[0].keys()))
        w.writeheader()
        w.writerows(df_rows)
