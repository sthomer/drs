import argparse
import pandas as pd

A_RAN_E2 = (77.9, 87.3)

def run_script(filename):
    path = f"./data/output/{filename}.csv"
    df = pd.read_csv(path)

    df = df[df["frequency"] > 0]
    df.to_csv(f"./data/output/{filename}_positive_freq.csv", index=False)
    print(f"Done filtering positive frequencies to {filename}_positive_freq.csv!")

    df_focused =df[df["frequency"].between(*A_RAN_E2)]
    df_focused.to_csv(f"./data/output/{filename}_focused.csv", index=False)
    print(f"Done filtering focused frequencies to {filename}_focused.csv!")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simple analysis", prog="drs_analysis", epilog="By Bach Tran")
    ap.add_argument('-f', '--filename', type=str, default="temp")
    args = ap.parse_args()

    run_script(filename=args.filename)
