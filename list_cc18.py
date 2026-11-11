#!/usr/bin/env python
"""Write the OpenML-CC18 task list (task_id, dataset name, size) to a TSV, smallest first."""
import argparse

import openml
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="cc18_tasks.tsv")
    p.add_argument("--suite", default="OpenML-CC18")
    a = p.parse_args()
    suite = openml.study.get_suite(a.suite)
    tasks = openml.tasks.list_tasks(task_id=list(suite.tasks), output_format="dataframe")
    cols = {"tid": "task_id", "name": "name", "NumberOfInstances": "n", "NumberOfFeatures": "d",
            "NumberOfClasses": "C", "NumberOfInstancesWithMissingValues": "n_missing_rows"}
    df = tasks[[c for c in cols if c in tasks.columns]].rename(columns=cols)
    df["cost"] = df["n"] * df["d"].clip(lower=1)
    df = df.sort_values("cost").drop(columns="cost")
    df.to_csv(a.out, sep="\t", index=False)
    print(f"{len(df)} tasks written to {a.out}")


if __name__ == "__main__":
    main()
