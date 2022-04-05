import os
import time
import joblib
import savReaderWriter as srw
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

import numpy as np


def run_gametes(h=0.2,
                maf1=0.2,  # allele freq 1
                maf2=0.2,  # allele freq 2
                model_name="model_1",  # model name
                q=1,  # quantile
                p=1,  # population
                t=10000,  # attempts
                n=0.01,  # minMAF
                x=0.5,  # maxMAF
                a2=1000,  # total_attributes
                s=2000,  # cases
                w=2000,  # controls
                r=1,  # replicates
                dataset_name='dataset_1'):  # dataset name
    command_str = 'java -jar models/GAMETES_2.1.jar -M " -h {h} -a {maf1} -a {maf2} -o models/gametes/{model_name}.txt" -q {q} -p {p} -t {t} ' \
                 '-D " -n {n} -x {x} -a {a2} -s {s} -w {w} -r {r} -o models/gametes/{dataset_name}"'
    command_str = command_str.format(
        h=h,  # heritability
        maf1=maf1,  # allele freq 1
        maf2=maf2,  # allele freq 2
        model_name=model_name,  # model name 1
        q=q,  # quantile
        p=p,  # population
        t=t,  # attempts
        n=n,  # minMAF
        x=x,  # maxMAF
        a2=a2,  # total_attributes
        s=s,  # cases
        w=w,  # controls
        r=r,  # replicates
        dataset_name=dataset_name  # model name 2
    )
    os.system(command_str)

    return


