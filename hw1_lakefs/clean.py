# clean.py
import pandas as pd
import numpy as np
import sys

def clean(input_csv, output_csv):
    data = pd.read_csv(input_csv)
    # Remove not relevant cols
    data = data.dropna(subset=['region','age','weight','height','howlong',
                               'gender','eat','train','background','experience',
                               'schedule','deadlift','candj','snatch','backsq'])
    data = data.drop(columns=['affiliate','team','name','athlete_id','fran',
                              'helen','grace','filthy50','fgonebad','run400',
                              'run5k','pullups','train'])
    # Remove Outliers
    data = data[data['weight'] < 1500]
    data = data[data['gender'] != '--']
    data = data[data['age'] >= 18]
    data = data[(data['height'] < 96) & (data['height'] > 48)]
    data = data[((data['deadlift'] > 0) & (data['deadlift'] <= 1105)) | 
                ((data['gender']=='Female') & (data['deadlift'] <= 636))]
    data = data[(data['candj'] > 0) & (data['candj'] <= 395)]
    data = data[(data['snatch'] > 0) & (data['snatch'] <= 496)]
    data = data[(data['backsq'] > 0) & (data['backsq'] <= 1069)]
    # Clean Survey Data
    data = data.replace({'Decline to answer|': np.nan})
    data = data.dropna(subset=['background','experience','schedule','howlong','eat'])
    # New feature
    data['total_lift'] = data['deadlift'] + data['candj'] + data['snatch'] + data['backsq']
    # Save
    data.to_csv(output_csv, index=False)
    print(f"cleaned → {output_csv}, shape={data.shape}")

if __name__=='__main__':
    in_csv, out_csv = sys.argv[1], sys.argv[2]
    clean(in_csv, out_csv)