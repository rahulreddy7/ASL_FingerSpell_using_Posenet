import json
import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

dict1 = dict()
dict1['buy'] = 0
dict1['communicate'] = 1
dict1['fun'] = 2
dict1['hope'] = 3
dict1['mother'] = 4
dict1['really'] = 5

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World from flaskapp.py!'


def maxoccur(df1):
    df_new = df1.groupby(['Label']).size().reset_index(name='counts')
    max1 = 0
    index1 = -1
    for index, row in df_new.iterrows():
        if max1 < row['counts']:
            max1 = row['counts']
            index1 = row['Label']
    return index1


@app.route('/processjson', methods=['POST'])
def processjson():
    getjson = request.get_json()
    with open('/home/ubuntu/flaskapp/kp/key_points.json', 'w+') as temp_data:
        json.dump(getjson, temp_data)

    runcommand = "python /home/ubuntu/flaskapp/convert_to_csv.py"
    os.system(runcommand)
    singledatafram = pd.read_csv("/home/ubuntu/flaskapp/kp/key_points.csv")
    singledatafram = singledatafram[["leftShoulder_x", "leftShoulder_y", "rightShoulder_x", "rightShoulder_y", "leftElbow_x", "leftElbow_y", "rightElbow_x", "rightElbow_y", "leftWrist_x", "leftWrist_y", "rightWrist_x", "rightWrist_y"]]
    singledatafram = singledatafram.fillna(singledatafram.mean())
    singledatafram['Label'] = 0
    if(singledatafram.shape[0] > 150):
        df1 = singledatafram[:150]
    else:
        len1 = singledatafram.shape[0]
        for i in range(len1, 150):
            singledatafram.loc[i] = singledatafram.mean()
    classlabel = []
    mlpmodel = pickle.load(open("/home/ubuntu/flaskapp/models/mlp.sav", 'rb'))
    knnmodel = pickle.load(open("/home/ubuntu/flaskapp/models/knn.sav", 'rb'))
    lrmodel = pickle.load(open("/home/ubuntu/flaskapp/models/lr.sav", 'rb'))
    rbfmodel = pickle.load(open("/home/ubuntu/flaskapp/models/svmrbf.sav", 'rb'))

    listofmodels = [mlpmodel, knnmodel, lrmodel, rbfmodel]

    countmodel = 0
    for each in listofmodels:
        data = singledatafram.iloc[:, :-1].values
        label = singledatafram.iloc[:, -1].values
        x = StandardScaler()
        newdata = x.fit_transform(data)

        predicted_label = each.predict(newdata)
        pl_dataframe = pd.DataFrame(predicted_label)
        pl_dataframe.columns = ['Label']

        #should find highest occurence
        getvalue = maxoccur(pl_dataframe)
        for x in dict1:
            if dict1[x] == int(getvalue):
                classlabel[countmodel] = x
        countmodel = countmodel + 1

    return jsonify(classlabel)


if __name__ == '__main__':
    app.run()
