import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
model = pickle.load(open('model1.pkl', 'rb'))
ALLOWED_HOSTS = ['*']

@app.route('/')
def home():
    return render_template('appui.html')

@app.route('/predict',methods=['POST'])
def predict():
    sentence = str(request.form.values()).lower()
    sentence = sentence.split()
    feature = []

    #check order of feature
    if 'iphone 11' in sentence and 'pro' not in sentence and 'max' not in sentence:
        feature.append(1)
        feature.append(0)
        feature.append(0)
        feature.append(0)
        feature.append(0)
        feature.append(0)
        feature.append(0)
    elif 'iphone 11' in sentence and 'pro' in sentence and 'max' not in sentence:
        feature.append(0)
        feature.append(1)
        feature.append(0)
        feature.append(0)
        feature.append(0)
        feature.append(0)
        feature.append(0)
    elif 'iphone 11' in sentence and 'pro' in sentence and 'max' in sentence:
        feature.append(0)
        feature.append(0)
        feature.append(1)
        feature.append(0)
        feature.append(0)
        feature.append(0)
        feature.append(0)
    elif 'iphone 12' in sentence and 'pro' not in sentence and 'max' not in sentence:
        feature.append(0)
        feature.append(0)
        feature.append(0)
        feature.append(1)
        feature.append(0)
        feature.append(0)
        feature.append(0)
    elif 'iphone 12' in sentence and 'pro' in sentence and 'max' not in sentence:
        feature.append(0)
        feature.append(0)
        feature.append(0)
        feature.append(0)
        feature.append(1)
        feature.append(0)
        feature.append(0)
    elif 'iphone 12' in sentence and 'pro' in sentence and 'max' in sentence:
        feature.append(0)
        feature.append(0)
        feature.append(0)
        feature.append(0)
        feature.append(0)
        feature.append(1)
        feature.append(0)
    else:
        feature.append(0)
        feature.append(0)
        feature.append(0)
        feature.append(0)
        feature.append(0)
        feature.append(0)
        feature.append(1)

    memory = ['64 gb', '64gb','256 gb', '128 gb', '512 gb', '256gb', '128gb', '512gb']
    #if len(set(memory).intersection(sentence.split()) != 0
     #   feature.append(1)
    #else:
     #   feature.append(0)

    #or you can use following:

    if not set(memory).isdisjoint(sentence) == True:
        feature.append(1)
    else:
        feature.append(0)


    unlocked = ['unlocked', 'locked', 'network locked']
    if not set(unlocked).isdisjoint(sentence) == True:
        feature.append(1)
    else:
        feature.append(0)

    #change order as per feature set
    if 'at&t' in sentence or 'att' in sentence:
        feature.append(1)
    else:
        feature.append(0)

    if 'verizon' in sentence:
        feature.append(1)
    else:
        feature.append(0)

    if 'sprint' in sentence:
        feature.append(1)
    else:
        feature.append(0)

    if 'mint' in sentence:
        feature.append(1)
    else:
        feature.append(0)

    if 't-mobile' in sentence:
        feature.append(1)
    else:
        feature.append(0)

    if 'cricket' in sentence:
        feature.append(1)
    else:
        feature.append(0)

    if 'GSM/CDMA' in sentence:
        feature.append(1)
    else:
        feature.append(0)

    if '4g/5g' in sentence:
        feature.append(1)
    else:
        feature.append(0)
    


    color = ['All Colors', 'Various Colors','All Color' , 'White' , 'Green' , 'Blue' , 'Pacific', 'Red',
                'Black','White','Purple','Silver','Gold','Graphite','Coral']

    if not set(color).isdisjoint(sentence) == True:
        feature.append(1)
    else:
        feature.append(0)

        
    final_features = [np.array(feature)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('appui.html', prediction_text='Predicted Class:  {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
