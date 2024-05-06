from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cosine,knn

app = Flask(__name__)

def secimlik(secim):
    data = pd.read_csv(f'C:\\Users\\AhmetCan\\source\\repos\\File_Upload\\File_Upload\\wwwroot\\dataset\\{secim}', encoding='"utf-8-sig')
    return  data

@app.route('/recommendations', methods=['GET'])
def get_recommendations_endpoint():
    secim = request.args.get('secim')
    data = secimlik(secim)
    selected_features = request.args.get('selected_features').split(',')
    title = request.args.get('title')
    p_name = request.args.get("p_name")
    p_type = request.args.get("p_type")
    recommendations =cosine.get_recommendations_cosine(title, data, selected_features,p_name,p_type,data.shape[0])
    # recommendations =knn.get_recommendations_knn(title, data, selected_features,p_name,p_type,data.shape[0])


    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)