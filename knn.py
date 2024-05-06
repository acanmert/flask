import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

def secimlik(secim):
    data = pd.read_csv(f'C:\\Users\\AhmetCan\\source\\repos\\File_Upload\\File_Upload\\wwwroot\\dataset\\{secim}', encoding='"utf-8-sig')
    return  data

def get_similar_items(title, data, selected_features, p_name, p_type, top_n=10):
    try:
        selected_data = data[selected_features]

        # Label Encoding işlemi
        label_encoder = LabelEncoder()
        for column in selected_data:
            selected_data[column] = label_encoder.fit_transform(selected_data[column])

        # Kullanıcının girdisine göre benzer öğeleri bulma
        if p_name in data.columns:
            if p_type == "evet":
                index = data[data[p_name] == int(title)].index[0]
            else:
                index = data[data[p_name] == title].index[0]
            knn_model = NearestNeighbors(n_neighbors=top_n, algorithm='auto', metric='euclidean')
            knn_model.fit(selected_data)

            # Girdiyi yeniden şekillendirme
            query_point = np.array(selected_data.iloc[index]).reshape(1, -1)

            distances, indices = knn_model.kneighbors(query_point)

            # Benzer öğelerin indekslerini ve mesafelerini alıp döndürme
            similar_items_indices = indices.flatten().tolist()
            distances = distances.flatten().tolist()

            similar_items = []
            for idx, dist in zip(similar_items_indices, distances):
                similar_items.append(data.iloc[idx].to_dict())

            return similar_items
        else:
            return ["No recommendation available"]  # veya başka bir hata durumu işleme
    except Exception as e:
        return [f"Error occurred: {str(e)}"]


@app.route('/recommendations', methods=['GET'])
def get_recommendations_endpoint():
    secim = request.args.get('secim')
    data = secimlik(secim)
    selected_features = request.args.get('selected_features').split(',')
    title = request.args.get('title')
    p_name = request.args.get("p_name")
    p_type = request.args.get("p_type")
    recommendations = get_similar_items(title, data, selected_features, p_name, p_type)

    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)
