import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


def get_recommendations_knn(title, data, selected_features, p_name, p_type, top_n):
    try:
        selected_data = data[selected_features]

        # Label Encoding işlemi
        label_encoder = LabelEncoder()
        for column in selected_data:
            selected_data.loc[:, column] = label_encoder.fit_transform(selected_data[column])


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
                similar_item_dict = data.iloc[idx].to_dict()
                similar_item_dict['distance'] = dist
                similar_items.append(similar_item_dict)

            return similar_items
        else:
            return ["No recommendation available"]  # veya başka bir hata durumu işleme
    except Exception as e:
        return [f"Error occurred: {str(e)}"]



