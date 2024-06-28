from flask import Flask, request, jsonify
from collections import defaultdict
import  suggestion
app = Flask(__name__)




@app.route('/recommendations', methods=['GET'])
def get_recommendations_endpoint():
    secim = request.args.get('secim')
    email = request.args.get("email")
    data = suggestion.secimlik(secim,email)
    selected_features = request.args.get('selected_features').split(',')
    p_name = request.args.get('p_name')
    p_pk = request.args.get("p_pk")
    p_type = request.args.get("p_type")

    # Önerileri al
    recommendations = suggestion.get_recommendations(p_name, data, selected_features, p_pk, p_type, 10)



    return recommendations

if __name__ == '__main__':
    app.run(debug=True)
