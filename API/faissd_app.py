from flask import Flask, request, jsonify
import faiss
import numpy as np

app = Flask(__name__)

def get_top_k_cosine_matches(query_embedding, embeddings_list, k):
    ''' 중심이 되는 embedding(query_embedding)과 이를 비교할 embedding들이 리스트(embeddings_list) 와 k를 입력받아
    embeddings_list에서 query_embedding와 가장 유사한 embedding들의 index number와 유사도를 json형태로 반환해줍니다. 
    아래에서는 D는 embedding의 dimension을 뜻하고 N은 embedding의 갯수를 뜻합니다. 
    
    Args:
        query_embedding: 
            한개의 embedidng(e.g, 홍길동 이력서의 embedding or 어떠한 공고 하나의 embedding)
            shape: (D,)
            
        embeddings_list: 
            embedding의 리스트(e.g, 공고 5000개의 embedding들 or 지원한 150명의 이력서들의 embedding)
            shape: (N, D)
            
        k: 
            상위 몇개를 걸러낼 것인지를 정하는 parameter
        
    Return:
        {'indices': indices, 'similarities': similarities})
        indices: list, embeddings_listd에서 상위 k개에 해당하는 공고들의 index number들
        similarities list, embeddings_listd에서 상위 k개에 해당하는 공고들의 유사도 점수들
    '''
    
    if not isinstance(query_embedding, np.ndarray) or not isinstance(embeddings_list, np.ndarray):
        raise ValueError("query_embedding and embeddings_list must be numpy arrays")

    if query_embedding.ndim != 1 or embeddings_list.ndim != 2:
        raise ValueError("query_embedding must be 1D and embeddings_list must be 2D")

    if query_embedding.shape[0] != embeddings_list.shape[1]:
        raise ValueError("query_embedding and embeddings_list must have matching dimensions")


    query_embedding = query_embedding.astype('float32').reshape(1, -1) # (1, D)
    embeddings_list = embeddings_list.astype('float32') # (N, D)

    index = faiss.IndexFlatIP(embeddings_list.shape[1])
    index.add(embeddings_list)
    similarities, indices = index.search(query_embedding, k)

    normalized_similarities = ((similarities[0] + 1.0) / 2.0).tolist()
    return indices[0].tolist(), normalized_similarities


@app.route('/match', methods=['POST'])
def match_embeddings():
    data = request.get_json()

    if 'query_embedding' not in data or 'embeddings_list' not in data or 'k' not in data:
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        query_embedding = np.array(data['query_embedding'], dtype=np.float32)
        embeddings_list = np.array(data['embeddings_list'], dtype=np.float32)
        k = int(data['k'])

        indices, similarities = get_top_k_cosine_matches(query_embedding, embeddings_list, k)
        return jsonify({'indices': indices, 'similarities': similarities})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)