import math
import re

class BM25SearchEngine:
    def __init__(self, k1=1.2, b=0.75):
        """
        k1: Parámetro de saturación de frecuencia de término (controla qué tanto aporta que una palabra se repita).
        b: Parámetro de penalización por longitud del documento (0 = sin penalización, 1 = penalización total).
        """
        self.k1 = k1
        self.b = b
        self.index = {}          # Índice invertido: {término: {doc_id: frecuencia}}
        self.doc_lengths = {}    # Longitud de cada documento
        self.avg_doc_length = 0  # Longitud promedio de los documentos en el corpus
        self.corpus_size = 0     # Número total de documentos
        self.raw_documents = {}  # Almacenamiento del texto original

    def _tokenize(self, text):
        """Preprocesamiento básico de texto: minúsculas y extracción de palabras."""
        return re.findall(r'\w+', text.lower())

    def fit(self, documents):
        """
        Construye el índice invertido a partir de un diccionario de documentos.
        documents: {doc_id: "texto del documento"}
        """
        self.raw_documents = documents
        self.corpus_size = len(documents)
        total_length = 0
        
        for doc_id, text in documents.items():
            tokens = self._tokenize(text)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            
            for token in tokens:
                if token not in self.index:
                    self.index[token] = {}
                if doc_id not in self.index[token]:
                    self.index[token][doc_id] = 0
                self.index[token][doc_id] += 1
                
        self.avg_doc_length = total_length / self.corpus_size if self.corpus_size > 0 else 0

    def _get_idf(self, term):
        """Calcula la Frecuencia Inversa de Documento (IDF) usando la fórmula estándar de BM25."""
        if term not in self.index:
            return 0
        # Número de documentos que contienen el término
        n_q = len(self.index[term])
        # Fórmula probabilística de IDF con suavizado para evitar valores negativos
        return math.log((self.corpus_size - n_q + 0.5) / (n_q + 0.5) + 1.0)

    def search(self, query, top_n=5):
        """Procesa una consulta y devuelve los documentos ordenados por relevancia BM25."""
        query_tokens = self._tokenize(query)
        scores = {}
        
        for token in query_tokens:
            if token not in self.index:
                continue
            
            idf = self._get_idf(token)
            
            for doc_id, freq in self.index[token].items():
                doc_len = self.doc_lengths[doc_id]
                
                # Componente de Frecuencia de Término Normalizada de BM25
                tf_numerator = freq * (self.k1 + 1)
                tf_denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
                tf_component = tf_numerator / tf_denominator
                
                if doc_id not in scores:
                    scores[doc_id] = 0
                # El score final es la suma del producto de IDF y el componente TF de cada término de la query
                scores[doc_id] += idf * tf_component
                
        # Ordenar documentos por score descendente
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [(doc_id, self.raw_documents[doc_id], round(score, 4)) for doc_id, score in ranked_docs[:top_n]]

# ==========================================
# Ejemplo de Uso y Validación del Sistema
# ==========================================
if __name__ == "__main__":
    # Un corpus de prueba (pueden ser resúmenes de papers o artículos informáticos)
    corpus = {
        "doc1": "The quick brown fox jumps over the lazy dog.",
        "doc2": "Search relevance systems use TF-IDF and BM25 algorithms to rank text documents.",
        "doc3": "Python is an excellent programming language for building search engines and data structures.",
        "doc4": "Information retrieval and search engine optimization require deep analysis of term frequency.",
        "doc5": "Dogs and foxes are mammals, but cats are the elite pets of software engineers."
    }
    
    # Inicializar y entrenar el motor
    engine = BM25SearchEngine()
    engine.fit(corpus)
    
    # Ejecutar una búsqueda de prueba
    query = "search engine algorithms python"
    results = engine.search(query)
    
    print(f"Resultados para la búsqueda: '{query}'\n")
    for doc_id, text, score in results:
        print(f"[{doc_id}] Score: {score} -> {text}")