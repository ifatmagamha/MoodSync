import openai
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json

class RAGModule:
    def __init__(self, model_name="all-MiniLM-L6-v2", index_path=None):
        """
        Initialize the RAG Module.
        
        Args:
            model_name (str): Name of the embedding model (default: 'all-MiniLM-L6-v2').
            index_path (str): Path to the FAISS index file, if exists.
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

        # Load FAISS index if available
        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(index_path.replace('.index', '.json'), 'r') as f:
                self.documents = json.load(f)

    def build_index(self, data, index_path):
        """
        Build a FAISS index from a list of text data.
        
        Args:
            data (list): List of text data to index.
            index_path (str): Path to save the FAISS index.
        """
        embeddings = self.embedding_model.encode(data, convert_to_tensor=True).detach().numpy()
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.documents = data

        # Save index and corresponding documents
        faiss.write_index(self.index, index_path)
        with open(index_path.replace('.index', '.json'), 'w') as f:
            json.dump(data, f)

    def retrieve_context(self, query, top_k=3):
        """
        Retrieve relevant documents based on the query.
        
        Args:
            query (str): Input query or text.
            top_k (int): Number of top results to retrieve.
        
        Returns:
            list: List of relevant documents.
        """
        if not self.index:
            raise ValueError("Index is not initialized. Build or load an index first.")

        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True).detach().numpy()
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]

    def generate_response(self, query, additional_context, openai_api_key):
        """
        Generate a response using OpenAI GPT based on query and additional context.
        
        Args:
            query (str): User's query.
            additional_context (list): Retrieved documents to augment the prompt.
            openai_api_key (str): OpenAI API key for GPT model.

        Returns:
            str: Generated response.
        """
        openai.api_key = openai_api_key

        prompt = (
            "You are an assistant helping a therapist analyze patient sessions. Below is the patient's recent transcript "
            "followed by related documents retrieved from previous sessions or relevant data. Provide a summary or response based on all the information.\n\n"
            f"Transcript:\n{query}\n\n"
            f"Additional Context:\n{'\n\n'.join(additional_context)}\n\n"
            "Provide a structured summary, actionable insights, and any relevant recommendations."
        )

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}],
        )
        return response.choices[0].message['content']

# Example usage within the AudioAgent workflow
class EnhancedAudioAgent(AudioAgent):
    def __init__(self, openai_api_key, patient_data, rag_module, target_language="en"):
        super().__init__(openai_api_key, patient_data, target_language)
        self.rag_module = rag_module

    def summarize_with_rag(self, transcript):
        """
        Summarize the transcription with additional context using RAG.
        
        Args:
            transcript (str): Transcribed text.

        Returns:
            dict: Summary enriched with RAG.
        """
        additional_context = self.rag_module.retrieve_context(transcript)
        summary = self.rag_module.generate_response(transcript, additional_context, self.openai_api_key)
        return {
            "patient_name": self.patient_data.get("name"),
            "date": str(datetime.date.today()),
            "summary": summary
        }
