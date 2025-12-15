import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from DatasetReader import DatasetReader

class PromptExamplesDB:

    def __init__(self):
        self.dataset_reader = DatasetReader(1000, "..\\train.json")
        data = self.dataset_reader.load()
        self.documents = dict()
        for item in data:
            self.documents[item["nl"].split("concode")[0]] = item["code"]
        self.idx_document = list(self.documents.keys())

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self.model.encode(self.idx_document, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype("float32")
        dim = embeddings.shape[1]  # dimensión del vector
        index = faiss.IndexFlatL2(dim)  # índice básico por L2
        index.add(embeddings)  # cargar vectores al índice

        self.index = index

    def search(self, query, n):
        query_vec = self.model.encode([query], normalize_embeddings=True).astype("float32")

        distancias, idx = self.index.search(query_vec, n)

        prompt = ""
        fewShorLearning = ""
        fewShorLearning += "You are an expert at writing prompts for code generation.\n"
        fewShorLearning +=  "\n"
        fewShorLearning +=  "Given:\n"
        fewShorLearning +=  "- A task description\n"
        fewShorLearning +=  "- A current prompt\n"
        fewShorLearning +=  "- Several input/output examples\n"
        fewShorLearning +=  "\n"
        fewShorLearning +=  "Your goal is to rewrite the prompt so that it better guides the model\n"
        fewShorLearning +=  "to produce correct Java code.\n"
        fewShorLearning +=  "\n"
        fewShorLearning +=  "Current prompt:\n"
        fewShorLearning +=  f"<{query}>\n"
        fewShorLearning +=  "\n"
        fewShorLearning +=  "Examples:\n"
        k = 1
        for i, id_doc in enumerate(idx[0]):
            prompt += f"Prompt {k}:\n"
            prompt += f"{self.idx_document[id_doc]} \n"
            prompt += "\n"
            prompt += f"Result {k}:\n"
            prompt += f"{self.documents[self.idx_document[id_doc]]}\n\n\n"
            prompt += f"_____________________________________\n"

            #fewShorLearning +=  f"<{self.idx_document[id_doc]}, {self.documents[self.idx_document[id_doc]]}>\n"
            fewShorLearning +=  f"<{self.idx_document[id_doc]}>\n"
            k += 1

        prompt += f"Prompt: {query}\n"
        prompt += "\n"
        prompt += f"Result:\n"
        fewShorLearning +=  "\n"
        fewShorLearning +=  "\n"
        fewShorLearning +=  "Rewrite the prompt as a precise task specification for a Java code generator.\n"
        fewShorLearning +=  "The prompt must:\n"
        fewShorLearning +=  "- explicitly describe how to use the provided class and method context,\n"
        fewShorLearning +=  "- forbid inventing methods, fields, or imports not present in the context,\n"
        fewShorLearning +=  "- state how to handle edge cases mentioned in the description,\n"
        fewShorLearning +=  "- define a strict output format (only valid Java code, no explanations).\n"
        fewShorLearning +=  "\n"
        fewShorLearning +=  "Return only the rewritten prompt.\n"

        return prompt, fewShorLearning
