import dspy
from typing import List, Union, Optional
import vecs
from sentence_transformers import SentenceTransformer
from dsp.utils import dotdict

class SupabaseRM(dspy.Retrieve):
    """
    A DSPy retrieval module that uses Supabase Vector for similarity search.
    Supports both text and image embeddings through configurable embedding models.
    """
    
    text_field = 'long_text'  # Class variable for the text field name
    
    def __init__(
        self,
        connection_string: str,
        collection_name: str,
        embedding_model: Union[str, SentenceTransformer] = 'BAAI/bge-small-en-v1.5',
        dimension: int | None = None,
        k: int = 3,
    ):
        """
        Initialize the Supabase Vector retrieval module.
        Args:
            connection_string (str): Supabase database connection string
            collection_name (str): Name of the vector collection
            embedding_model (Union[str, SentenceTransformer]): Model name or instance for embeddings
            dimension (int): Dimension of the vector embeddings
            k (int): Default number of results to return
            text_field (str, optional): Name of the field to use for text content in DSPy
        """
        super().__init__(k=k)
        
        # Initialize Supabase Vector client
        self.vx = vecs.create_client(connection_string)
        
        # Setup embedding model
        if isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformer(embedding_model)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        else:
            self.embedding_model = embedding_model
            self.dimension = dimension

        # Get or create collection
        print(f"DIMENSION: {self.dimension}")
        print(f"COLLECTION NAME: {collection_name}")
        print(type(self.dimension))
        self.collection = self.vx.get_or_create_collection(
            name=collection_name,
            dimension=dimension
        )

    def forward(
        self,
        query_or_queries: Union[str, List[str]],
        k: Optional[int] = None
    ) -> dspy.Prediction:
        """
        Retrieve similar documents from Supabase Vector.

        Args:
            query_or_queries: Single query or list of queries
            k: Number of results to return (optional, defaults to self.k)

        Returns:
            dspy.Prediction containing retrieved passages
        """
        k = k if k is not None else self.k
        
        # Handle single query or multiple queries
        if isinstance(query_or_queries, str):
            queries = [query_or_queries]
        else:
            queries = query_or_queries
        
        all_results = []
        
        for query in queries:
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(query)
            
            # Perform vector similarity search
            results = self.collection.query(
                data=query_embedding,
                limit=k,
                include_metadata=True
            )
            
            # Extract passages from results and convert to dotdict format
            for row in results:
                # Convert SQLAlchemy Row to dict
                row_dict = row._asdict()
                
                passage = dotdict({
                    'id': row_dict.get('id', 'unknown'),
                    'score': row_dict.get('score', 0.0),
                    'long_text': row_dict.get('metadata', {}).get('content', str(row))
                })
                all_results.append(passage)
        
        return all_results 