"""
Parallel processing utilities for head angle extraction.
Separates multiprocessing logic from business logic for clean modularity.
"""

import multiprocessing as mp
import time
from typing import Callable, Any, List, Tuple
from .config import Config

class ParallelProcessor:
    """
    A utility class for parallel processing that separates multiprocessing concerns
    from business logic.
    """
    
    def __init__(self, num_processes=None):
        """
        Initialize the parallel processor.
        
        Args:
            num_processes: Number of processes to use. If None, uses Config.get_num_processes()
        """
        self.num_processes = num_processes or Config.get_num_processes()
        Config.debug_print(f"Initialized ParallelProcessor with {self.num_processes} processes")
    
    def process_in_chunks(self, data_items: List[Any], processing_function: Callable, 
                         chunk_size_multiplier: int = None, description: str = "Processing") -> List[Any]:
        """
        Process a list of data items in parallel chunks.
        
        Args:
            data_items: List of items to process
            processing_function: Function that takes a chunk of items and returns results
            chunk_size_multiplier: Multiplier for chunk size calculation
            description: Description for logging
            
        Returns:
            List of results from processing_function
        """
        if chunk_size_multiplier is None:
            chunk_size_multiplier = Config.CHUNK_SIZE_MULTIPLIER
        
        start_time = time.time()
        
        print(f"🚀 Using {self.num_processes} processes for {description} on {mp.cpu_count()}-core machine...")
        print(f"Processing {len(data_items)} items...")
        
        # Optimize chunk sizing for better load balancing on many cores
        optimal_chunk_size = max(1, len(data_items) // (self.num_processes * chunk_size_multiplier))
        chunks = [data_items[i:i + optimal_chunk_size] for i in range(0, len(data_items), optimal_chunk_size)]
        
        print(f"📦 Split into {len(chunks)} chunks of ~{optimal_chunk_size} items each for optimal load balancing")
        
        # Process chunks in parallel
        try:
            if self.num_processes == 1:
                # Sequential processing for debugging
                Config.debug_print("Running in sequential mode")
                results = [processing_function(chunk) for chunk in chunks]
            else:
                with mp.Pool(processes=self.num_processes) as pool:
                    print(f"⚡ Starting parallel {description.lower()}...")
                    results = pool.map(processing_function, chunks)
                    print(f"✅ {description} completed!")
        except Exception as e:
            print(f"❌ Error in parallel processing: {e}")
            # Fallback to sequential processing
            print("🔄 Falling back to sequential processing...")
            results = [processing_function(chunk) for chunk in chunks]
        
        end_time = time.time()
        if Config.VERBOSE_TIMING:
            print(f"⏱️  {description} took {end_time - start_time:.2f} seconds")
        
        return results
    
    def process_dict_in_chunks(self, data_dict: dict, processing_function: Callable,
                              chunk_size_multiplier: int = None, description: str = "Processing") -> dict:
        """
        Process a dictionary's items in parallel chunks and combine results.
        
        Args:
            data_dict: Dictionary to process
            processing_function: Function that takes a chunk of (key, value) tuples and returns dict
            chunk_size_multiplier: Multiplier for chunk size calculation  
            description: Description for logging
            
        Returns:
            Combined dictionary of results
        """
        # Convert dictionary items to list for chunking
        data_items = list(data_dict.items())
        
        # Process in chunks
        chunk_results = self.process_in_chunks(
            data_items, processing_function, chunk_size_multiplier, description
        )
        
        # Combine results
        combined_result = {}
        for chunk_result in chunk_results:
            combined_result.update(chunk_result)
        
        print(f"🎉 {description} complete! Processed {len(combined_result)} items using {self.num_processes} cores.")
        return combined_result
    
    @staticmethod
    def combine_chunk_results(chunk_results: List[Tuple[dict, List]], description: str = "items") -> Tuple[dict, List]:
        """
        Combine results from multiple chunks that return (dict, list) tuples.
        
        Args:
            chunk_results: List of (dict, list) tuples from processing chunks
            description: Description for logging
            
        Returns:
            Tuple of (combined_dict, combined_list)
        """
        combined_dict = {}
        combined_list = []
        
        for chunk_dict, chunk_list in chunk_results:
            combined_dict.update(chunk_dict)
            combined_list.extend(chunk_list)
        
        Config.debug_print(f"Combined {len(combined_dict)} {description} from {len(chunk_results)} chunks")
        return combined_dict, combined_list 