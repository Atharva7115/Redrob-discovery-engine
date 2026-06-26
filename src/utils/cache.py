import os
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from src.config import CACHE_DIR

class FileCache:
    """Thread-safe on-disk JSON cache.
    Saves key-value pairs as individual JSON files under a namespace directory.
    """
    
    def __init__(self, namespace: str):
        self.cache_dir = Path(CACHE_DIR) / namespace
        os.makedirs(self.cache_dir, exist_ok=True)
        self.lock = threading.Lock()
        
    def _get_file_path(self, key: str) -> Path:
        # Sanitize key to be a valid filename
        safe_key = "".join(c for c in key if c.isalnum() or c in ("_", "-"))
        return self.cache_dir / f"{safe_key}.json"
        
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a value from the cache. Returns None if not found."""
        path = self._get_file_path(key)
        if not path.exists():
            return None
            
        with self.lock:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
                
    def set(self, key: str, value: Dict[str, Any]):
        """Save a value to the cache."""
        path = self._get_file_path(key)
        with self.lock:
            try:
                # Write to a temp file first, then rename (atomic write)
                temp_path = path.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(value, f, indent=2, ensure_ascii=False)
                if os.path.exists(path):
                    os.remove(path)
                os.rename(temp_path, path)
            except Exception as e:
                print(f"Failed to write to cache: {e}")
                # Clean up temp file if it exists
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                        
    def delete(self, key: str):
        """Remove a value from the cache."""
        path = self._get_file_path(key)
        with self.lock:
            if path.exists():
                try:
                    os.remove(path)
                except Exception:
                    pass
