from pathlib import Path
import shutil
from ultralytics import YOLO
from huggingface_hub import snapshot_download

class ModelManager:
    def __init__(self):
        self.model_cache_dir = Path("./storage/models")
        self.model_weights_path = self.model_cache_dir / "license_yolo_N_96.5_1024.pt"
        self.model = None
        
    async def load_model(self):
        if self.model_weights_path.exists():
            self.model = YOLO(str(self.model_weights_path))
            return
            
        # Download model if not exists
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = self.model_cache_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            model_dir = snapshot_download(
                "omarelsayeed/licence_plates",
                cache_dir=str(temp_dir),
                local_files_only=False,
                local_dir=str(temp_dir)
            )
            
            source_path = Path(model_dir) / "license_yolo_N_96.5_1024.pt"
            if source_path.exists():
                shutil.copy2(source_path, self.model_weights_path)
                self.model = YOLO(str(self.model_weights_path))
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)