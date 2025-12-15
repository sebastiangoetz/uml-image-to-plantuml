import os
import shutil
from pathlib import Path

import pipeline
import config

processed_dir = Path(config.PROCESSED_DIR)
if os.path.exists(processed_dir):
    shutil.rmtree(processed_dir)
os.makedirs(processed_dir, exist_ok=True)

pipeline.run_uml_extraction_pipeline("C:\\Users\\edgar\\Documents\\Universitaet\\Semester_X\\Diplomarbeit\\ma-edgar-solf\\thesis\\evaluation\\diagrams\\handwritten\\M\\IMG_3669.JPG", model_size="n")