import logging
from pathlib import Path
import Stage_1.parsers.parse_text
import Stage_1.parsers.parse_image
import Stage_1.parsers.parse_audio
import Stage_1.parsers.parse_tabular
import Stage_1.parsers.parse_container
import Stage_1.parsers.parse_video
import Stage_1.registry as registry

logger = logging.getLogger(__name__)

image_to_test = r"C:\Users\henry\Documents\My_Code\Small Database\DSCF0182.JPG"
video_to_test = r"Z:\My Drive\Demo2.mp4"
pdf_to_test = r"Z:\My Drive\2025\BDSM Subtypes And Their Prevalence - Aella.pdf"
audio_to_test = r"Z:\My Drive\_Photos and Media\Videos\Bleep for cussing.wav"
spreadsheet_to_test = r"C:\Users\henry\Downloads\multimodal_pipeline_registry.xlsx - File Type Registry.csv"
zip_to_test = r"C:\Users\henry\Downloads\Skyblock-2.1.zip"

result = registry.parse(zip_to_test, config={})

print(f"Modality: {result.modality}")
print(f"Success: {result.success}")
print(f"Error: {result.error}")
print(f"Output: {result.output}")
print(f"Metadata: {result.metadata}")
print(f"Also Contains: {result.also_contains}")
