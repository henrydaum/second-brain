import logging
from pathlib import Path
import Stage_1.parsers.parse_text
import Stage_1.parsers.parse_image
import Stage_1.parsers.parse_audio
import Stage_1.parsers.parse_tabular
import Stage_1.parsers.parse_container
import Stage_1.parsers.parse_video
import Stage_1.registry as registry

logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

image_to_test = r"C:\Users\henry\Documents\My_Code\Small Database\DSCF0182.JPG"
video_to_test = r"Z:\My Drive\Demo2.mp4"
pdf_to_test = r"Z:\My Drive\Resources\The Whole by Henry Daum.pdf"
audio_to_test = r"Z:\My Drive\_Photos and Media\Videos\Bleep for cussing.wav"
spreadsheet_to_test = r"C:\Users\henry\Downloads\multimodal_pipeline_registry.xlsx - File Type Registry.csv"
zip_to_test = r"C:\Users\henry\Downloads\Skyblock-2.1.zip"

for file in [image_to_test, video_to_test, pdf_to_test, audio_to_test, spreadsheet_to_test, zip_to_test]:
	result = registry.parse(file, config={})

	print(f"Modality: {result.modality}")
	print(f"Success: {result.success}")
	print(f"Error: {result.error}")
	# print(f"Output: {result.output}")
	print(f"Metadata: {result.metadata}")
	print(f"Also Contains: {result.also_contains}")
	print("-" * 40)
