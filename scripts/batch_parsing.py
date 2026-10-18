import os
from pathlib import Path
import time

# --- CPU & Performance Optimizations ---
# Limit the math and layout threads to prevent the 4-core CPU from freezing
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
except ImportError:
    raise ImportError("Please install Docling first: pip install docling")


def batch_process_pdfs(input_dir: str, output_dir: str):
    """
    Scans a folder for PDFs, parses them using Docling, 
    and exports them to structural Markdown optimized for RAG chunking.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup specialized PDF pipeline configurations
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True            # Critical if bidding PDFs have scanned components
    pipeline_options.do_table_structure = True # Critical for transformer specs and manufacturer matrices
    pipeline_options.images_scale = 2.0       # Avoid storing visual artifacts to keep RAM usage low

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # 2. Collect all target PDF documents
    pdf_files = list(input_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'.")
        return

    print(f"Found {len(pdf_files)} PDFs. Initializing models on CPU...")
    
    # 3. Sequentially process documents to control memory footprints
    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")
        start_time = time.time()
        
        try:
            # Conversion run
            result = converter.convert(pdf_file)
            
            # Export payload directly into Markdown text format
            markdown_content = result.document.export_to_markdown()
            
            # Save output using the same filename but a .md extension
            output_file = output_path / f"{pdf_file.stem}.md"
            output_file.write_text(markdown_content, encoding="utf-8")
            
            elapsed = time.time() - start_time
            print(f"   Success! Saved to {output_file.name} ({elapsed:.1f}s)")
            
        except Exception as e:
            print(f"   ❌ Error processing {pdf_file.name}: {str(e)}")


if __name__ == "__main__":
    # 1. Descobre onde o script atual está rodando (raiz/script/)
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    # 2. Sobe para a raiz do projeto e aponta para a pasta 'data'
    PROJECT_ROOT = SCRIPT_DIR.parent
    BASE_DATA_DIR = PROJECT_ROOT / "data"
    
    # 3. Define as subpastas de entrada e saída dentro de 'data'
    INPUT_FOLDER = BASE_DATA_DIR / "samples"
    OUTPUT_FOLDER = BASE_DATA_DIR / "processed"
    
    # Cria as pastas automaticamente se elas não existirem na raiz
    INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    print(f"Caminho absoluto de entrada: {INPUT_FOLDER}")
    print(f"Caminho absoluto de saída: {OUTPUT_FOLDER}")
    print(f"\n👉 Insira seus arquivos PDF em: {INPUT_FOLDER}")
    
    # Executa o processo em lote
    batch_process_pdfs(INPUT_FOLDER, OUTPUT_FOLDER)

