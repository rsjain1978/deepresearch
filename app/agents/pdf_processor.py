from termcolor import colored
from typing import List, Dict, Any
import os
import asyncio
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import base64
import io
from openai import AsyncOpenAI
from .vector_store import VectorStore
from .knowledge_graph import KnowledgeGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize OpenAI client
client = AsyncOpenAI()

class DocumentProcessor:
    def __init__(self):
        self.vector_store = VectorStore()
        self.knowledge_graph = KnowledgeGraph()
        
        # Text splitter for knowledge graph chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for better entity extraction
            chunk_overlap=50,
            length_function=len,
        )
        
        # Create directory for extracted images
        os.makedirs("extracted_pages", exist_ok=True)
    
    async def extract_text(self, file_path: str) -> str:
        """Extract text from uploaded document based on file type."""
        try:
            print(colored(f"Extracting text from {file_path}", "cyan"))
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.pdf':
                return await self._extract_from_pdf(file_path)
            elif file_extension in ['.txt', '.md', '.json']:
                return self._extract_from_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
                
        except Exception as e:
            print(colored(f"Error extracting text: {str(e)}", "red"))
            raise

    async def _extract_from_pdf(self, file_path: str) -> List[str]:
        """Extract text from PDF file, including text from images."""
        try:
            print(colored(f"[PDF] Opening PDF file: {file_path}", "cyan"))
            doc = fitz.open(file_path)
            
            async def process_page(page_num: int) -> str:
                """Process a single page including its images."""
                page = doc[page_num]
                
                # Get text content
                text_content = page.get_text()
                
                # Extract and process images
                image_list = page.get_images()
                if image_list:
                    print(colored(f"[PDF] Found {len(image_list)} images on page {page_num + 1}", "cyan"))
                    
                    # Process each image in the page
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            # Convert to PIL Image
                            image = Image.open(io.BytesIO(image_bytes))
                            
                            # Save image temporarily
                            temp_path = f"extracted_pages/page_{page_num + 1}_img_{img_index + 1}.png"
                            image.save(temp_path)
                            
                            # Extract text from image using GPT-4 Vision
                            with open(temp_path, "rb") as img_file:
                                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                            
                            # Call GPT-4 Vision
                            response = await client.chat.completions.create(
                                model="gpt-4-vision-preview",
                                messages=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": "Extract and describe any text or important information from this image. Be concise and focus on factual content."},
                                            {"type": "image_url", "image_url": f"data:image/png;base64,{img_base64}"}
                                        ]
                                    }
                                ],
                                max_tokens=300
                            )
                            
                            # Add extracted image text to page content
                            image_text = response.choices[0].message.content
                            if image_text and image_text.strip():
                                text_content += f"\n[Image Content: {image_text}]\n"
                            
                            # Clean up temporary image file
                            os.remove(temp_path)
                            
                        except Exception as img_error:
                            print(colored(f"[PDF] Error processing image {img_index + 1} on page {page_num + 1}: {str(img_error)}", "yellow"))
                
                return text_content

            # Process all pages in parallel
            print(colored(f"[PDF] Processing {len(doc)} pages in parallel", "cyan"))
            page_contents = await asyncio.gather(*[process_page(i) for i in range(len(doc))])
            
            doc.close()
            return page_contents
            
        except Exception as e:
            print(colored(f"[PDF] Error extracting text from PDF: {str(e)}", "red"))
            raise

    def _convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to images with caching using PyMuPDF."""
        try:
            output_dir = Path('extracted_pages')
            pdf_name = Path(pdf_path).stem
            
            # Create a specific directory for this PDF
            pdf_output_dir = output_dir / pdf_name
            pdf_output_dir.mkdir(exist_ok=True)
            
            # Check if directory already has images for this PDF
            existing_images = sorted(pdf_output_dir.glob('page_*.png'))
            
            if existing_images:
                print(colored(f"Found {len(existing_images)} existing images for {pdf_name}", "green"))
                # Load existing images
                images = [Image.open(img_path) for img_path in existing_images]
                return images
            
            # Convert PDF to images using PyMuPDF
            print(colored(f"No existing images found. Converting {pdf_name} to images...", "cyan"))
            
            pdf_document = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Set higher resolution for better quality
                zoom_factor = 2.0  # Higher zoom factor = higher resolution
                matrix = fitz.Matrix(zoom_factor, zoom_factor)
                pixmap = page.get_pixmap(matrix=matrix)
                
                # Convert pixmap to PIL Image
                img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                images.append(img)
                
                # Save the image
                image_path = pdf_output_dir / f"page_{page_num+1:03d}.png"
                img.save(image_path, "PNG", optimize=True)
                print(colored(f"Saved page {page_num+1} to {image_path}", "cyan"))
            
            pdf_document.close()
            return images
            
        except Exception as e:
            print(colored(f"Error converting PDF to images: {str(e)}", "red"))
            raise

    async def _extract_text_from_image(self, image: Image.Image) -> str:
        """Use GPT-4o to extract text from an image."""
        try:
            # Convert PIL Image to base64 string
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Call OpenAI API with image
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a document text extractor. Extract ALL text content from the provided image, preserving paragraphs, lists, and other structural elements. Include ALL text visible in the image."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract the text from this document image:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                        ]
                    }
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(colored(f"Error extracting text from image with GPT-4o: {str(e)}", "red"))
            raise

    def _extract_from_text(self, file_path: str) -> List[str]:
        """Extract text from text-based files and split into page-like chunks."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        text = file.read()
                        # Split text into page-like chunks based on double newlines
                        pages = [page.strip() for page in text.split('\n\n') if page.strip()]
                        if not pages:  # If no double newlines found, return whole text as one page
                            pages = [text]
                        return pages
                except UnicodeDecodeError:
                    continue
                    
            raise ValueError(f"Could not decode file with any of the following encodings: {encodings}")
            
        except Exception as e:
            print(colored(f"Error extracting text from file: {str(e)}", "red"))
            raise

    async def process_document(self, file_path: str) -> Dict:
        """Process document through extraction, vector store, and knowledge graph."""
        try:
            # Extract text - returns list of page contents
            page_contents = await self.extract_text(file_path)
            
            if not isinstance(page_contents, list):
                # For non-PDF files that return a single string, wrap in list
                page_contents = [page_contents]
            
            print(colored(f"Processing {len(page_contents)} pages", "cyan"))
            
            # 1. Add full page contents to vector store
            metadata = {"source": file_path}
            await self.vector_store.add_chunks(page_contents, metadata)
            
            # 2. Create smaller chunks for knowledge graph processing
            all_chunks = []
            for page in page_contents:
                chunks = self.text_splitter.split_text(page)
                all_chunks.extend(chunks)
            
            print(colored(f"Created {len(all_chunks)} chunks for knowledge graph processing", "cyan"))
            
            # 3. Update knowledge graph with smaller chunks
            graph_stats = await self.knowledge_graph.update_graph(all_chunks)
            
            return {
                "pages": page_contents,
                "num_pages": len(page_contents),
                "num_chunks": len(all_chunks),
                "graph_stats": graph_stats
            }
        except Exception as e:
            print(colored(f"Error processing document: {str(e)}", "red"))
            raise 