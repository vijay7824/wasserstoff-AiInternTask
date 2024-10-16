# PDF Summarizer and Keyword Extractor

This project processes PDF files to extract text, generate summaries, and extract keywords using natural language processing techniques. It uses MongoDB to store metadata, summaries, and keywords of the uploaded PDFs.

## Features
- **Summarization**: Automatically summarizes large texts extracted from PDFs using a pre-trained BART model.
- **Keyword Extraction**: Extracts the most relevant keywords from the PDF content using TF-IDF.
- **MongoDB Storage**: Stores PDF metadata, summaries, and keywords in a MongoDB database.
- **Streamlit UI**: Provides a user-friendly interface for uploading and processing PDFs.
- 
## Technologies Used
- Python 3.11
- PyPDF2 for PDF text extraction
- Hugging Face `transformers` for summarization (BART model)
- `TfidfVectorizer` for keyword extraction
- MongoDB for database storage
- Streamlit for UI

## Setup Instructions

### Prerequisites
Ensure you have the following installed:
- Python 3.11 or higher
- MongoDB server or Atlas (if using a cloud instance)
- Pip package manager
- Docker (optional, for Docker setup)

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/pdf-summarizer-keyword-extractor.git](https://github.com/vijay7824/wasserstoff-AiInternTask.git
    cd pdf-summarizer-keyword-extractor
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Environment Variables**:
   You will need to configure MongoDB credentials. You can set these in your environment or directly modify the connection string in the code.

   Update the MongoDB URI in the script (`MongoClient`):
   ```python
   client = MongoClient('mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority')
