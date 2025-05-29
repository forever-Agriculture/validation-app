# Citation Validation System: Approach and Methodology

## 1. Project Goal

The primary objective of this project is to develop an automated system to validate AI-generated citations against a provided source text. The system aims to classify each citation's veracity (e.g., "likely true," "likely untrue," "ambiguous") based on semantic similarity, thereby assisting in prioritizing manual data review efforts.

## 2. Core Technology: Semantic Similarity

The chosen approach leverages state-of-the-art **sentence embeddings** to capture the semantic meaning of text.

*   **Model**: `all-MiniLM-L6-v2` from the `sentence-transformers` library https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2.
    *   **Rationale**: This model offers an excellent balance between computational efficiency (fast inference, smaller size) and strong performance on semantic similarity tasks. It's well-suited for processing potentially large volumes of text and citations without requiring extensive computational resources.
*   **Similarity Metric**: Cosine similarity is used to measure the relatedness between the embedding of a citation and the embeddings of chunks from the source text. A higher cosine similarity score (closer to 1.0) indicates greater semantic overlap.

## 3. Text Preprocessing Pipeline

Effective preprocessing is crucial for accurate semantic comparison. The `source_text` provided in the data has a specific structure: it often concatenates content from multiple "files," each potentially starting with a "Content from " delimiter, and is formatted in Markdown.

The preprocessing pipeline (`chunk_source_text` function) involves:

1.  **Segmentation**: The raw `source_text` is first split by the `"Content from "` delimiter. This treats each resulting segment as a distinct unit of text, likely corresponding to an original source document.
2.  **Markdown to Plain Text Conversion**:
    *   Each segment is processed as Markdown. The `markdown-it-py` library converts the Markdown to HTML.
    *   `BeautifulSoup4` is then used to parse this HTML and extract clean, readable plain text. The `get_text(separator=" ")` method helps preserve word separation.
    *   **Rationale for Simplicity (V9 Approach)**: Initial iterations involved complex regular expression-based stripping of metadata headers (e.g., `filename.md`, `Title:`, `URL Source:`) *before* Markdown conversion. However, empirical testing showed that this complexity did not yield significantly better validation results with the `all-MiniLM-L6-v2` model. The current, simpler approach relies on the robustness of the Markdown parser and the sentence embedding model to handle residual structural text. This "Ultra-Simple Structure-Aware Chunker (V9)" was found to be more maintainable while providing comparable performance.
3.  **Sentence Tokenization**: The extracted plain text from each segment is tokenized into individual sentences using NLTK's `punkt` tokenizer. This breaks down the text into manageable units for embedding.
4.  **Normalization and Filtering**:
    *   Each sentence is normalized (converted to lowercase, leading/trailing whitespace removed, newlines replaced with spaces).
    *   Sentences that are too short (based on word count defined by `MIN_CITATION_LENGTH_WORDS` or character count by `MIN_CHUNK_LENGTH_CHARS`) are filtered out to reduce noise and focus on meaningful content. These filtered sentences form the "source text chunks."

## 4. Citation Validation Logic

The `validate_citation_item` function performs the core validation:

1.  **Citation Normalization**: The input `citation_text` undergoes the same normalization process as the source text chunks.
2.  **Edge Case Handling**:
    *   If a normalized citation is too short (fewer words than `MIN_CITATION_LENGTH_WORDS`), it's marked "untrue (too short)".
    *   If no valid source text chunks were generated from the `source_text` for a record, the citation is marked "untrue (no source text)".
3.  **Embedding Generation**: The normalized citation and all valid source text chunks are converted into dense vector embeddings using the loaded `all-MiniLM-L6-v2` model.
4.  **Similarity Calculation**: Cosine similarity is computed between the citation's embedding and each source text chunk's embedding.
5.  **Confidence Score**: The highest similarity score obtained from comparing the citation against all chunks is considered the primary indicator of a match. This score is clamped between 0.0 and 1.0.
6.  **Status Assignment**: The confidence score is then mapped to a human-readable validation status using a set of thresholds:
    *   `>= 0.80`: "very likely true"
    *   `>= SIMILARITY_THRESHOLD` (default 0.6): "likely true"
    *   `>= 0.40`: "ambiguous / review recommended"
    *   `>= 0.20`: "likely untrue"
    *   `< 0.20`: "very likely untrue"
    These thresholds can be tuned to adjust the system's sensitivity.

## 5. Output

The script updates the input records by adding two new fields to each citation object:
*   `validation_status`: The assigned status string.
*   `confidence_score_percent`: The confidence score, rounded to two decimal places and expressed as a percentage.
The modified records are saved to `output_validated.json`.

## 6. Key Dependencies

*   `nltk`: For sentence tokenization.
*   `sentence-transformers`: For the pre-trained language model and embedding utilities.
*   `numpy`: Underlying numerical library for `sentence-transformers`.
*   `markdown-it-py`: For robust Markdown to HTML conversion.
*   `beautifulsoup4`: For HTML parsing and text extraction.

## 7. Potential Future Enhancements (Beyond Current Scope)

*   **Add Second Layer ov Validation**: The idea is to select ambiguous and untrue citations and run them again against a different (simpler) validation layer - for example this layer could employ more direct lexical matching techniques.
*   **Advanced Parameter Tuning**: More systematic tuning of similarity thresholds using a labeled validation set, if available.
*   **Alternative Embedding Models**: Experimentation with larger or domain-specific embedding models if higher accuracy is required and computational resources permit.
*   **More Sophisticated Chunking Strategies**: If the current chunking proves insufficient for certain complex document structures, re-evaluating more advanced text segmentation techniques could be considered.

## 8. Potential Business Impact and Workflow Integration

The citation validation system offers several potential benefits and can be integrated into data quality assurance workflows:

### a. Enhanced Efficiency in Manual Review

*   **Prioritization**: The primary business value lies in significantly reducing the manual effort required to verify citations. By categorizing citations into statuses like "very likely true," "ambiguous / review recommended," and "very likely untrue," reviewers can prioritize their attention:
    *   **"Very Likely True" (e.g., >80% confidence)**: These citations could potentially be fast-tracked or undergo a much lighter spot-check, freeing up significant reviewer time.
    *   **"Ambiguous / Review Recommended" (e.g., 40-60% confidence)**: These become the prime candidates for thorough manual review, as the system is uncertain.
    *   **"Likely Untrue" / "Very Likely Untrue" (e.g., <40% confidence)**: These also require manual verification, but the system provides a strong indication of potential issues, allowing reviewers to approach them with a specific focus.
    *   **"Untrue (no source text)" / "Untrue (too short)"**: These highlight clear issues—either missing source data or insufficient citation context—that need immediate attention, likely data correction or enrichment.
*   **Time Savings**: By focusing manual effort on the most uncertain or problematic citations, the overall time spent on data validation can be drastically reduced. The exact savings would depend on the volume of data and the baseline accuracy of the AI-generated citations.

### b. Improved Data Quality and Consistency

*   **Systematic Checking**: The automated system provides a consistent method for checking all citations, reducing human error or variability that can occur in purely manual processes.
*   **Feedback Loop**: Persistently low confidence scores for citations related to specific data sources or types of information can highlight systemic issues in the upstream data generation or extraction processes. This feedback can be used to improve those processes.

### c. Scalability

*   As the volume of data and citations grows, a manual validation approach becomes increasingly untenable. This automated system provides a scalable solution to maintain data quality standards.

### d. Workflow Integration Example

1.  **Data Ingestion**: New records with AI-generated citations and source texts are ingested.
2.  **Automated Validation**: The `validator.py` script is run as a batch process on these new records.
3.  **Triage and Review Assignment**:
    *   Citations marked "very likely true" (above a high confidence threshold, e.g., 80-85%) might be automatically approved or passed to a junior reviewer for a quick sanity check.
    *   Citations marked "ambiguous / review recommended," "likely untrue," or "very likely untrue" are routed to data reviewers/analysts. The confidence score and status help set the context for their review.
    *   Citations with structural issues ("no source text," "too short") are flagged for data remediation.
4.  **Manual Review and Correction**: Reviewers investigate the flagged citations, correct errors in the data values or citations, or confirm the system's findings.
5.  **Performance Monitoring**: Periodically analyze the distribution of validation statuses and confidence scores.
    *   If too many items fall into "ambiguous," the similarity thresholds might need adjustment.
    *   Track the accuracy of the "very likely true" predictions through spot-checks to build confidence in automated approvals.

### e. Quantifiable Metrics (Illustrative)

While precise quantification requires a baseline and further testing, potential metrics to track could include:
*   **Reduction in Manual Review Time per Record/Citation**: Compare time taken with and without the validation tool.
*   **Percentage of Citations Auto-Verified**: Number of citations confidently marked "very likely true" that bypass intensive manual review.
*   **Error Catch Rate**: Proportion of actual errors identified by the system (especially in "likely untrue" or "ambiguous" categories) compared to a full manual review.

By implementing this system, we can expect to achieve a more efficient, scalable, and reliable process for ensuring the accuracy of its cited data.

## Script Configuration

The validator script (`app/validator.py`) supports runtime configuration through a JSON file named `app/config.json`. This allows users to adjust key parameters without modifying the script code directly. Parameters that can be configured include:

*   **`model_name`**: Specifies the sentence-transformer model.
*   **`similarity_threshold`**: The main threshold for classifying a citation as "likely true."
*   **`min_citation_length_words`**: Filters out very short citations.
*   **`min_chunk_length_chars`**: Filters out very short source text chunks.
*   **`data_file_name`**: Defines the input data file.
*   **`output_file_name`**: Defines the output results file.

If `app/config.json` is not found, or if specific parameters are missing within it, the script will fall back to predefined default values. This ensures the script can always run and provides a clear way to manage experimental settings or operational parameters.