import json
import logging
import os
from typing import Any, BinaryIO, cast

from cdie.config import VARIABLE_NAMES

logger = logging.getLogger(__name__)


class PriorExtractor:
    """
    RAG-based module that uses OPEA TextGen to extract causal
    relationships from unstructured telecom guidelines or PDFs.
    """

    def __init__(self) -> None:
        self.opea_endpoint = os.environ.get('OPEA_LLM_ENDPOINT')
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.llm_model = os.environ.get('LLM_MODEL_ID', 'Intel/neural-chat-7b-v3-3')
        self.client = None

        if self.opea_endpoint:
            try:
                from openai import OpenAI

                self.client = OpenAI(
                    base_url=f'{self.opea_endpoint}/v1',
                    api_key=os.environ.get('OPENAI_API_KEY', 'opea-placeholder'),
                )
                logger.info(f'[PriorExtractor] OPEA TextGen connected at {self.opea_endpoint}')
            except ImportError:
                logger.warning('[PriorExtractor] OpenAI package not installed. Cannot use LLM Extractor.')
        elif self.openai_api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.openai_api_key)
                self.llm_model = 'gpt-4o-mini'
                logger.info('[PriorExtractor] OpenAI connected (fallback from OPEA)')
            except ImportError:
                logger.warning('[PriorExtractor] OpenAI package not installed.')
        else:
            logger.warning('[PriorExtractor] No LLM endpoint configured. Extraction will fail.')

    def extract_text_from_pdf(self, file_obj: BinaryIO) -> str:
        """Extract paragraph text content from a PDF file using pdfplumber."""
        try:
            import pdfplumber
        except ImportError as e:
            raise ImportError("pdfplumber is required to extract text from PDF.") from e

        full_text = []
        with pdfplumber.open(cast(Any, file_obj)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text)
        return '\n'.join(full_text)

    def extract_from_text(self, text: str) -> list[dict[str, Any]]:
        """
        Send text to OPEA TextGen to extract structured JSON causal priors.
        Only matching `VARIABLE_NAMES` are retained.
        """
        if not self.client:
            raise ValueError('LLM client not configured. Cannot extract priors.')

        # Optional: chunk text if too large (placeholder logic expects < 10k tokens)
        if len(text) > 30000:
            text = text[:30000]

        prompt = (
            f'You are the CDIE v4 Causal Prior Extractor.\n'
            f'Read the following chunk of text from a telecom guideline, and extract direct causal mechanisms.\n'
            f'Allowed Variable Nodes: {", ".join(VARIABLE_NAMES)}\n'
            f'We only care about causal relations explicitly mentioned between these nodes.\n'
            f'Return the result STRICTLY as a JSON array of objects, with no markdown formatting or extra text.\n'
            f'Schema: [{{"source": "A", "target": "B", "confidence": c}}]\n'
            f'Where confidence is between 0.0 and 1.0. High confidence must be explicit.\n'
            f'If none are found, return [].\n\n'
            f'TEXT:\n{text}'
        )

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a rigid structural causal extractor. Output only valid JSON arrays.',
                    },
                    {'role': 'user', 'content': prompt},
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content
            if content is None:
                return []
            content = content.strip()

            # Clean possible markdown block wrappers
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]

            priors = json.loads(content.strip())

            # Filter and validate against target vocabulary
            valid_priors = []
            for p in priors:
                if (
                    isinstance(p, dict)
                    and 'source' in p
                    and 'target' in p
                    and 'confidence' in p
                    and p['source'] in VARIABLE_NAMES
                    and p['target'] in VARIABLE_NAMES
                ):
                    valid_priors.append(
                        {
                            'source': p['source'],
                            'target': p['target'],
                            'confidence': float(p['confidence']),
                        }
                    )

            return valid_priors

        except Exception as e:
            logger.error(f'[PriorExtractor] Failed to extract from LLM: {e}')
            return []
