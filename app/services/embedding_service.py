class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except Exception as exc:
            raise RuntimeError(
                f"Could not load embedding model '{model_name}': {exc}"
            ) from exc

    def chunk_text(self, text: str, max_tokens: int) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []

        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            first_module = getattr(self.model, "_first_module", None)
            if callable(first_module):
                tokenizer = getattr(first_module(), "tokenizer", None)

        if tokenizer is None:
            words = cleaned.split()
            if len(words) <= max_tokens:
                return [cleaned]
            return [" ".join(words[index:index + max_tokens]) for index in range(0, len(words), max_tokens)]

        token_ids = tokenizer.encode(cleaned, add_special_tokens=False, truncation=False)
        if len(token_ids) <= max_tokens:
            return [cleaned]

        chunks: list[str] = []
        for index in range(0, len(token_ids), max_tokens):
            chunk_ids = token_ids[index:index + max_tokens]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks or [cleaned]

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]
