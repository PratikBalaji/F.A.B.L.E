from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # OpenRouter (unified LLM gateway)
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Models routed via OpenRouter
    primary_model: str = "anthropic/claude-sonnet-4-5"    # analyst (deep reasoning)
    secondary_model: str = "openai/gpt-4o-mini"           # critic + synthesizer (cheap)
    critic_model: str = "openai/gpt-4o-mini"

    # Adversarial pipeline — per-role model assignments
    planner_model: str = "anthropic/claude-sonnet-4-5"    # Claude: structured decomposition
    actor_model: str = "openai/gpt-4o"                    # GPT-4o: primary generation
    adv_critic_model: str = "meta-llama/llama-3-70b-instruct"   # Groq: fast adversarial
    validator_model: str = "google/gemini-pro-1.5"        # Gemini: large-context validation
    refiner_model: str = "meta-llama/llama-3-70b-instruct"      # Groq: fast directives
    judge_model: str = "anthropic/claude-sonnet-4-5"      # Claude: final arbitration
    adversarial_max_rounds: int = 2                       # default 2 to conserve credits
    adversarial_judge_threshold: float = 0.80             # score >= 0.80 → ACCEPT

    # RAG
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_store_path: str = "./data/vectorstore"
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 5

    # AWS
    aws_region: str = "us-east-1"
    s3_bucket: str = "fable-rag-docs"

    # App
    log_level: str = "INFO"
    feedback_db_path: str = "./data/feedback.jsonl"
    app_name: str = "F.A.B.L.E."
    app_url: str = "http://localhost:3000"

    # --- Multi-user platform (Supabase) ---------------------------------
    # Feature flag: when False the app runs in legacy single-user/file mode.
    use_supabase: bool = Field(default=False, alias="USE_SUPABASE")

    supabase_url: str = Field(default="", alias="SUPABASE_URL")
    supabase_anon_key: str = Field(default="", alias="SUPABASE_ANON_KEY")
    supabase_service_role_key: str = Field(default="", alias="SUPABASE_SERVICE_ROLE_KEY")

    # JWT verification. Default path: asymmetric JWKS (no shared secret).
    # jwks_url defaults to "{supabase_url}/auth/v1/.well-known/jwks.json" when blank.
    supabase_jwks_url: str = Field(default="", alias="SUPABASE_JWKS_URL")
    # Legacy fallback for HS256 projects; only used when use_jwks=False.
    supabase_jwt_secret: str = Field(default="", alias="SUPABASE_JWT_SECRET")
    use_jwks: bool = Field(default=True, alias="USE_JWKS")
    jwt_audience: str = "authenticated"

    # Provider credential encryption (AES-256-GCM). 32-byte key, base64-encoded.
    # Generate: python -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())"
    app_encryption_key: str = Field(default="", alias="APP_ENCRYPTION_KEY")

    # OpenRouter OAuth (PKCE). Callback must be HTTPS on port 443 or 3000.
    openrouter_oauth_callback: str = Field(
        default="https://localhost:3000/auth/openrouter/callback",
        alias="OPENROUTER_OAUTH_CALLBACK",
    )
    openrouter_auth_url: str = "https://openrouter.ai/auth"
    openrouter_key_exchange_url: str = "https://openrouter.ai/api/v1/auth/keys"

    @property
    def resolved_jwks_url(self) -> str:
        if self.supabase_jwks_url:
            return self.supabase_jwks_url
        if self.supabase_url:
            return f"{self.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
        return ""


settings = Settings()
