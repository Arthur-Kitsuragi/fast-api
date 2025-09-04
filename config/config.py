from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    bert_model_name: str = "bert-base-uncased"
    lstm_units: int  = 128
    dense_units: int = 256
    num_classes: int = 20
    nhead: int = 4
    num_layers: int = 1
    dropout: float =0.5
    class Config:
        env_file = ".env"