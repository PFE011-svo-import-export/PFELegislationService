from pydantic import BaseModel

class ChatRequest(BaseModel):
    """Schema for chat request."""
    prompt: str

class InformationMerchandise(BaseModel):
    """Schema for information on a merchanidse name
    prompt: C'est quoi le code sh pour du café non torréfié?
    """
    merchandise_name: str
    code_sh: str
    explication: str
    unite_mesure: str

class TraitementTarifiare(BaseModel):
    """"""
    traitement_tarifiaire_applicable: list[str]
    code_traitement_tarifiaire: str
    taux_tariff: float
    explication_traitement: str

class ExigencesImportation(BaseModel):
    """"""
    documents: list[str]
    normes: list[str]
    exigences: list[str]
