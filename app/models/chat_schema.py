from pydantic import BaseModel

class ChatRequest(BaseModel):
    """Schema for chat request."""
    prompt: str

class InformationMerchandise(BaseModel):
    """
    Schema for information on a merchanidse name
    prompt: C'est quoi le code sh pour du café non torréfié?
    """
    merchandise_name: str
    code_sh: str
    explication: str
    unite_mesure: str

class Exigence(BaseModel):
    """Une exigence d'importation portant sur un sujet précis
    (ex: Emballage de Bois, Justification de l'origine, Marquage / Étiquetage)."""
    sujet: str
    description: str
    documents_a_fournir: list[str]

class ExigencesImportation(BaseModel):
    """
    Liste des exigences que l'exportateur doit respecter pour importer un produit au Canada.
    """
    marchandise: str
    pays_origine: str
    exigences: list[Exigence]
