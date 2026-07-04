from pydantic import BaseModel

class TraitementTarifiare(BaseModel):
    """
    Retourne les traitements tarifiares pour un pays

    """
    traitement_tarifiaire_applicable: list[str]
    taux_tariff: float
    explication_traitement: str
    source: str