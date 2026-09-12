from pydantic import BaseModel


'''class AlphaProposal(BaseModel):
    w1: float
    w2: float
    thesis: str'''

class AlphaProposal(BaseModel):
    feature_1: str
    feature_2: str
    operation: str
    w1: float = 1
    w2: float = 1
    thesis: str

'''class AlphaCritique(BaseModel):
    weakness: str
    suggested_w1: float
    suggested_w2: float
    reasoning: str'''

class AlphaCritique(BaseModel):
    weakness: str
    suggested_feature_1: str
    suggested_feature_2: str
    suggested_operation: str
    suggested_w1: float
    suggested_w2: float
    reasoning: str