from src.methods.picr_ma import symmetric_kl

def test_sym_kl_zero_same():
    p=[0.5,0.5]
    assert float(symmetric_kl(p,p))==0.0
