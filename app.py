import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title('Predicting Car Insurance claim')

st.text('')
if st.button("Claim"):
    result = predict(
        np.array([[claim_yes,claim_no]]))
    st.text(result[0])
